"""
HRHC (高级参考，高级控制) 控制器模块。
该模块实现了一个二级控制器：
1. 离散轨迹搜索 (Trim Library)，寻找进度最大的安全名义轨迹。
2. 非线性 MPC 跟踪层，用于精确跟踪该名义轨迹。
"""
from types import SimpleNamespace
import casadi as ca
import numpy as np

class HRHC:
    """
    HRHC 控制器类。
    结合了高效的轨迹搜索和基于 CasADi 的非线性优化跟踪。
    """
    def __init__(self, vehicle_model, track):
        self.model = vehicle_model
        self.track = track
        self.use_tracking_fallback = True
        self.st8 = SimpleNamespace(last_x=None, last_u=None)

        # 控制参数
        self.params = {
            'n_horizon': 20,
            'dt': 0.05,
            'trim_v': np.linspace(1.0, 8.0, 6),
            'trim_delta': np.linspace(-0.35, 0.35, 9),
            'safety_margin': 0.8,
            'planner_violation_weight': 50.0
        }

        # 离散仿真函数
        self.f_sim = self.model.get_discrete_model(self.params['dt'])

        # NLP 符号及函数
        self.nlp = {}

        # 设置 LQP (线性/二次规划跟踪层)
        self._setup_lqp()

    def _evaluate_trim(self, state0, v_target, delta_target):
        """前向仿真车辆，保持速度和转向角恒定。"""
        n_h = self.params['n_horizon']
        traj_x = np.zeros((self.model.nx, n_h + 1))
        traj_u = np.zeros((self.model.nu, n_h))

        traj_x[:, 0] = state0
        curr_x = state0

        for k in range(n_h):
            # 比例节气门控制，维持目标速度
            throttle = min(1.0, max(-1.0, 2.0 * (v_target - curr_x[3])))
            u_k = np.array([throttle, delta_target])
            traj_u[:, k] = u_k

            # 仿真动力学步进
            curr_x = self.f_sim(curr_x, u_k).full().flatten()
            traj_x[:, k + 1] = curr_x

        return traj_x, traj_u

    def _evaluate_trim_safety(self, traj, start_theta):
        """评估轨迹安全性，并返回最大越界量。"""
        max_violation = 0.0
        for k in range(1, self.params['n_horizon'] + 1, 4):
            pos_x, pos_y = traj[0, k], traj[1, k]
            theta_k = self.track.project_to_centerline(pos_x, pos_y, start_theta)
            res = self.track.f_theta(theta_k)
            t_pos = res[0].full().flatten()
            t_w = float(res[2].full().flatten()[0])

            dist = np.sqrt((pos_x - t_pos[0])**2 + (pos_y - t_pos[1])**2)
            max_violation = max(max_violation, dist - t_w * self.params['safety_margin'])
        return max_violation <= 0.0, max_violation

    def _is_trim_safe(self, traj, start_theta):
        """检查轨迹是否在赛道边界内。"""
        is_safe, _ = self._evaluate_trim_safety(traj, start_theta)
        return is_safe

    def _get_best_trim(self, current_state, current_theta):
        """在线前向仿真轨迹库，寻找进度最大化的最优路径。"""
        best_progress = -np.inf
        best_traj = (None, None)
        best_relaxed_score = -np.inf
        best_relaxed = (None, None)
        self.last_plan = {
            'used_relaxed': False,
            'safe_count': 0,
            'best_violation': None
        }

        for vel in self.params['trim_v']:
            for delta in self.params['trim_delta']:
                traj_x, traj_u = self._evaluate_trim(current_state, vel, delta)
                t_end = self.track.project_to_centerline(
                    traj_x[0, -1], traj_x[1, -1], current_theta
                )
                progress = (t_end - current_theta) - 0.05 * abs(delta)
                is_safe, violation = self._evaluate_trim_safety(traj_x, current_theta)

                if is_safe:
                    self.last_plan['safe_count'] += 1
                    if progress > best_progress:
                        best_progress = progress
                        best_traj = (traj_x, traj_u)
                else:
                    relaxed_score = progress - self.params['planner_violation_weight'] * violation
                    if relaxed_score > best_relaxed_score:
                        best_relaxed_score = relaxed_score
                        best_relaxed = (traj_x, traj_u)
                        self.last_plan['best_violation'] = violation

        if best_traj[0] is None:
            if best_relaxed[0] is not None:
                self.last_plan['used_relaxed'] = True
                return best_relaxed
            # 降级方案：安全直线制动
            self.last_plan['used_relaxed'] = True
            return self._evaluate_trim(current_state, 0.0, 0.0)

        return best_traj

    def _set_initial_guess(self, best_x, best_u):
        """为跟踪层设置初值，优先复用上一轮解的平移结果。"""
        if self.st8.last_x is None or self.st8.last_u is None:
            self.nlp['opti'].set_initial(self.nlp['x'], best_x)
            self.nlp['opti'].set_initial(self.nlp['u'], best_u)
            return

        warm_x = np.copy(best_x)
        warm_u = np.copy(best_u)
        warm_x[:, :-1] = self.st8.last_x[:, 1:]
        warm_x[:, -1] = best_x[:, -1]
        warm_u[:, :-1] = self.st8.last_u[:, 1:]
        warm_u[:, -1] = best_u[:, -1]

        self.nlp['opti'].set_initial(self.nlp['x'], warm_x)
        self.nlp['opti'].set_initial(self.nlp['u'], warm_u)

    def _setup_lqp(self):
        """设置非线性 MPC 跟踪问题的符号公式和约束。"""
        opti = ca.Opti()
        n_h = self.params['n_horizon']

        sym_x = opti.variable(self.model.nx, n_h + 1)
        sym_u = opti.variable(self.model.nu, n_h)

        p_x0 = opti.parameter(self.model.nx)
        p_xref = opti.parameter(self.model.nx, n_h + 1)
        p_uref = opti.parameter(self.model.nu, n_h)

        opti.subject_to(sym_x[:, 0] == p_x0)

        cost = 0
        for k in range(n_h):
            # 动力学约束
            opti.subject_to(sym_x[:, k + 1] == self.f_sim(sym_x[:, k], sym_u[:, k]))

            # 跟踪代价 (位置、航向、速度等惩罚项)
            cost += 100.0 * ca.sumsqr(sym_x[0:2, k] - p_xref[0:2, k])
            cost += 10.0 * (sym_x[2, k] - p_xref[2, k])**2
            cost += 10.0 * (sym_x[3, k] - p_xref[3, k])**2
            cost += 1.0 * (sym_u[0, k] - p_uref[0, k])**2
            cost += 10.0 * (sym_u[1, k] - p_uref[1, k])**2

            # 输入限制
            opti.subject_to(opti.bounded(-1.0, sym_u[0, k], 1.0))
            opti.subject_to(opti.bounded(-0.4, sym_u[1, k], 0.4))

        opti.minimize(cost)
        opti.solver("ipopt", {"expand": True, "print_time": 0},
                    {"max_iter": 400, "print_level": 0, "sb": "yes",
                     "acceptable_tol": 1e-4, "acceptable_iter": 5})

        self.nlp = {'opti': opti, 'x': sym_x, 'u': sym_u,
                    'p_x0': p_x0, 'p_xref': p_xref, 'p_uref': p_uref}

    def solve(self, current_state, current_theta):
        """
        根据当前状态和进度，求解 HRHC 控制量。
        
        返回:
            u_opt: 优化后的控制指令 [throttle, steering]
            proj_theta: 投影后的当前进度值
        """
        proj_theta = self.track.project_to_centerline(
            current_state[0], current_state[1], current_theta
        )

        # 寻找最优 TRIM 轨迹作为参考
        best_x, best_u = self._get_best_trim(current_state, proj_theta)

        # 设置 NLP 参数和初值
        self.nlp['opti'].set_value(self.nlp['p_x0'], current_state)
        self.nlp['opti'].set_value(self.nlp['p_xref'], best_x)
        self.nlp['opti'].set_value(self.nlp['p_uref'], best_u)
        self._set_initial_guess(best_x, best_u)

        try:
            sol = self.nlp['opti'].solve()
            self.st8.last_x = sol.value(self.nlp['x'])
            self.st8.last_u = sol.value(self.nlp['u'])
            u_opt = sol.value(self.nlp['u'][:, 0])
            return u_opt, proj_theta
        except Exception as error:
            if self.use_tracking_fallback:
                self.st8.last_x = best_x
                self.st8.last_u = best_u
                return best_u[:, 0], proj_theta, True
            raise RuntimeError(f"HRHC 非线性 MPC 跟踪失败: {str(error)}") from error

    def get_name(self):
        """返回控制器名称"""
        return "HRHC"
