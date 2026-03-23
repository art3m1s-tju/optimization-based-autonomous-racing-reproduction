"""
MPCC (模型预测轮廓控制) 模块。
基于 LTV-QP (线性时变二次规划) 和 RTI (实时迭代) 分离方案实现的赛车控制器。
该控制器专门针对高性能轨迹跟踪设计，通过最大化进度同时最小化横向和纵向误差。
"""
from contextlib import contextmanager
import json
import os
from types import SimpleNamespace
import casadi as ca
import numpy as np

class MPCC:
    """
    MPCC 控制器类。
    实现了显式线性化、状态转移 Jacobian 提取以及基于误差 AD 的赛道边界约束。
    """
    def __init__(self, model, track, horizon=40, dt_val=0.05):
        self.model = model
        self.track = track

        # 系统配置与运行状态
        self.cfg = SimpleNamespace(n_h=horizon, dt=dt_val, max_ld=track.max_lapdist)
        self.st8 = SimpleNamespace(init=False, last_x=None, last_u=None)
        # 下面这些命名直接对应论文中的 MPCC 代价与约束结构，方便后续校核。
        self.weights = SimpleNamespace(
            q_c=20.0,          # 轮廓误差权重，对应论文中的 contouring cost
            q_l=15.0,          # 滞后误差权重，对应论文中的 lag cost
            gamma=1.0,         # 进度奖励，对应 -gamma * v
            q_slack_lin=10000.0,
            q_slack_quad=10.0,
            q_reg=1e-4
        )
        self.bounds = SimpleNamespace(
            track_margin=0.8,
            throttle=(-1.0, 1.0),
            steering=(-0.4, 0.4),
            progress_rate=(0.0, 15.0),
            delta_throttle=(-0.2, 0.2),
            delta_steering=(-0.1, 0.1),
            delta_progress=(-2.0, 2.0),
            speed=(0.0, 5.0)
        )
        self.solver_opts = {"printLevel": "none", "error_on_fail": False}
        self.silent_solver_output = True

        # 核心算子
        self.f_sim = self.model.get_discrete_model(self.cfg.dt)
        self.qp_vars = {}
        self.err_func = None

        # 构建 LTV-QP 命题
        self._setup_nlp()

    @staticmethod
    def _dm_to_numpy(data):
        """将 CasADi 数据统一转成 numpy 数组。"""
        return np.array(data.full() if hasattr(data, 'full') else data, dtype=float)

    def _define_params(self, opti, nx_dim, nu_dim):
        """定义决策变量和参数"""
        n_h = self.cfg.n_h
        # 决策变量：LTV 求解的状态增量 dX 和输入增量 dU
        d_x = opti.variable(nx_dim, n_h + 1)
        d_u = opti.variable(nu_dim, n_h)

        # Opti 参数：用于每步热启动的线性化点 (x_bar, u_bar) 和当前状态 x0
        p_x_bar = opti.parameter(nx_dim, n_h + 1)
        p_u_bar = opti.parameter(nu_dim, n_h)
        p_x0 = opti.parameter(nx_dim)

        self.qp_vars.update({
            'd_x': d_x, 'd_u': d_u,
            'p_x_bar': p_x_bar, 'p_u_bar': p_u_bar, 'p_x0': p_x0
        })

    def _setup_error_functions(self, nx_dim):
        """利用符号微分构建中心线几何误差函数 (Contour/Lag Error)"""
        x_sym = ca.SX.sym('x_sym', nx_dim)

        # 获取赛道几何：中心线位置、切向角、半宽
        pos_c, phi_c, t_w = self.track.f_theta(x_sym[6])

        # 轮廓误差 (Contour) 和 滞后误差 (Lag) 定义
        e_c = ca.sin(phi_c) * (x_sym[0] - pos_c[0]) - ca.cos(phi_c) * (x_sym[1] - pos_c[1])
        e_l = -ca.cos(phi_c) * (x_sym[0] - pos_c[0]) - ca.sin(phi_c) * (x_sym[1] - pos_c[1])

        # 对状态量求显式 Jacobian
        self.err_func = ca.Function('err_jac', [x_sym],
                                    [e_c, e_l, ca.jacobian(e_c, x_sym),
                                     ca.jacobian(e_l, x_sym), t_w])

    def _add_stage_constraints(self, opti, f_jac, k):
        """添加 LTV 动力学、赛道边界以及控制约束"""
        d_x, d_u = self.qp_vars['d_x'], self.qp_vars['d_u']
        p_xb, p_ub = self.qp_vars['p_x_bar'], self.qp_vars['p_u_bar']
        bounds = self.bounds

        # 1. LTV 动力学与进度积分
        a_k, b_k, f_v = f_jac(p_xb[:6, k], p_ub[:2, k])
        opti.subject_to(p_xb[:6, k+1] + d_x[:6, k+1] ==
                        f_v + a_k @ d_x[:6, k] + b_k @ d_u[:2, k])
        opti.subject_to(p_xb[6, k+1] + d_x[6, k+1] ==
                        p_xb[6, k] + d_x[6, k] + self.cfg.dt * (p_ub[2, k] + d_u[2, k]))

        # 2. 线性化赛道边界
        res_e = self.err_func(p_xb[:, k])
        e_c_tilde = res_e[0] + res_e[2] @ d_x[:, k]

        slack = opti.variable()
        opti.subject_to(slack >= 0)
        bound_limit = bounds.track_margin * res_e[4]
        opti.subject_to(e_c_tilde <= bound_limit + slack)
        opti.subject_to(e_c_tilde >= -bound_limit - slack)

        # 3. 控制量及信赖域约束
        u_k = p_ub[:, k] + d_u[:, k]
        opti.subject_to(opti.bounded(bounds.throttle[0], u_k[0], bounds.throttle[1]))
        opti.subject_to(opti.bounded(bounds.steering[0], u_k[1], bounds.steering[1]))
        opti.subject_to(opti.bounded(bounds.progress_rate[0], u_k[2], bounds.progress_rate[1]))
        opti.subject_to(opti.bounded(bounds.delta_throttle[0], d_u[0, k],
                                     bounds.delta_throttle[1]))
        opti.subject_to(opti.bounded(bounds.delta_steering[0], d_u[1, k],
                                     bounds.delta_steering[1]))
        opti.subject_to(opti.bounded(bounds.delta_progress[0], d_u[2, k],
                                     bounds.delta_progress[1]))

        return e_c_tilde, slack, u_k

    def _setup_nlp(self):
        """构建 LTV-QP RTI 架构的数学命题"""
        opti = ca.Opti('conic')
        self._define_params(opti, self.model.nx + 1, self.model.nu + 1)
        self._setup_error_functions(self.model.nx + 1)

        opti.subject_to(self.qp_vars['d_x'][:, 0] ==
                        self.qp_vars['p_x0'] - self.qp_vars['p_x_bar'][:, 0])

        weights = self.weights
        cost = weights.q_reg * (ca.sumsqr(self.qp_vars['d_x']) + ca.sumsqr(self.qp_vars['d_u']))

        f_jac = self.model.get_discrete_jacobians(self.cfg.dt)
        for k in range(self.cfg.n_h):
            e_c_t, slack, u_k = self._add_stage_constraints(opti, f_jac, k)
            _, e_l_b, _, j_el, _ = self.err_func(self.qp_vars['p_x_bar'][:, k])

            cost += weights.q_c * e_c_t**2 + \
                    weights.q_l * (e_l_b + j_el @ self.qp_vars['d_x'][:, k])**2 - \
                    weights.gamma * u_k[2] + \
                    weights.q_slack_lin * slack + \
                    weights.q_slack_quad * slack**2

            if k > 0:
                opti.subject_to(opti.bounded(self.bounds.speed[0],
                                             (self.qp_vars['p_x_bar'][3, k] +
                                              self.qp_vars['d_x'][3, k]),
                                             self.bounds.speed[1]))

        opti.subject_to(opti.bounded(self.bounds.speed[0],
                                     (self.qp_vars['p_x_bar'][3, self.cfg.n_h] +
                                      self.qp_vars['d_x'][3, self.cfg.n_h]),
                                     self.bounds.speed[1]))

        opti.minimize(cost)
        opti.solver("qpoases", self.solver_opts)
        self.qp_vars['opti'] = opti

    def _warm_start(self, curr_full):
        """热启动外插"""
        n_h = self.cfg.n_h
        x_bar = np.zeros((self.model.nx + 1, n_h + 1))
        u_bar = np.zeros((self.model.nu + 1, n_h))

        if self.st8.init and self.st8.last_x is not None:
            x_bar[:, :-1] = self.st8.last_x[:, 1:]
            u_bar[:, :-1] = self.st8.last_u[:, 1:]
            u_bar[:, -1] = u_bar[:, -2]
            x_bar[:6, -1] = self.f_sim(x_bar[:6, -2], u_bar[:2, -1]).full().flatten()
            x_bar[6, -1] = x_bar[6, -2] + self.cfg.dt * u_bar[2, -1]
        else:
            v_i = max(0.5, curr_full[3])
            for k in range(n_h + 1):
                dist = v_i * k * self.cfg.dt
                x_bar[:, k] = curr_full + [dist * np.cos(curr_full[2]),
                                           dist * np.sin(curr_full[2]), 0, 0, 0, 0, dist]
            u_bar[:, :] = np.array([0.0, 0.0, v_i])[:, None]
            self.st8.init = True
        return x_bar, u_bar

    def build_stage_qp_data(self, current_state, current_theta, x_bar=None, u_bar=None):
        """
        导出结构化 QP 所需的逐阶段线性化数据。

        该接口不改变当前求解器，只是把论文中多阶段 QP 的核心数据显式暴露出来，
        便于后续切换到代码生成或结构化求解器。
        """
        curr_full = np.concatenate([current_state, [current_theta]])
        if x_bar is None or u_bar is None:
            x_bar, u_bar = self._warm_start(curr_full)

        f_jac = self.model.get_discrete_jacobians(self.cfg.dt)
        stages = []
        for k in range(self.cfg.n_h):
            a_k, b_k, f_v = f_jac(x_bar[:6, k], u_bar[:2, k])
            err_res = self.err_func(x_bar[:, k])

            stages.append({
                "k": k,
                "A": self._dm_to_numpy(a_k),
                "B": self._dm_to_numpy(b_k),
                "f": self._dm_to_numpy(f_v).reshape(-1),
                "x_bar": np.array(x_bar[:, k], dtype=float),
                "u_bar": np.array(u_bar[:, k], dtype=float),
                "contour_base": float(err_res[0]),
                "lag_base": float(err_res[1]),
                "contour_jac": self._dm_to_numpy(err_res[2]).reshape(-1),
                "lag_jac": self._dm_to_numpy(err_res[3]).reshape(-1),
                "track_half_width": float(err_res[4]),
                "track_bound": float(self.bounds.track_margin * err_res[4])
            })

        return {
            "dt": self.cfg.dt,
            "horizon": self.cfg.n_h,
            "x0": curr_full,
            "x_bar": np.array(x_bar, dtype=float),
            "u_bar": np.array(u_bar, dtype=float),
            "weights": vars(self.weights),
            "bounds": vars(self.bounds),
            "stages": stages,
            "terminal_speed_bounds": list(self.bounds.speed)
        }

    def export_kernel_metadata(self, output_dir):
        """导出结构化 QP 和代码生成需要的静态元数据。"""
        metadata = {
            "controller": "MPCC",
            "dt": self.cfg.dt,
            "horizon": self.cfg.n_h,
            "nx_aug": self.model.nx + 1,
            "nu_aug": self.model.nu + 1,
            "weights": vars(self.weights),
            "bounds": vars(self.bounds)
        }
        out_path = os.path.join(output_dir, "mpcc_metadata.json")
        with open(out_path, "w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2, ensure_ascii=False)
        return out_path

    def _solve_qp(self, x_bar, u_bar, curr_full):
        """执行 QP 求解"""
        opti = self.qp_vars['opti']
        opti.set_value(self.qp_vars['p_x_bar'], x_bar)
        opti.set_value(self.qp_vars['p_u_bar'], u_bar)
        opti.set_value(self.qp_vars['p_x0'], curr_full)

        opti.set_initial(self.qp_vars['d_x'], 0)
        opti.set_initial(self.qp_vars['d_u'], 0)

        try:
            with self._mute_solver_output():
                sol = opti.solve()
            dx_res, du_res = sol.value(self.qp_vars['d_x']), sol.value(self.qp_vars['d_u'])
        except RuntimeError:
            dx_res, du_res = opti.debug.value(self.qp_vars['d_x']), \
                             opti.debug.value(self.qp_vars['d_u'])

        self.st8.last_x, self.st8.last_u = x_bar + dx_res, u_bar + du_res
        return self.st8.last_u[:, 0], self.st8.last_x[6, 1], False

    @contextmanager
    def _mute_solver_output(self):
        """静默底层 QP 求解器的 C 层 stdout/stderr 输出。"""
        if not self.silent_solver_output:
            yield
            return

        stdout_fd = os.dup(1)
        stderr_fd = os.dup(2)
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            try:
                os.dup2(devnull.fileno(), 1)
                os.dup2(devnull.fileno(), 2)
                yield
            finally:
                os.dup2(stdout_fd, 1)
                os.dup2(stderr_fd, 2)
                os.close(stdout_fd)
                os.close(stderr_fd)

    def _fallback_recovery(self, curr_full, x_b, u_b):
        """紧急恢复程序"""
        rec_x, rec_u = np.zeros_like(x_b), np.zeros_like(u_b)
        rec_x[:, 0] = curr_full
        for k in range(self.cfg.n_h):
            e_c_v = float(self.err_func(rec_x[:, k])[0])
            steer = max(-0.4, min(0.4, -0.5 * e_c_v))
            brake = -1.0 if rec_x[3, k] > 1.0 else 0.0
            rec_u[:, k] = [brake, steer, max(0.0, rec_x[3, k])]
            rec_x[:6, k+1] = self.f_sim(rec_x[:6, k], rec_u[:2, k]).full().flatten()
            rec_x[6, k+1] = rec_x[6, k] + self.cfg.dt * rec_u[2, k]
        self.st8.last_x, self.st8.last_u = rec_x, rec_u
        self.st8.init = False
        return rec_u[:, 0], curr_full[6], True

    def solve(self, current_state, current_theta):
        """求解入口"""
        curr_full = np.concatenate([current_state, [current_theta]])
        x_bar, u_bar = self._warm_start(curr_full)
        try:
            return self._solve_qp(x_bar, u_bar, curr_full)
        except Exception as error:  # pylint: disable=broad-except
            print(f"[MPCC] QP 彻底失效: {error}")
            return self._fallback_recovery(curr_full, x_bar, u_bar)

    def get_prediction(self):
        """获取预测轨迹"""
        return self.st8.last_x if self.st8.last_x is not None else None

    def get_name(self):
        """返回控制器名称"""
        return "MPCC"
