"""
动力学自行车模型模块。
该模块实现了用于控制和仿真的动力学自行车模型。
"""
import casadi as ca

class DynamicBicycleModel:
    """
    动力学自行车模型类。
    包含了车辆的物理参数、轮胎 Pacejka 模型以及电机模型。
    """
    def __init__(self):
        # 车辆物理及模型参数 (针对 1:43 RC 小车的近似)
        self.params = {
            'mass': 0.167,      # 质量 (kg)
            'iz': 0.00025,      # 转动惯量 (kg m^2)
            'lf': 0.045,        # 前轴到重心距离 (m)
            'lr': 0.045,        # 后轴到重心距离 (m)
            'p_b': 4.0,         # Pacejka 参数 B
            'p_c': 1.3,         # Pacejka 参数 C
            'p_d': 0.3,         # Pacejka 参数 D
            'cm1': 1.5,         # 电机增益
            'cm2': 0.15         # 电机阻尼因子
        }

        # 状态空间维度
        self.nx = 6
        self.nu = 2

        # 下面定义的 CasADi 符号和函数在 _setup_casadi_model 中初始化
        self.x_sym = None
        self.u_sym = None
        self.rhs = None
        self.f_func = None

        # 构建连续时间模型
        self._setup_casadi_model()

    def _setup_casadi_model(self):
        """初始化 CasADi 连续时间动力学方程"""
        # 状态量和控制输入符号
        self.x_sym = ca.SX.sym('x', self.nx)
        self.u_sym = ca.SX.sym('u', self.nu)

        # 1. 运动学关系
        phi = self.x_sym[2]
        v_x, v_y = self.x_sym[3], self.x_sym[4]

        x_dot = v_x * ca.cos(phi) - v_y * ca.sin(phi)
        y_dot = v_x * ca.sin(phi) + v_y * ca.cos(phi)

        # 2. 动力学计算
        vx_dot, vy_dot, omega_dot = self._compute_dynamics(self.x_sym, self.u_sym)

        self.rhs = ca.vertcat(x_dot, y_dot, self.x_sym[5], vx_dot, vy_dot, omega_dot)

        # 连续时间模型函数
        self.f_func = ca.Function('f', [self.x_sym, self.u_sym], [self.rhs], ['x', 'u'], ['rhs'])

    def _get_tire_forces(self, v_x, v_y, omega, delta):
        """计算 Pacejka 轮胎侧向力"""
        # 侧偏角计算 (使用 epsilon 避免除零)
        eps = 1e-4
        v_x_s = ca.if_else(ca.fabs(v_x) < eps, eps * ca.sign(v_x + 1e-9), v_x)

        alpha_f = delta - ca.atan((v_y + self.params['lf'] * omega) / v_x_s)
        alpha_r = - ca.atan((v_y - self.params['lr'] * omega) / v_x_s)

        # Pacejka 公式
        f_yf = self.params['p_d'] * ca.sin(
            self.params['p_c'] * ca.atan(self.params['p_b'] * alpha_f)
        )
        f_yr = self.params['p_d'] * ca.sin(
            self.params['p_c'] * ca.atan(self.params['p_b'] * alpha_r)
        )
        return f_yf, f_yr

    def _compute_dynamics(self, x, u):
        """核心物理动力学计算，包括胎力及电机模型"""
        # 1. 轮胎侧向力
        f_yf, f_yr = self._get_tire_forces(x[3], x[4], x[5], u[1])

        # 2. 纵向力 (电机模型)
        f_rx = self.params['cm1'] * u[0] - self.params['cm2'] * x[3]

        # 3. 加速度计算
        # vx_dot = (F_rx - F_yf * sin(delta) + m * vy * omega) / m
        vx_dot = (f_rx - f_yf * ca.sin(u[1]) + self.params['mass'] * x[4] * x[5]) / \
                 self.params['mass']

        # vy_dot = (F_yr + F_yf * cos(delta) - m * vx * omega) / m
        vy_dot = (f_yr + f_yf * ca.cos(u[1]) - self.params['mass'] * x[3] * x[5]) / \
                 self.params['mass']

        # omega_dot = (F_yf * cos(delta) * lf - F_yr * lr) / iz
        om_dot = (f_yf * ca.cos(u[1]) * self.params['lf'] -
                  f_yr * self.params['lr']) / self.params['iz']

        return vx_dot, vy_dot, om_dot

    def get_discrete_model(self, dt_val):
        """
        使用 RK4 方法离散化模型。
        
        参数:
            dt_val: 步长 (s)
            
        返回:
            离散时间转换函数 F(x, u) -> x_next
        """
        x_in = ca.SX.sym('x_in', self.nx)
        u_in = ca.SX.sym('u_in', self.nu)

        k1 = self.f_func(x_in, u_in)
        k2 = self.f_func(x_in + dt_val / 2.0 * k1, u_in)
        k3 = self.f_func(x_in + dt_val / 2.0 * k2, u_in)
        k4 = self.f_func(x_in + dt_val * k3, u_in)

        x_next = x_in + dt_val / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        return ca.Function('F', [x_in, u_in], [x_next], ['x', 'u'], ['x_next'])

    def get_discrete_jacobians(self, dt_val):
        """
        获取离散模型的雅可比矩阵 A_k 和 B_k。
        用于构建 LTV-QP 每步的状态空间线性化。
        
        参数:
            dt_val: 步长 (s)
            
        返回:
            函数 [x_bar, u_bar] -> [A, B, x_next]
        """
        x_bar = ca.SX.sym('x_bar', self.nx)
        u_bar = ca.SX.sym('u_bar', self.nu)

        f_dis = self.get_discrete_model(dt_val)
        x_next = f_dis(x_bar, u_bar)

        jac_a = ca.jacobian(x_next, x_bar)
        jac_b = ca.jacobian(x_next, u_bar)

        return ca.Function('discrete_jac', [x_bar, u_bar], [jac_a, jac_b, x_next],
                           ['x_bar', 'u_bar'], ['A', 'B', 'f_val'])
