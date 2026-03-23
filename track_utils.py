"""
赛道工具模块。
该模块负责加载赛道数据，构建样条曲线，并提供点到中心线的投影功能。
"""
import pandas as pd
import numpy as np
import casadi as ca
import scipy.optimize

class Track:
    """
    赛道类。
    用于管理赛道几何信息，包括中心线样条和宽度样条。
    """
    def __init__(self, csv_path, scale=1/43.0):
        # 加载并缩放赛道数据
        df_raw = pd.read_csv(csv_path)
        if df_raw['Lapdist'].iloc[-1] <= df_raw['Lapdist'].iloc[-2]:
            df_raw = df_raw.iloc[:-1]

        # 核心几何数据 (缩放后)
        self.geo = {
            'lapdist': df_raw['Lapdist'].values * scale,
            'pos_x': df_raw['pos_x'].values * scale,
            'pos_y': df_raw['pos_y'].values * scale,
            'left_x': df_raw['left_border_x'].values * scale,
            'left_y': df_raw['left_border_y'].values * scale,
            'right_x': df_raw['right_border_x'].values * scale,
            'right_y': df_raw['right_border_y'].values * scale,
            'width': np.sqrt((df_raw['left_border_x'] - df_raw['right_border_x'])**2 +
                             (df_raw['left_border_y'] - df_raw['right_border_y'])**2).values *
                     scale / 2.0
        }

        self.max_lapdist = self.geo['lapdist'][-1]

        # 样条曲线函数
        self.splines = self._init_splines()

        # 生成 CasADi 函数
        self.f_theta = self._get_casadi_functions()

    def _init_splines(self):
        """初始化基于距离的 B 样条插值"""
        _, unique_idx = np.unique(self.geo['lapdist'], return_index=True)
        unique_idx = np.sort(unique_idx)

        ld_u = self.geo['lapdist'][unique_idx]
        px_u = self.geo['pos_x'][unique_idx]
        py_u = self.geo['pos_y'][unique_idx]
        w_u = self.geo['width'][unique_idx]

        return {
            'x': ca.interpolant('x_s', 'bspline', [ld_u], px_u),
            'y': ca.interpolant('y_s', 'bspline', [ld_u], py_u),
            'w': ca.interpolant('w_s', 'bspline', [ld_u], w_u)
        }

    def _get_casadi_functions(self):
        """构建 CasADi 符号函数，用于位置、切向角和宽度的实时计算"""
        theta = ca.SX.sym('theta')
        pos_x = self.splines['x'](theta)
        pos_y = self.splines['y'](theta)

        # 使用自动微分计算切向角
        dx_dt = ca.jacobian(pos_x, theta)
        dy_dt = ca.jacobian(pos_y, theta)
        phi_c = ca.atan2(dy_dt, dx_dt)

        # 赛道半宽
        width = self.splines['w'](theta)

        return ca.Function('f_theta', [theta], [ca.vertcat(pos_x, pos_y), phi_c, width],
                           ['theta'], ['pos', 'phi_c', 'width'])

    def project_to_centerline(self, pos_x, pos_y, theta_guess):
        """
        将全局坐标 (x, y) 投影到中心线。
        使用最小化二乘法寻找最近的 theta (进度值)。
        """
        def dist_sq(theta):
            t_eval = float(theta) % self.max_lapdist
            eval_res = self.f_theta(t_eval)
            px_c = float(eval_res[0].full().flatten()[0])
            py_c = float(eval_res[0].full().flatten()[1])
            return (px_c - pos_x)**2 + (py_c - pos_y)**2

        res = scipy.optimize.minimize_scalar(
            dist_sq,
            bounds=(theta_guess - 5.0, theta_guess + 15.0),
            method='bounded'
        )

        return res.x % self.max_lapdist if res.success else theta_guess

    def get_curvature(self, theta):
        """计算给定进度 theta 处的近似曲率"""
        t_s = ca.SX.sym('t_s')
        phi = self.f_theta(t_s)[1]
        kappa_f = ca.Function('kappa', [t_s], [ca.jacobian(phi, t_s)])
        return float(kappa_f(theta % self.max_lapdist))
