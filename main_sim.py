#!/usr/bin/env python3
"""
主仿真模块。
用于运行 HRHC 或 MPCC 控制器的闭环仿真，并可视化车辆轨迹和赛道边界。
"""
import argparse
from types import SimpleNamespace
import numpy as np
import matplotlib.pyplot as plt

from dynamic_model import DynamicBicycleModel
from track_utils import Track
from hrhc_controller import HRHC
from mpcc_controller import MPCC

def get_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='运行车辆轨迹跟踪仿真')
    parser.add_argument('--mode', type=str, default='mpcc',
                        choices=['hrhc', 'mpcc'], help='控制模式: mpcc 或 hrhc')
    return parser.parse_args()

def init_simulation(mode):
    """初始化仿真环境、模型和控制器"""
    model = DynamicBicycleModel()
    track = Track('track.csv', scale=1/43.0)

    if mode == 'mpcc':
        controller = MPCC(model, track)
    else:
        controller = HRHC(model, track)

    # 计算初始位姿：从赛道第一个中心点开始，朝向第二个点
    start_x = track.geo['pos_x'][0]
    start_y = track.geo['pos_y'][0]
    delta_x = track.geo['pos_x'][1] - start_x
    delta_y = track.geo['pos_y'][1] - start_y
    start_phi = np.arctan2(delta_y, delta_x)

    state = np.array([start_x, start_y, start_phi, 0.5, 0.0, 0.0])
    theta = track.project_to_centerline(start_x, start_y, 0.0)

    return model, track, controller, state, theta

def perform_step(controller, f_sim, state_pack):
    """执行单个仿真步"""
    state, theta = state_pack
    res = controller.solve(state, theta)

    is_fb = False
    if len(res) == 3:
        u_opt, opt_theta, is_fb = res
    else:
        u_opt, opt_theta = res

    # 状态更新 (Plant 仿真)
    next_state = f_sim(state, u_opt[:2]).full().flatten()
    # 钳制速度 >= 0，匹配控制器假设
    next_state[3] = max(0.0, next_state[3])

    return next_state, opt_theta, is_fb

def plot_results(track, history, mode):
    """绘制仿真结果并保存图像"""
    plt.figure(figsize=(10, 8))
    # 绘制赛道边界和中心线
    plt.plot(track.geo['left_x'], track.geo['left_y'], 'k-', alpha=0.3, label='Left boundary')
    plt.plot(track.geo['right_x'], track.geo['right_y'], 'k-', alpha=0.3, label='Right boundary')
    plt.plot(track.geo['pos_x'], track.geo['pos_y'], 'r--', alpha=0.4, label='Centerline')

    # 绘制执行轨迹
    hx_v, hy_v = history
    plt.plot(hx_v, hy_v, 'b-', linewidth=2, label=f'{mode.upper()} trajectory')
    plt.plot(hx_v[0], hy_v[0], 'go', label='Start')
    plt.plot(hx_v[-1], hy_v[-1], 'rx', label='End')

    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.title(f"Simulation of {mode.upper()}")
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')

    filename = f'{mode}_simulation.png'
    plt.savefig(filename, dpi=300)
    print(f"图像已保存至 {filename}")

def main():
    """主程序逻辑"""
    args = get_args()
    model, track, controller, state, theta = init_simulation(args.mode)

    f_sim = model.get_discrete_model(0.05)
    hx_v, hy_v = [state[0]], [state[1]]
    stats = SimpleNamespace(fb=0, rec=0, is_rec=False, relaxed=0)

    print(f"正在启动 {args.mode.upper()} 仿真 (N=80)...")

    for k in range(80):
        state, opt_t, is_fb = perform_step(controller, f_sim, (state, theta))

        if is_fb:
            stats.fb += 1
            stats.is_rec = True
        elif stats.is_rec:
            stats.rec += 1
            stats.is_rec = False

        if hasattr(controller, 'last_plan') and controller.last_plan.get('used_relaxed'):
            stats.relaxed += 1

        theta = track.project_to_centerline(state[0], state[1], opt_t)
        hx_v.append(state[0])
        hy_v.append(state[1])

        if k % 20 == 0:
            print(f"步数: {k}/80 | 速度: {state[3]:.2f}m/s | 进度: {theta:.2f}")

    print(f"仿真结束。降级: {stats.fb}, 恢复: {stats.rec}, relaxed_plan: {stats.relaxed}")
    plot_results(track, (hx_v, hy_v), args.mode)

if __name__ == '__main__':
    main()
