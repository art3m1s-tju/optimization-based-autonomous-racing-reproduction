#!/usr/bin/env python3
"""
回归测试模块。
用于对 MPCC/HRHC 控制器进行批量回归测试，评估不同初始状态下的求解稳定性和耗时。
"""
import argparse
import time
import csv
from types import SimpleNamespace
import numpy as np

from dynamic_model import DynamicBicycleModel
from hrhc_controller import HRHC
from track_utils import Track
from mpcc_controller import MPCC

def get_start_state(track, theta_start):
    """根据给定的 theta 进度计算初始车辆状态"""
    th_norm = float(theta_start) % track.max_lapdist
    # 在航位推演数据中寻找最近点
    idx = int(np.argmin(np.abs(track.geo['lapdist'] - th_norm)))

    px_val = track.geo['pos_x'][idx]
    py_val = track.geo['pos_y'][idx]

    # 计算朝向角：朝向下一个航点
    idx_next = min(idx + 1, len(track.geo['pos_x']) - 1)
    delta_x = track.geo['pos_x'][idx_next] - px_val
    delta_y = track.geo['pos_y'][idx_next] - py_val
    phi = np.arctan2(delta_y, delta_x)

    state = np.array([px_val, py_val, phi, 0.5, 0.0, 0.0])
    return state, th_norm

def build_controller(track, cfg):
    """按模式构造控制器。"""
    model = DynamicBicycleModel()
    if cfg.mode == "mpcc":
        return MPCC(model, track, horizon=cfg.horizon, dt_val=cfg.dt_val)
    return HRHC(model, track)

def run_one_test(track, theta_start, cfg):
    """
    执行单个回归测试用例。

    参数:
        track: 赛道对象
        theta_start: 初始进度
        cfg: 配置对象
    """
    controller = build_controller(track, cfg)
    ctrl_dt = cfg.dt_val if cfg.mode == "mpcc" else controller.params['dt']
    f_sim = controller.model.get_discrete_model(ctrl_dt)

    state, theta = get_start_state(track, theta_start)
    theta = track.project_to_centerline(state[0], state[1], theta)

    stats = SimpleNamespace(fb=0, rec=0, first_inf=None, is_rec=False, times=[],
                            relaxed=0)

    # 热启动预热 (静默异常处理)
    try:
        controller.solve(state, theta)
    except Exception: # pylint: disable=broad-except
        pass

    for k in range(cfg.n_sim):
        stats.times.append(time.perf_counter())
        res = controller.solve(state, theta)
        stats.times[-1] = time.perf_counter() - stats.times[-1]

        u_opt, opt_t, is_fb = (res if len(res) == 3 else (res[0], res[1], False))

        if is_fb:
            stats.fb += 1
            stats.is_rec = True
            if stats.first_inf is None:
                stats.first_inf = k
        elif stats.is_rec:
            stats.rec += 1
            stats.is_rec = False

        if hasattr(controller, 'last_plan') and controller.last_plan.get('used_relaxed'):
            stats.relaxed += 1

        state = f_sim(state, u_opt[:2]).full().flatten()
        if not cfg.no_clamp:
            state[3] = max(0.0, state[3])
        theta = track.project_to_centerline(state[0], state[1], opt_t)

    return {
        "mode": cfg.mode,
        "theta_start": theta_start, "fallbacks": stats.fb, "recoveries": stats.rec,
        "relaxed_plan_steps": stats.relaxed,
        "first_inf": stats.first_inf if stats.first_inf is not None else "none",
        "seamless": stats.fb == 0, "final_theta": theta,
        "mean_ms": np.mean(stats.times) * 1000, "max_ms": np.max(stats.times) * 1000
    }

def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="控制器回归测试")
    parser.add_argument("--mode", choices=["mpcc", "hrhc", "both"], default="mpcc",
                        help="测试 MPCC、HRHC 或两者都测")
    parser.add_argument("--cases", type=int, default=10, help="测试起点数量")
    parser.add_argument("--steps", type=int, default=80, help="每个用例的闭环步数")
    parser.add_argument("--theta-span", type=float, default=0.3,
                        help="测试起点覆盖赛道长度的比例")
    parser.add_argument("--horizon", type=int, default=40,
                        help="MPCC 预测时域长度")
    parser.add_argument("--dt", type=float, default=0.05, help="仿真采样时间")
    parser.add_argument("--csv", type=str, default="regression_results.csv",
                        help="结果输出 CSV 路径")
    parser.add_argument("--no-clamp", action="store_true",
                        help="关闭速度非负钳制")
    return parser.parse_args()

def print_summary(results):
    """打印汇总信息。"""
    if not results:
        return

    by_mode = {}
    for item in results:
        by_mode.setdefault(item["mode"], []).append(item)

    print("\n汇总:")
    for mode, mode_results in by_mode.items():
        seamless_count = sum(1 for item in mode_results if item["seamless"])
        mean_ms = np.mean([item["mean_ms"] for item in mode_results])
        max_ms = np.max([item["max_ms"] for item in mode_results])
        relaxed_steps = sum(item["relaxed_plan_steps"] for item in mode_results)
        print(f"  {mode.upper()}: {seamless_count}/{len(mode_results)} 无缝求解 | "
              f"case_mean={mean_ms:.1f}ms | case_max={max_ms:.1f}ms | "
              f"relaxed_steps={relaxed_steps}")

def main():
    """回归测试主程序"""
    args = parse_args()
    track = Track('track.csv', scale=1/43.0)
    theta_starts = np.linspace(0.0, track.max_lapdist * args.theta_span, args.cases)
    modes = ["mpcc", "hrhc"] if args.mode == "both" else [args.mode]

    results = []
    print(f"正在启动 {len(modes)} 个模式、每个 {len(theta_starts)} 个回归测试用例...")

    for mode in modes:
        cfg = SimpleNamespace(
            mode=mode, horizon=args.horizon, dt_val=args.dt,
            n_sim=args.steps, no_clamp=args.no_clamp
        )
        print(f"\n[{mode.upper()}]")
        for i, th_val in enumerate(theta_starts):
            print(f"  [{i+1}/{len(theta_starts)}] theta_start={th_val:.2f}",
                  end="", flush=True)
            try:
                res_dict = run_one_test(track, th_val, cfg)
                status = "OK" if res_dict["seamless"] else f"DEGRADED({res_dict['fallbacks']})"
                print(f" -> {status} mean={res_dict['mean_ms']:.1f}ms")
                results.append(res_dict)
            except Exception as err:  # pylint: disable=broad-except
                print(f" -> CRASH: {err}")

    # 保存结果到 CSV
    if results:
        with open(args.csv, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print_summary(results)
        print(f"\n结果已保存至 {args.csv}")

if __name__ == "__main__":
    main()
