#!/usr/bin/env python3
"""
MPCC 代码生成脚本。

导出 MPCC 原型中最关键的 CasADi 内核函数，为后续切换到
结构化 QP 求解器或嵌入式代码生成做准备。
"""
import argparse
import os

import casadi as ca

from dynamic_model import DynamicBicycleModel
from track_utils import Track
from mpcc_controller import MPCC

def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="导出 MPCC CasADi 内核代码")
    parser.add_argument("--output-dir", default="generated/mpcc",
                        help="代码生成输出目录")
    parser.add_argument("--horizon", type=int, default=40,
                        help="MPCC 预测时域")
    parser.add_argument("--dt", type=float, default=0.05,
                        help="采样时间")
    return parser.parse_args()

def main():
    """脚本主入口。"""
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    model = DynamicBicycleModel()
    track = Track('track.csv', scale=1/43.0)
    controller = MPCC(model, track, horizon=args.horizon, dt_val=args.dt)

    f_discrete = model.get_discrete_model(args.dt)
    f_jac = model.get_discrete_jacobians(args.dt)

    generator = ca.CodeGenerator("mpcc_kernels.c")
    generator.add(model.f_func)
    generator.add(f_discrete)
    generator.add(f_jac)
    generator.add(track.f_theta)
    generator.add(controller.err_func)
    generated_path = generator.generate(os.path.join(args.output_dir, ""))

    metadata_path = controller.export_kernel_metadata(args.output_dir)
    print(f"代码已生成至 {generated_path}")
    print(f"元数据已生成至 {metadata_path}")

if __name__ == "__main__":
    main()
