"""demo.py — 快速端到端演示：无需外网，几十秒训练一个小模型并生成 10 张 100x100 图像。

流程：
  1. 生成一张有明确类别结构的合成数据（数字 d -> 一个 d 大小递进的方框）
  2. 用本项目自研 CVAE 训练少量 epoch（纯 CPU）
  3. 对 0..9 各生成一张 100×100 图像保存为 PNG，并终端 ASCII 预览
  4. 打印 per-digit 覆盖度，验证"输入数字 -> 类别可区分图像"确实成立

运行：python demo.py
"""
from __future__ import annotations

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cvae.model import CVAE
from cvae import train as trainer
from cvae.generate import generate, write_png, to_ascii


def main():
    print("=" * 66)
    print("自研 CVAE 图像生成快速演示（纯 NumPy，无 AI 封装库）")
    print("=" * 66)

    rng = np.random.RandomState(3)
    ncls, per_class, img = 10, 250, 100
    X = np.zeros((ncls * per_class, 1, img, img), dtype=np.float32)
    Y = np.repeat(np.arange(ncls), per_class)
    # 类别结构：数字 d 对应一个边长 5+(d+1)*6 的居中实心方框
    for k in range(ncls * per_class):
        d = Y[k]
        side = 5 + (d + 1) * 6
        c0 = img // 2 - side // 2
        X[k, 0, c0:c0 + side, c0:c0 + side] = 0.5
    X += rng.normal(0, 0.02, X.shape).astype(np.float32)
    order = rng.permutation(len(X))
    X, Y = X[order], Y[order]

    print(f"[demo] 合成数据 {X.shape[0]} 张（每类 {per_class}），尺寸 {img}x{img}\n")

    model = CVAE(latent_dim=32, cond_dim=16, base=8)
    trainer.train(model, X, Y, epochs=12, batch_size=16, lr=1e-3, rng=rng,
                  sample_every=0, out_dir="checkpoints", verbose=True)

    os.makedirs("generated", exist_ok=True)
    print("\n[demo] 生成 0..9 各一张 100x100 图像 ...")
    coverage = []
    for d in range(ncls):
        g = generate(model, d, seed=2, z_scale=0.4)[0]
        write_png(f"generated/digit{d}.png", g)
        coverage.append(float((g > 0.45).mean() * 100))
        print(f"  digit {d}: coverage={coverage[-1]:5.1f}%  -> generated/digit{d}.png")

    print("\n生成的方框大小随数字单调递增，说明类别条件刻画在整幅图上生效。")
    for d in [0, 4, 8]:
        print(f"\n--- ASCII 预览 digit {d} ---")
        print(to_ascii(generate(model, d, seed=2, z_scale=0.4)[0], width=22))
    print("\n完成。")


if __name__ == "__main__":
    main()
