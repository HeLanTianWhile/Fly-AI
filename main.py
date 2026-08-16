"""main.py — 命令行入口：train / generate。

用法：
  python main.py train  [--epochs 30] [--per_class 300] [--batch 16]
                        [--data_dir ./data] [--out ./checkpoints]
  python main.py generate --digit 5 [--model ./checkpoints/latest.pkl]
                          [--out generated/5.png] [--n 1]
  python main.py ascii   --digit 5 [--model ./checkpoints/latest.pkl]
"""
from __future__ import annotations

import argparse
import os
import numpy as np

from cvae.model import CVAE
from cvae import train as trainer
from cvae import data as datalib
from cvae import generate as gentool


def cmd_train(args):
    rng = np.random.RandomState(args.seed)
    X, Y = datalib.make_dataset(args.data_dir, per_class=args.per_class,
                                img_size=100, download=not args.no_download,
                                seed=args.seed)
    model = CVAE(latent_dim=args.latent, cond_dim=args.cond, base=args.base)
    trainer.train(model, X, Y, epochs=args.epochs, batch_size=args.batch,
                  lr=args.lr, rng=rng, sample_every=args.sample_every,
                  out_dir=args.out, beta=args.beta, sigma=args.sigma)


def cmd_generate(args):
    model, _ = trainer.load_model(args.model)
    imgs = gentool.generate(model, args.digit, n=args.n, seed=args.seed,
                            z_scale=args.z)
    if args.ascii:
        print(gentool.to_ascii(imgs[0], width=args.width))
    else:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                    exist_ok=True)
        if args.out.endswith(".npy"):
            np.save(args.out, imgs)
            print(f"[gen] 已保存 {imgs.shape} -> {args.out}")
        else:
            gentool.write_png(args.out, imgs[0])
            print(f"[gen] 数字 {args.digit} -> 100x100 图像已保存 -> {args.out}\n")
            print(gentool.to_ascii(imgs[0], width=args.width))


def cmd_ascii(args):
    model, _ = trainer.load_model(args.model)
    imgs = gentool.generate(model, args.digit, n=1, seed=args.seed,
                            z_scale=args.z)
    print(gentool.to_ascii(imgs[0], width=args.width))


def build_parser():
    p = argparse.ArgumentParser(description="自研 CVAE 图像生成（纯 NumPy）")
    sub = p.add_subparsers(dest="cmd")

    tr = sub.add_parser("train", help="训练模型")
    tr.add_argument("--epochs", type=int, default=30)
    tr.add_argument("--per_class", type=int, default=300,
                    help="每类采样图像数（少数据）")
    tr.add_argument("--batch", type=int, default=16)
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument("--data_dir", type=str, default="./data")
    tr.add_argument("--out", type=str, default="./checkpoints")
    tr.add_argument("--latent", type=int, default=48)
    tr.add_argument("--cond", type=int, default=16)
    tr.add_argument("--base", type=int, default=16)
    tr.add_argument("--beta", type=float, default=1.0)
    tr.add_argument("--sigma", type=float, default=0.3)
    tr.add_argument("--sample_every", type=int, default=5)
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--no_download", action="store_true")

    ge = sub.add_parser("generate", help="生成图像")
    ge.add_argument("--digit", type=int, required=True)
    ge.add_argument("--model", type=str, default="./checkpoints/latest.pkl")
    ge.add_argument("--out", type=str, default="generated/digit.png")
    ge.add_argument("--n", type=int, default=1)
    ge.add_argument("--z", type=float, default=0.6)
    ge.add_argument("--seed", type=int, default=0)
    ge.add_argument("--ascii", action="store_true")
    ge.add_argument("--width", type=int, default=48)

    ac = sub.add_parser("ascii", help="终端 ASCII 预览")
    ac.add_argument("--digit", type=int, required=True)
    ac.add_argument("--model", type=str, default="./checkpoints/latest.pkl")
    ac.add_argument("--z", type=float, default=0.6)
    ac.add_argument("--seed", type=int, default=0)
    ac.add_argument("--width", type=int, default=48)

    return p


def main():
    args = build_parser().parse_args()
    if args.cmd == "train":
        cmd_train(args)
    elif args.cmd == "generate":
        cmd_generate(args)
    elif args.cmd == "ascii":
        cmd_ascii(args)
    else:
        build_parser().print_help()


if __name__ == "__main__":
    main()
