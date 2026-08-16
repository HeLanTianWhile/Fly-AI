"""train.py — 自研 CVAE 训练循环（纯 CPU + numpy 逆传播）。"""
from __future__ import annotations

import os
import time
import pickle
import numpy as np

from .tensor import Tensor
from .model import CVAE
from .losses import GaussianVAELoss
from .optimizer import Adam
from . import data as datalib


def _collect_bn_state(model):
    """收集所有 BatchNorm 层的 running_mean/var，固定顺序为模块列表序。"""
    state = []
    for m in model.modules:
        if hasattr(m, "running_mean") and hasattr(m, "running_var"):
            state.append((np.array(m.running_mean), np.array(m.running_var)))
    return state


def save_model(model, path):
    """保存自研网络的所有参数与配置（非封装库格式，纯自有格式）。
    除可学习参数外，还保存 BatchNorm 的 running stats，以保证推理（生成）正确。
    """
    state = {
        "type": "CVAE",
        "config": {
            "num_classes": model.num_classes,
            "latent_dim": model.latent_dim,
            "cond_dim": model.cond_dim,
            "base": model.base,
        },
        "params": {str(name): np.array(p.data)
                   for name, p in enumerate(model.parameters())},
        "bn_stats": _collect_bn_state(model),
    }
    with open(path, "wb") as f:
        pickle.dump(state, f)
    print(f"[train] 模型已保存 -> {path}")


def load_model(path, model=None):
    """加载模型参数。可传入一个已实例化的模型或让函数按配置重建。"""
    with open(path, "rb") as f:
        state = pickle.load(f)
    if model is None:
        cfg = state["config"]
        model = CVAE(num_classes=cfg["num_classes"],
                     latent_dim=cfg["latent_dim"],
                     cond_dim=cfg["cond_dim"], base=cfg["base"])
    params = state["params"]
    for name_str, p in enumerate_model_params(model):
        p.data[:] = params[str(name_str)]
    # 恢复 BatchNorm running stats
    bn_state = state.get("bn_stats", [])
    idx = 0
    for m in model.modules:
        if hasattr(m, "running_mean") and hasattr(m, "running_var"):
            if idx < len(bn_state):
                mean, var = bn_state[idx]
                m.running_mean[:] = mean
                m.running_var[:] = var
            idx += 1
    return model, state


def enumerate_model_params(model):
    """按固定顺序返回 (索引字符串, 参数Tensor)。索引须与保存时一致。"""
    return [(str(i), p) for i, p in enumerate(model.parameters())]


def enumerate_model_params(model):
    """按固定顺序返回 (索引字符串, 参数Tensor)。索引须与保存时一致。"""
    return [(str(i), p) for i, p in enumerate(model.parameters())]


def train(model, X, Y, epochs=30, batch_size=16, lr=1e-3, rng=None,
          sample_every=5, out_dir="checkpoints", beta=1.0, sigma=0.3,
          device="cpu", verbose=True):
    rng = rng or np.random.RandomState(0)
    loss_fn = GaussianVAELoss(sigma=sigma, beta=beta)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    os.makedirs(out_dir, exist_ok=True)
    n_batches = max(1, int(np.ceil(len(X) / batch_size)))

    model.train_mode(True)
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tot_loss = 0.0
        tot_recon = 0.0
        tot_kl = 0.0
        for xb, yb in datalib.iterate_batches(X, Y, batch_size, rng):
            x_t = Tensor(xb, requires_grad=False)
            recon, mu, logvar = model.forward(x_t, yb, rng)
            loss, recon_part, kl_part = loss_fn(recon, x_t, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tot_loss += float(loss.data)
            tot_recon += float(recon_part.data)
            tot_kl += float(kl_part.data)
        avg = tot_loss / n_batches
        ar = tot_recon / n_batches
        ak = tot_kl / n_batches
        if verbose:
            print(f"[epoch {epoch}/{epochs}] loss={avg:.4f} "
                  f"recon={ar:.4f} kl={ak:.4f} "
                  f"({time.time()-t0:.1f}s)")
        save_model(model, os.path.join(out_dir, "latest.pkl"))
        if sample_every and epoch % sample_every == 0:
            _save_samples(model, rng, os.path.join(out_dir,
                                                   f"samples_e{epoch}.npy"))
    return model


def _save_samples(model, rng, path):
    rows = []
    for d in range(10):
        g = model.generate(np.array([d]), rng)
        rows.append(g.data[0, 0])
    stack = np.stack(rows, axis=0)
    np.save(path, stack)
    print(f"[train] 生成样例已保存 -> {path} (10 x 100 x 100)")
