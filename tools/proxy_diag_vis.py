# tools/proxy_diag_vis.py
import os
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import sys

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config import cfg
from data import build_dataloader
from model import build_model

def seed_all(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def strip_module_prefix(state_dict):
    # 兼容 DDP 保存出来的 module.xxx
    if not isinstance(state_dict, dict):
        return state_dict
    if any(k.startswith("module.") for k in state_dict.keys()):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict

@torch.no_grad()
def collect_feats(model, loader, add_text: bool, max_batches: int = 200):
    feats, pids, clothes = [], [], []
    model.eval()
    for bi, data in enumerate(loader):
        if bi >= max_batches:
            break
        # ImageDataset(aux_info=False) -> (img, pid, camid, clothes_id, cloth_id_batch)
        if len(data) == 5:
            imgs, pid, camid, clothes_id, cloth_id_batch = data
        else:
            # 你的工程里也可能出现 6/7 项 batch（text/meta），这里尽量兼容
            imgs = data[0]
            pid = data[1]
            clothes_id = data[3] if len(data) > 3 else data[-2]
            cloth_id_batch = data[4] if len(data) > 4 else data[-1]

        imgs = imgs.cuda(non_blocking=True)
        pid = pid.cuda(non_blocking=True)

        if add_text:
            score, feat = model(imgs, pids=pid, text=None)
        else:
            score, feat = model(imgs)

        feat_main = feat[0] if isinstance(feat, list) else feat
        feat_main = F.normalize(feat_main, dim=1)

        feats.append(feat_main.detach().cpu())
        pids.append(pid.detach().cpu())
        # clothes_id 可能是 python int list；cloth_id_batch 是 tensor
        if isinstance(cloth_id_batch, torch.Tensor):
            clothes.append(cloth_id_batch.detach().cpu())
        else:
            clothes.append(torch.tensor(clothes_id, dtype=torch.long))

    feats = torch.cat(feats, dim=0)
    pids = torch.cat(pids, dim=0)
    clothes = torch.cat(clothes, dim=0)
    return feats, pids, clothes

def pick_subset(feats, pids, clothes, num_ids=30, per_id=25, seed=42):
    seed_all(seed)
    uniq = pids.unique().tolist()
    random.shuffle(uniq)
    pick_ids = uniq[:min(num_ids, len(uniq))]

    idx_all = []
    for pid in pick_ids:
        idx = (pids == pid).nonzero(as_tuple=False).view(-1).tolist()
        random.shuffle(idx)
        idx_all.extend(idx[:min(per_id, len(idx))])

    idx_all = torch.tensor(idx_all, dtype=torch.long)
    return feats[idx_all], pids[idx_all], clothes[idx_all], pick_ids

def sample_pair_sims(feats, pids, clothes, max_pairs=20000, seed=42):
    """
    返回四类 pairwise cosine similarity 的采样分布：
    (same pid same cloth), (same pid diff cloth), (diff pid same cloth), (diff pid diff cloth)
    """
    seed_all(seed)
    N = feats.size(0)
    if N < 2:
        return {}, {}

    # 预计算相似度（N不大时可行；我们已做了子集采样）
    sim = feats @ feats.t()  # [N,N]
    # 上三角采样避免重复
    pairs = []
    for _ in range(max_pairs):
        i = random.randrange(N)
        j = random.randrange(N)
        if i == j:
            continue
        if i > j:
            i, j = j, i
        pairs.append((i, j))
    # 去重（可选）
    pairs = list(set(pairs))

    buckets = {
        "same_pid_same_cloth": [],
        "same_pid_diff_cloth": [],
        "diff_pid_same_cloth": [],
        "diff_pid_diff_cloth": [],
    }
    for i, j in pairs:
        sp = bool(pids[i] == pids[j])
        sc = bool(clothes[i] == clothes[j])
        val = float(sim[i, j].item())
        if sp and sc:
            buckets["same_pid_same_cloth"].append(val)
        elif sp and (not sc):
            buckets["same_pid_diff_cloth"].append(val)
        elif (not sp) and sc:
            buckets["diff_pid_same_cloth"].append(val)
        else:
            buckets["diff_pid_diff_cloth"].append(val)

    stats = {k: (np.mean(v) if len(v) else float("nan"),
                 np.std(v) if len(v) else float("nan"),
                 len(v)) for k, v in buckets.items()}
    return buckets, stats

def plot_tsne(X, labels, title, outpath, proxy_mask=None):
    """
    依赖 sklearn，如果你环境没装，会提示安装。
    """
    try:
        from sklearn.manifold import TSNE
    except Exception as e:
        raise RuntimeError("需要 sklearn 才能画 t-SNE：pip install scikit-learn") from e

    X = X.numpy()
    n = X.shape[0]
    perplexity = min(30, max(5, (n - 1) // 3))
    tsne = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=perplexity, random_state=42)
    Z = tsne.fit_transform(X)

    plt.figure(figsize=(10, 8))
    labels = labels.numpy()

    # proxy 点用不同 marker
    if proxy_mask is None:
        plt.scatter(Z[:, 0], Z[:, 1], c=labels, s=10, alpha=0.85)
    else:
        proxy_mask = proxy_mask.numpy().astype(bool)
        idx_s = ~proxy_mask
        idx_p = proxy_mask
        plt.scatter(Z[idx_s, 0], Z[idx_s, 1], c=labels[idx_s], s=10, alpha=0.85)
        plt.scatter(Z[idx_p, 0], Z[idx_p, 1], c=labels[idx_p], s=90, marker="X", edgecolors="k")

    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_hist(buckets, outpath):
    plt.figure(figsize=(10, 6))
    for k, v in buckets.items():
        if len(v) == 0:
            continue
        plt.hist(v, bins=60, alpha=0.5, density=True, label=f"{k} (n={len(v)})")
    plt.legend()
    plt.xlabel("cosine similarity")
    plt.ylabel("density")
    plt.title("Pairwise cosine similarity distributions")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="output/proxy_diag")
    parser.add_argument("--num_ids", type=int, default=30)
    parser.add_argument("--per_id", type=int, default=25)
    parser.add_argument("--max_batches", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    seed_all(args.seed)

    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    # dataloader
    if cfg.DATA.DATASET == "prcc":
        trainloader, queryloader_same, queryloader_diff, galleryloader, dataset, train_sampler, val_loader, val_loader_same = build_dataloader(cfg)
        loader = val_loader  # 默认用 clothes-changing setting（query_diff + gallery）
    else:
        trainloader, queryloader, galleryloader, dataset, train_sampler, val_loader = build_dataloader(cfg)
        loader = val_loader

    # model
    model = build_model(cfg, dataset.num_train_pids)
    model = model.cuda()

    ckpt = torch.load(args.model_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    ckpt = strip_module_prefix(ckpt)
    msg = model.load_state_dict(ckpt, strict=False)
    print("[load] missing:", len(msg.missing_keys), "unexpected:", len(msg.unexpected_keys))

    add_text = bool(getattr(cfg.MODEL, "ADD_TEXT", False))

    # collect
    feats, pids, clothes = collect_feats(model, loader, add_text=add_text, max_batches=args.max_batches)
    feats_s, pids_s, clothes_s, pick_ids = pick_subset(feats, pids, clothes, num_ids=args.num_ids, per_id=args.per_id, seed=args.seed)

    # proxies for picked ids
    if not hasattr(model, "head") or not hasattr(model.head, "weight"):
        raise AttributeError("需要 model.head.weight 作为 proxy（你的项目就是这样用的）")
    head_w = F.normalize(model.head.weight.detach().cpu(), dim=1)  # [C,D]
    pick_ids_t = torch.tensor(pick_ids, dtype=torch.long)
    proxies = head_w.index_select(0, pick_ids_t)  # [K,D]

    # concat samples + proxies
    X_all = torch.cat([feats_s, proxies], dim=0)
    # labels for t-SNE coloring
    # 1) PID 上色：样本用 pid，proxy 用对应 pid
    pid_labels = torch.cat([pids_s, pick_ids_t], dim=0)
    proxy_mask = torch.cat([torch.zeros(feats_s.size(0), dtype=torch.bool),
                            torch.ones(proxies.size(0), dtype=torch.bool)], dim=0)

    # 2) cloth 上色：样本用 clothes，proxy 用 -1（单独一类）
    cloth_labels = torch.cat([clothes_s, torch.full((proxies.size(0),), -1, dtype=torch.long)], dim=0)

    # plots
    plot_tsne(X_all, pid_labels, "t-SNE colored by PID (proxy points are X)", os.path.join(args.outdir, "tsne_pid.png"), proxy_mask=proxy_mask)
    plot_tsne(X_all, cloth_labels, "t-SNE colored by clothes_id (proxy points are X, cloth=-1)", os.path.join(args.outdir, "tsne_cloth.png"), proxy_mask=proxy_mask)

    # similarity hist (pairwise on samples only)
    buckets, stats = sample_pair_sims(feats_s, pids_s, clothes_s, max_pairs=40000, seed=args.seed)
    plot_hist(buckets, os.path.join(args.outdir, "pairwise_sim_hist.png"))

    # proxy alignment distribution
    # 每个样本与其 pid 对应 proxy 的相似度
    # 注意：pids_s 是真实 pid，head_w 是所有类的 proxy
    px = head_w.index_select(0, pids_s.long())
    sim_fp = (feats_s * px).sum(dim=1).numpy()
    plt.figure(figsize=(10, 4))
    plt.hist(sim_fp, bins=60, alpha=0.85, density=True)
    plt.xlabel("cos(feat, proxy_pid)")
    plt.ylabel("density")
    plt.title("Feature-Proxy cosine similarity distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "feat_proxy_hist.png"), dpi=200)
    plt.close()

    print("\n=== Pairwise similarity stats (mean/std/n) ===")
    for k, (m, s, n) in stats.items():
        print(f"{k:22s}: mean={m:.4f} std={s:.4f} n={n}")
    print(f"\nfeat-proxy: mean={float(np.mean(sim_fp)):.4f}, std={float(np.std(sim_fp)):.4f}, n={sim_fp.shape[0]}")
    print(f"\nSaved figures to: {args.outdir}")

if __name__ == "__main__":
    main()
