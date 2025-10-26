import argparse, csv
from pathlib import Path
import matplotlib.pyplot as plt
import torch.nn as nn

def read_summary(p: Path):
    rows=[]
    with p.open("r", encoding="utf-8") as f:
        r=csv.DictReader(f)
        for row in r:
            row["final_acc"]=float(row.get("final_acc",0) or 0)
            row["final_loss"]=float(row.get("final_loss",0) or 0)
            row["best_acc"]=float(row.get("best_acc",0) or 0)
            row["t95"]=float(row.get("t95", float("inf")) or float("inf"))
            row["stage1_convs"]=int(float(row.get("stage1_convs",1) or 1))
            row["stage2_convs"]=int(float(row.get("stage2_convs",1) or 1))
            rows.append(row)
    return rows

def read_metrics(run_dir: Path):
    rounds, acc, loss=[], [], []
    p=run_dir/"metrics.csv"
    if not p.exists(): return rounds, acc, loss
    with p.open("r", encoding="utf-8") as f:
        r=csv.DictReader(f)
        for row in r:
            rounds.append(int(row["round"]))
            acc.append(float(row["test_accuracy"]))
            loss.append(float(row["test_loss"]))
    return rounds, acc, loss

def run_id(name: str) -> str:
    if name.startswith("run_"):
        parts=name.split("_")
        if len(parts)>=2 and parts[1].isdigit():
            return "_".join(parts[:2])
    return name[:10]

def build_variant(s1:int, s2:int) -> nn.Module:
    c1, c2 = 32, 64
    layers = []
    layers += [nn.Conv2d(1, c1, 3, padding=1), nn.ReLU(inplace=True)]
    for _ in range(s1-1):
        layers += [nn.Conv2d(c1, c1, 3, padding=1), nn.ReLU(inplace=True)]
    layers += [nn.MaxPool2d(2)]
    layers += [nn.Conv2d(c1, c2, 3, padding=1), nn.ReLU(inplace=True)]
    for _ in range(s2-1):
        layers += [nn.Conv2d(c2, c2, 3, padding=1), nn.ReLU(inplace=True)]
    layers += [nn.MaxPool2d(2)]
    layers += [nn.Flatten(), nn.Linear(c2*7*7,128), nn.ReLU(inplace=True), nn.Linear(128,10)]
    return nn.Sequential(*layers)

def count_params(s1:int, s2:int) -> int:
    return sum(p.numel() for p in build_variant(s1, s2).parameters())

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--exp-dir", default="experiments/depth_sweep")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--truncate-at", type=float, default=None)
    ap.add_argument("--out-prefix", default=None)
    args=ap.parse_args()

    exp_dir=Path(args.exp_dir).resolve()
    out_dir=exp_dir if not args.out_prefix else Path(args.out_prefix); out_dir.mkdir(parents=True, exist_ok=True)

    rows=read_summary(exp_dir/"summary_depth_clean.csv")

    runs=[]
    for row in rows:
        full=Path(row["exp_dir"]).name
        rid=run_id(full)
        rdir=exp_dir/full if (exp_dir/full).exists() else Path(row["exp_dir"]).resolve()
        R,A,L=read_metrics(rdir)
        if not R: continue
        s1, s2 = row["stage1_convs"], row["stage2_convs"]
        runs.append({
            "rid": rid, "name": full, "dir": rdir,
            "rounds": R, "acc": A, "loss": L,
            "final_acc": row["final_acc"], "final_loss": row["final_loss"],
            "best_acc": row["best_acc"], "t95": row["t95"],
            "s1": s1, "s2": s2, "params": count_params(s1, s2),
        })

    runs_sorted = sorted(runs, key=lambda x: (x["t95"], -x["best_acc"]))
    sel = runs_sorted[:args.top]

    palette = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_map = {r["rid"]: palette[i % len(palette)] for i,r in enumerate(runs_sorted)}

    plt.figure(figsize=(10,6))
    for r in sel:
        R, A = r["rounds"], r["acc"]
        if args.truncate_at is not None:
            idx = None
            for i,(rr,aa) in enumerate(zip(R,A)):
                if aa >= args.truncate_at:
                    idx = i; break
            if idx is not None:
                R, A = R[:idx+1], A[:idx+1]
        lbl = f'{r["rid"]} (s1={r["s1"]}, s2={r["s2"]}, t95={ "∞" if r["t95"]==float("inf") else int(r["t95"])})'
        plt.plot(R, A, label=lbl, linewidth=2, alpha=0.9, color=color_map[r["rid"]])
    plt.xlabel("Round"); plt.ylabel("Test Accuracy")
    title = "Depth clean: Accuracy vs Round"
    if args.truncate_at is not None: title += f" (truncated at {int(args.truncate_at*100)}%)"
    plt.title(title)
    plt.ylim(0.0,1.0)
    plt.legend(fontsize=8, ncol=1, loc="upper left", bbox_to_anchor=(1.02,1))
    plt.tight_layout()
    plt.savefig(out_dir/"depth_clean_acc_curves.png", dpi=160, bbox_inches="tight")
    plt.close()

    labels=[r["rid"] for r in runs_sorted]
    losses=[r["final_loss"] for r in runs_sorted]
    plt.figure(figsize=(max(10, 0.8*len(labels)), 5))
    bars=plt.bar(range(len(labels)), losses)
    for i,b in enumerate(bars):
        b.set_color(color_map[labels[i]])
    step=max(1, len(labels)//25)
    plt.xticks(range(0,len(labels),step), [labels[i] for i in range(0,len(labels),step)],
               rotation=45, ha="right", fontsize=8)
    plt.ylabel("Final Test Loss"); plt.title("Depth clean: Final Loss (all runs)")
    plt.tight_layout()
    plt.savefig(out_dir/"depth_clean_final_loss_all.png", dpi=160, bbox_inches="tight")
    plt.close()

    xs=[r["params"]/1e6 for r in runs_sorted]
    ys=[r["best_acc"] for r in runs_sorted]
    plt.figure(figsize=(8,6))
    for r in runs_sorted:
        plt.scatter(r["params"]/1e6, r["best_acc"], color=color_map[r["rid"]])
    for r in runs_sorted:
        plt.text(r["params"]/1e6, r["best_acc"], r["rid"], fontsize=7, ha="left", va="bottom")
    plt.xlabel("Parameters (Millions)")
    plt.ylabel("Best Accuracy")
    plt.title("Depth clean: Model Size vs Best Accuracy")
    plt.tight_layout()
    plt.savefig(out_dir/"depth_clean_params_vs_bestacc.png", dpi=160, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    main()