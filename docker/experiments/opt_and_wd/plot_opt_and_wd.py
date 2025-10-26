import argparse, csv
from pathlib import Path
import matplotlib.pyplot as plt

def read_summary(p: Path):
    rows=[]
    with p.open("r", encoding="utf-8") as f:
        r=csv.DictReader(f)
        for row in r:
            row["group"]=row.get("group","").strip()
            row["optimizer"]=row.get("optimizer","").strip().lower()
            row["lr"]=float(row.get("lr",0) or 0)
            row["momentum"]=float(row.get("momentum",0) or 0)
            row["weight_decay"]=float(row.get("weight_decay",0) or 0)
            row["final_acc"]=float(row.get("final_acc",0) or 0)
            row["final_loss"]=float(row.get("final_loss",0) or 0)
            row["best_acc"]=float(row.get("best_acc",0) or 0)
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
    if name.startswith("A_run_") or name.startswith("B_run_"):
        parts=name.split("_")
        if len(parts)>=3 and parts[2].isdigit():
            return "_".join(parts[:3])  # e.g., A_run_001
    return name[:12]

def time_to_threshold(R, A, thr=0.95):
    for r,a in zip(R,A):
        if a>=thr: return r
    return float("inf")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--exp-dir", default="experiments/opt_and_wd")
    ap.add_argument("--thr", type=float, default=0.95)
    ap.add_argument("--top", type=int, default=6)
    args=ap.parse_args()

    exp_dir=Path(args.exp_dir).resolve()
    summary=read_summary(exp_dir/"summary_opt_and_wd.csv")

    runs=[]
    for row in summary:
        full=Path(row["exp_dir"]).name
        rid=run_id(full)
        rdir=exp_dir/full if (exp_dir/full).exists() else Path(row["exp_dir"]).resolve()
        R,A,L=read_metrics(rdir)
        if not R: continue
        runs.append({
            "rid": rid, "name": full, "dir": rdir, "group": row["group"],
            "optimizer": row["optimizer"], "lr": row["lr"], "momentum": row["momentum"], "wd": row["weight_decay"],
            "rounds": R, "acc": A, "loss": L,
            "final_acc": row["final_acc"], "final_loss": row["final_loss"], "best_acc": row["best_acc"],
            "tthr": time_to_threshold(R, A, args.thr)
        })

    groupA=[r for r in runs if r["group"]=="A"]
    groupB=[r for r in runs if r["group"]=="B"]

    palette = plt.rcParams['axes.prop_cycle'].by_key()['color']
    def keyA(r): return (r["optimizer"], r["lr"])
    labelsA=sorted({keyA(r) for r in groupA}, key=lambda x:(x[0],x[1]))
    colorsA={lab: palette[i%len(palette)] for i,lab in enumerate(labelsA)}
    def keyB(r): return (r["wd"], r["lr"])
    labelsB=sorted({keyB(r) for r in groupB}, key=lambda x:(x[0],x[1]))
    colorsB={lab: palette[i%len(palette)] for i,lab in enumerate(labelsB)}

    # A1: Accuracy curves (top N by time-to-threshold per optimizer)
    for opt in sorted({r["optimizer"] for r in groupA}):
        subset=[r for r in groupA if r["optimizer"]==opt]
        subset_sorted=sorted(subset, key=lambda r:(r["tthr"], -r["best_acc"]))
        sel=subset_sorted[:args.top]
        plt.figure(figsize=(10,6))
        for r in sel:
            lab=(r["optimizer"], r["lr"])
            lbl=f'{r["rid"]} {r["optimizer"]}@lr={r["lr"]} (t{int(args.thr*100)}={"∞" if r["tthr"]==float("inf") else int(r["tthr"])})'
            plt.plot(r["rounds"], r["acc"], label=lbl, linewidth=2, alpha=0.9, color=colorsA[lab])
        plt.xlabel("Round"); plt.ylabel("Test Accuracy")
        plt.ylim(0.0,1.0)
        plt.title(f'Group A: Accuracy vs Round (optimizer={opt})')
        plt.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02,1))
        plt.tight_layout()
        plt.savefig(exp_dir/f"optA_acc_curves_{opt}.png", dpi=160, bbox_inches="tight")
        plt.close()

    # A2: Final loss vs lr lines per optimizer
    by_opt={}
    for r in groupA:
        by_opt.setdefault(r["optimizer"], []).append(r)
    plt.figure(figsize=(8,6))
    for i,(opt,items) in enumerate(sorted(by_opt.items())):
        items_sorted=sorted(items, key=lambda r:r["lr"])
        xs=[r["lr"] for r in items_sorted]
        ys=[r["final_loss"] for r in items_sorted]
        c=palette[i%len(palette)]
        plt.plot(xs, ys, marker="o", linewidth=2, label=f"{opt}", color=c)
    plt.xlabel("Learning Rate"); plt.ylabel("Final Test Loss"); plt.title("Group A: Final Loss vs LR")
    plt.xscale("log")
    plt.tight_layout(); plt.legend()
    plt.savefig(exp_dir/"optA_final_loss_vs_lr.png", dpi=160, bbox_inches="tight")
    plt.close()

    # B1: Final loss vs lr, one line per weight decay (SGD)
    by_wd={}
    for r in groupB:
        by_wd.setdefault(r["wd"], []).append(r)
    plt.figure(figsize=(8,6))
    for i,(wd,items) in enumerate(sorted(by_wd.items())):
        items_sorted=sorted(items, key=lambda r:r["lr"])
        xs=[r["lr"] for r in items_sorted]
        ys=[r["final_loss"] for r in items_sorted]
        c=palette[i%len(palette)]
        plt.plot(xs, ys, marker="o", linewidth=2, label=f"wd={wd:g}", color=c)
    plt.xlabel("Learning Rate"); plt.ylabel("Final Test Loss"); plt.title("Group B (SGD): Final Loss vs LR by Weight Decay")
    plt.xscale("log")
    plt.tight_layout(); plt.legend()
    plt.savefig(exp_dir/"optB_final_loss_vs_lr_wd.png", dpi=160, bbox_inches="tight")
    plt.close()

    # B2: Heatmap of final loss (rows: wd, cols: lr)
    wds=sorted({r["wd"] for r in groupB})
    lrs=sorted({r["lr"] for r in groupB})
    grid=[[float("nan") for _ in lrs] for _ in wds]
    for r in groupB:
        i=wds.index(r["wd"]); j=lrs.index(r["lr"]); grid[i][j]=r["final_loss"]
    plt.figure(figsize=(1.6*len(lrs)+2, 0.6*len(wds)+2))
    plt.imshow(grid, aspect="auto", interpolation="nearest")
    plt.colorbar(label="Final Loss")
    plt.xticks(range(len(lrs)), [f"{v:g}" for v in lrs]); plt.yticks(range(len(wds)), [f"{v:g}" for v in wds])
    plt.xlabel("Learning Rate"); plt.ylabel("Weight Decay"); plt.title("Group B (SGD): Final Loss Heatmap")
    plt.tight_layout()
    plt.savefig(exp_dir/"optB_final_loss_heatmap.png", dpi=160, bbox_inches="tight")
    plt.close()

    # Combined small-multiples: final accuracy bars for all runs (A then B), compact labels
    all_runs_sorted=sorted(runs, key=lambda r:(r["group"], r["optimizer"], r["wd"], r["lr"]))
    labels=[f'{r["group"]}:{r["optimizer"] or "sgd"} lr={r["lr"]} wd={r["wd"]}' for r in all_runs_sorted]
    vals=[r["final_acc"] for r in all_runs_sorted]
    plt.figure(figsize=(max(12, 0.5*len(labels)), 5))
    plt.bar(range(len(labels)), vals)
    step=max(1, len(labels)//30)
    plt.xticks(range(0,len(labels),step), [labels[i] for i in range(0,len(labels),step)], rotation=45, ha="right", fontsize=8)
    plt.ylabel("Final Accuracy"); plt.title("All runs: Final Accuracy")
    plt.tight_layout()
    plt.savefig(exp_dir/"all_runs_final_acc.png", dpi=160, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    main()
