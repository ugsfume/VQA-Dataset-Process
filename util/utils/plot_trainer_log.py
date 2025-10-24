
"""
Plot training curves from a LLaMA-Factory / HF-style JSONL trainer log.

Usage:
    python plot_trainer_log.py /path/to/trainer_log.jsonl
If no path is provided, defaults to ./trainer_log.jsonl relative to the script.
Outputs:
    - loss_vs_steps.png
    - eval_loss_vs_steps.png
    - lr_vs_steps.png
    - loss_vs_epoch.png
    - elapsed_time_vs_steps.png
    - A combined PDF "trainer_log_plots.pdf" (optional if matplotlib backend supports it)
"""

import sys
import json
from pathlib import Path
from datetime import timedelta

import matplotlib.pyplot as plt


def parse_duration_to_seconds(s: str) -> float:
    """
    Parse durations like:
        "0:10:04"
        "1 day, 4:06:56"
        "2 days, 1:44:27"
    Return seconds as float.
    """
    if s is None:
        return None
    s = s.strip()
    days = 0
    if "day" in s:
        # split "X day(s), HH:MM:SS"
        parts = s.split(",")
        day_part = parts[0].strip()
        days = int(day_part.split()[0])
        time_part = parts[1].strip() if len(parts) > 1 else "0:00:00"
    else:
        time_part = s

    hh, mm, ss = time_part.split(":")
    total = days * 24 * 3600 + int(hh) * 3600 + int(mm) * 60 + int(ss)
    return float(total)


def load_log(jsonl_path: Path):
    steps, epochs, losses, lrs, eval_steps, eval_losses = [], [], [], [], [], []
    elapsed_steps, elapsed_seconds = [], []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # Skip malformed lines
                continue

            step = obj.get("current_steps")
            epoch = obj.get("epoch")
            if "loss" in obj:
                loss = obj.get("loss")
                lr = obj.get("lr")
                steps.append(step)
                epochs.append(epoch)
                losses.append(loss)
                lrs.append(lr)
                # elapsed time parsing (optional)
                et = obj.get("elapsed_time")
                if et is not None:
                    elapsed_steps.append(step)
                    elapsed_seconds.append(parse_duration_to_seconds(et))
            elif "eval_loss" in obj:
                eval_steps.append(step)
                eval_losses.append(obj.get("eval_loss"))
            else:
                # ignore other types
                pass
    return {
        "steps": steps,
        "epochs": epochs,
        "losses": losses,
        "lrs": lrs,
        "eval_steps": eval_steps,
        "eval_losses": eval_losses,
        "elapsed_steps": elapsed_steps,
        "elapsed_seconds": elapsed_seconds,
    }


def plot_series(x, y, xlabel, ylabel, title, outfile):
    plt.figure()
    plt.plot(x, y)  # do not set any colors/styles explicitly
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.show()



def plot_overlap_loss(steps, losses, eval_steps, eval_losses, outfile):
    # Plot both training loss and eval loss on the same axes
    import matplotlib.pyplot as plt
    plt.figure()
    if steps and losses:
        plt.plot(steps, losses, label="train loss")
    if eval_steps and eval_losses:
        plt.plot(eval_steps, eval_losses, label="eval loss")
    plt.xlabel("steps")
    plt.ylabel("loss")
    plt.title("Training vs Eval Loss (overlapping)")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.show()


def main():

    if len(sys.argv) > 1:
        jsonl_path = Path(sys.argv[1])
    else:
        jsonl_path = Path(__file__).parent / "trainer_log.jsonl"

    if not jsonl_path.exists():
        print(f"[!] Log file not found: {jsonl_path}")
        print("    Provide a path like: python plot_trainer_log.py /path/to/trainer_log.jsonl")
        sys.exit(1)

    data = load_log(jsonl_path)

    if data["steps"]:
        plot_series(
            data["steps"],
            data["losses"],
            "steps",
            "training loss",
            "Training Loss vs Steps",
            "loss_vs_steps.png",
        )

        # Loss vs epoch (can be more directly interpretable across runs)
        plot_series(
            data["epochs"],
            data["losses"],
            "epoch",
            "training loss",
            "Training Loss vs Epoch",
            "loss_vs_epoch.png",
        )

        # Learning rate
        plot_series(
            data["steps"],
            data["lrs"],
            "steps",
            "learning rate",
            "Learning Rate vs Steps",
            "lr_vs_steps.png",
        )

    if data["eval_steps"]:
        plot_series(
            data["eval_steps"],
            data["eval_losses"],
            "steps",
            "eval loss",
            "Eval Loss vs Steps",
            "eval_loss_vs_steps.png",
        )

    if data["elapsed_steps"]:
        plot_series(
            data["elapsed_steps"],
            data["elapsed_seconds"],
            "steps",
            "elapsed seconds",
            "Elapsed Time vs Steps",
            "elapsed_time_vs_steps.png",
        )


if __name__ == "__main__":
    main()
