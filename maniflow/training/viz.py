import os

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")


def save_training_plots(plot_dir: str, history: dict):
    os.makedirs(plot_dir, exist_ok=True)
    epochs = history["epochs"]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, history["train_loss"], label="train", linewidth=1.5)
    ax.plot(epochs, history["val_loss"], label="val", linewidth=1.5)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Flow-matching Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "loss.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(epochs, history["lr"], color="orange", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate")
    ax.set_title("LR schedule")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "lr.png"), dpi=120)
    plt.close(fig)

    eval_epochs = history.get("eval_epochs")
    if eval_epochs:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        for ax, (title, prefix) in zip(
            axes, [("No obstacles", "obs0"), ("All obstacles", "obsall")]
        ):
            for thr, marker, label in [
                ("sr05", "s", "@0.05 rad"),
                ("sr20", "^", "@0.20 rad"),
                ("sr50", "D", "@0.50 rad"),
            ]:
                ax.plot(
                    eval_epochs,
                    history[f"{prefix}_{thr}"],
                    marker=marker,
                    markersize=4,
                    linewidth=1.5,
                    label=label,
                )
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

        axes[0].set_ylabel("Success rate")
        fig.suptitle("In-env success rate (EMA policy)")
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, "success_rate.png"), dpi=120)
        plt.close(fig)
