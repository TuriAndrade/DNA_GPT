import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from cycler import cycler


def plot_losses(train_losses, val_losses, save_path):
    plt.rcParams["axes.prop_cycle"] = cycler(
        color=["#377eb8", "#ff7f00"]
    )  # Enhanced color palette

    epochs = range(1, len(train_losses) + 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        epochs,
        train_losses,
        label="Train Loss",
        linewidth=2,
        linestyle="--",
        marker="o",
        markersize=5,
    )
    ax.plot(
        epochs,
        val_losses,
        label="Validation Loss",
        linewidth=2,
        linestyle="-",
        marker="s",
        markersize=5,
    )

    ax.set_xlabel("Epochs", fontsize=14)
    ax.set_ylabel("Loss", fontsize=14)
    ax.set_title(
        "Training and Validation Losses",
        fontsize=16,
        fontweight="bold",
    )

    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend(fontsize=12, frameon=True, framealpha=0.7, edgecolor="none")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.spines["left"].set_linewidth(0.5)

    ax.tick_params(width=0.5)
    ax.set_facecolor("whitesmoke")
    fig.tight_layout()

    plt.savefig(save_path, dpi=300, format="png")
    plt.close()
