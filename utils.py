"""Small shared helpers (formatting, plot saving)."""
import os
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams.update({"figure.figsize": (11, 6), "font.size": 11,
                     "axes.titleweight": "bold"})


def banner(title: str, char: str = "=") -> None:
    line = char * 78
    print(f"\n{line}\n  {title}\n{line}\n")


def sub(title: str) -> None:
    print(f"\n--- {title} ---\n")


def save_fig(fig, path: str, dpi: int = 150) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {os.path.relpath(path)}")
