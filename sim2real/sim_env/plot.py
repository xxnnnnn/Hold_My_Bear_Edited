import argparse
from pathlib import Path
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np


def load_npz(npz_path: Path) -> dict:
    """Load a single npz file and return data dict."""
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")
    data = np.load(npz_path)
    required_keys = [
        "wrist_roll_acc",
        "wrist_pitch_acc",
        "wrist_yaw_acc",
        "wrist_total_acc_magnitude",
    ]
    for k in required_keys:
        if k not in data:
            raise KeyError(f"NPZ missing key: {k}")
    return {k: data[k] for k in required_keys}


def find_npz_files(folder_path: Path) -> List[Path]:
    """Find all npz files in the given folder."""
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    if not folder_path.is_dir():
        raise ValueError(f"Path is not a directory: {folder_path}")
    
    npz_files = sorted(folder_path.glob("*.npz"))
    if len(npz_files) == 0:
        raise FileNotFoundError(f"No npz files found in: {folder_path}")
    return npz_files


def plot_wrist_acc_single(data: dict, title: str = ""):
    """Plot wrist acceleration data from a single npz file."""
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    keys = [
        ("wrist_roll_acc", "Right Wrist Roll Acc (rad/s²)"),
        ("wrist_pitch_acc", "Right Wrist Pitch Acc (rad/s²)"),
        ("wrist_yaw_acc", "Right Wrist Yaw Acc (rad/s²)"),
        ("wrist_total_acc_magnitude", "Total Magnitude (rad/s²)"),
    ]
    for ax, (k, label) in zip(axes, keys):
        y = data[k]
        ax.plot(y, lw=1.0)
        ax.set_ylabel(label)
        ax.grid(True, linestyle="--", alpha=0.5)
    axes[-1].set_xlabel("Sample Index")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_wrist_acc_multiple(data_list: List[Dict], labels: List[str], title: str = ""):
    """Plot wrist acceleration data from multiple npz files on the same subplots."""
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    keys = [
        ("wrist_roll_acc", "Right Wrist Roll Acc (rad/s²)"),
        ("wrist_pitch_acc", "Right Wrist Pitch Acc (rad/s²)"),
        ("wrist_yaw_acc", "Right Wrist Yaw Acc (rad/s²)"),
        ("wrist_total_acc_magnitude", "Total Magnitude (rad/s²)"),
    ]
    
    # Use a high-contrast colormap with better color separation
    # Try different colormaps based on number of files
    n_files = len(data_list)
    if n_files <= 10:
        # Use Set3 for better color distinction (up to 12 colors)
        colors = plt.cm.Set3(np.linspace(0, 1, n_files))
    elif n_files <= 20:
        # Use tab20 for more colors
        colors = plt.cm.tab20(np.linspace(0, 1, n_files))
    else:
        # Use hsv for many files (cyclic, but distinct)
        colors = plt.cm.hsv(np.linspace(0, 1, n_files))
    
    # Alternative: manually define high-contrast colors
    manual_colors = [
        '#1f77b4',  # blue
        '#ff7f0e',  # orange
        '#2ca02c',  # green
        '#d62728',  # red
        '#9467bd',  # purple
        '#8c564b',  # brown
        '#e377c2',  # pink
        '#7f7f7f',  # gray
        '#bcbd22',  # olive
        '#17becf',  # cyan
        '#aec7e8',  # light blue
        '#ffbb78',  # light orange
        '#98df8a',  # light green
        '#ff9896',  # light red
        '#c5b0d5',  # light purple
    ]
    
    # Use manual colors if available, otherwise use colormap
    if n_files <= len(manual_colors):
        colors = [manual_colors[i] for i in range(n_files)]
    
    for ax, (k, ylabel) in zip(axes, keys):
        for i, (data, label, color) in enumerate(zip(data_list, labels, colors)):
            y = data[k]
            ax.plot(y, lw=1.5, label=label, color=color, alpha=0.9)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.5)
        if len(data_list) > 1:
            ax.legend(loc='upper right', fontsize=8)
    
    axes[-1].set_xlabel("Sample Index")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description="Plot wrist acceleration recording")
    parser.add_argument("--folder_path", type=str, help="Path to folder containing npz files")
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Save image path (default: folder_path/plot.png)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save only, don't show window",
    )
    args = parser.parse_args()

    folder_path = Path(args.folder_path)
    npz_files = find_npz_files(folder_path)
    
    print(f"Found {len(npz_files)} npz file(s) in {folder_path}")
    
    # Load all npz files
    data_list = []
    labels = []
    for npz_file in npz_files:
        data = load_npz(npz_file)
        data_list.append(data)
        labels.append(npz_file.stem)  # filename without extension
    
    # Plot based on number of files
    if len(data_list) == 1:
        title = f"Wrist Acceleration: {labels[0]}"
        fig = plot_wrist_acc_single(data_list[0], title=title)
        default_save = folder_path / f"{labels[0]}.png"
    else:
        title = f"Wrist Acceleration Comparison ({len(data_list)} files)"
        fig = plot_wrist_acc_multiple(data_list, labels, title=title)
        default_save = folder_path / "plot.png"
    
    save_path = Path(args.save) if args.save else default_save
    fig.savefig(save_path, dpi=200)
    print(f"Image saved to: {save_path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()

