
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

RESULTS = Path("results")
PLOTS = RESULTS / "plots"
PLOTS.mkdir(exist_ok=True)

COVER = RESULTS.parent / "imgs" / "cover" / "gradient.png"

STEGOS = {
    "Plain LSB 5%": RESULTS / "gradient_5pct_plain.png",
    "Encrypted LSB 5%": RESULTS / "gradient_5pct_encrypted.png",
    "Encrypted Random LSB 5%": RESULTS / "gradient_5pct_random.png",
}


def load_image(path):
    return np.array(Image.open(path).convert("RGB"))


def print_table():
    path = RESULTS / "volume_metrics.csv"
    df = pd.read_csv(path)

    columns = [
        "fill_percent",
        "method",
        "payload_bytes",
        "mse",
        "psnr",
        "ssim",
        "changed_ratio",
        "chi2",
        "log10_p_value",
    ]

    print("\nEXPERIMENT RESULTS\n")
    print(df[columns].to_string(index=False))


def check_images():
    print("\nIMAGE DIMENSIONS\n")

    paths = [COVER] + list(STEGOS.values())

    for path in paths:
        if not path.exists():
            print(f"MISSING: {path}")
            continue

        image = Image.open(path)
        print(f"{path}: {image.size}, {image.mode}")


def create_rgb_histograms():
    cover = load_image(COVER)

    channel_names = ["Red", "Green", "Blue"]

    fig, axes = plt.subplots(3, 1, figsize=(12, 12))

    for channel in range(3):
        axes[channel].hist(
            cover[:, :, channel].flatten(),
            bins=256,
            range=(0, 256),
            density=True,
            histtype="step",
            label="Cover",
        )

        for label, path in STEGOS.items():
            stego = load_image(path)

            axes[channel].hist(
                stego[:, :, channel].flatten(),
                bins=256,
                range=(0, 256),
                density=True,
                histtype="step",
                label=label,
            )

        axes[channel].set_title(
            f"{channel_names[channel]} channel"
        )
        axes[channel].set_xlabel("Intensity")
        axes[channel].set_ylabel("Normalized frequency")
        axes[channel].legend()
        axes[channel].grid(alpha=0.3)

    fig.suptitle("RGB intensity distributions: Cover vs Stego")
    fig.tight_layout()

    fig.savefig(
        PLOTS / "rgb_histograms.png",
        dpi=200,
    )
    plt.close(fig)


def create_histogram_differences():
    cover = load_image(COVER)

    channel_names = ["Red", "Green", "Blue"]
    bins = np.arange(257)

    fig, axes = plt.subplots(3, 1, figsize=(12, 12))

    for channel in range(3):
        cover_hist, _ = np.histogram(
            cover[:, :, channel],
            bins=bins,
        )

        for label, path in STEGOS.items():
            stego = load_image(path)

            stego_hist, _ = np.histogram(
                stego[:, :, channel],
                bins=bins,
            )

            difference = stego_hist - cover_hist

            axes[channel].plot(
                np.arange(256),
                difference,
                label=label,
            )

        axes[channel].axhline(0, linewidth=1)
        axes[channel].set_title(
            f"{channel_names[channel]}: change in histogram frequencies"
        )
        axes[channel].set_xlabel("Intensity")
        axes[channel].set_ylabel("Frequency difference")
        axes[channel].legend()
        axes[channel].grid(alpha=0.3)

    fig.suptitle("Histogram differences: Stego minus Cover")
    fig.tight_layout()

    fig.savefig(
        PLOTS / "histogram_differences.png",
        dpi=200,
    )
    plt.close(fig)


if __name__ == "__main__":
    print_table()
    check_images()
    create_rgb_histograms()
    create_histogram_differences()

    print("\nInspection completed.")
    print(f"Plots saved in: {PLOTS}")
