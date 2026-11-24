import numpy as np
import pandas as pd
import mpmath as mp

from PIL import Image
from scipy.stats import chi2 as chi2_distribution
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity


def load_image(path):
    return np.array(
        Image.open(path).convert("RGB"),
        dtype=np.uint8
    )


def calculate_mse(original, stego):
    original = original.astype(np.float64)
    stego = stego.astype(np.float64)

    return float(np.mean((original - stego) ** 2))


def calculate_psnr(original, stego):
    return float(
        peak_signal_noise_ratio(
            original,
            stego,
            data_range=255
        )
    )


def calculate_ssim(original, stego):
    return float(
        structural_similarity(
            original,
            stego,
            channel_axis=2,
            data_range=255
        )
    )


def calculate_changed_components(original, stego):
    difference = original != stego
    changed = int(np.count_nonzero(difference))
    total = int(original.size)

    return {
        "changed_components": changed,
        "changed_ratio": changed / total
    }


def calculate_lsb_distribution(image):
    bits = image & 1

    zeros = int(np.count_nonzero(bits == 0))
    ones = int(np.count_nonzero(bits == 1))
    total = zeros + ones

    return {
        "lsb_0": zeros / total,
        "lsb_1": ones / total
    }


def calculate_chi_square(image):
    """
    Pair-of-values chi-square test.

    Compares frequencies of 2k and 2k+1
    for all 128 pairs of 8-bit values.
    """

    values = image.flatten()
    histogram = np.bincount(values, minlength=256)

    even_counts = histogram[0::2].astype(np.float64)
    odd_counts = histogram[1::2].astype(np.float64)

    observed = np.column_stack(
        (even_counts, odd_counts)
    ).flatten()

    expected_per_pair = (
        even_counts + odd_counts
    ) / 2.0

    expected = np.column_stack(
        (expected_per_pair, expected_per_pair)
    ).flatten()

    mask = expected > 0

    # One degree of freedom for each non-empty pair.
    non_empty_pairs = (
        (even_counts + odd_counts) > 0
    )

    degrees_of_freedom = int(
        np.count_nonzero(non_empty_pairs)
    )

    if degrees_of_freedom == 0:
        return {
            "chi2": float("nan"),
            "p_value": float("nan"),
            "log10_p_value": float("nan")
        }

    statistic = float(
        np.sum(
            (observed[mask] - expected[mask]) ** 2
            / expected[mask]
        )
    )

    # High-precision survival function calculation.
    # p-value = Q(df / 2, chi2 / 2),
    # where Q is the regularized upper incomplete gamma function.
    with mp.workdps(80):
        mp_statistic = mp.mpf(str(statistic))
        mp_df = mp.mpf(degrees_of_freedom)

        p_mp = mp.gammainc(
            mp_df / 2,
            mp_statistic / 2,
            mp.inf,
            regularized=True
        )

        log10_p_mp = mp.log10(p_mp)

        log10_p_value = float(log10_p_mp)

        # Conversion to float can underflow for extremely small
        # probabilities. In that case p_value is 0.0, but
        # log10_p_value remains finite and informative.
        p_value = float(p_mp)

    return {
        "chi2": statistic,
        "p_value": p_value,
        "log10_p_value": log10_p_value
    }


def analyze_images(original_path, stego_path, method):
    original = load_image(original_path)
    stego = load_image(stego_path)

    if original.shape != stego.shape:
        raise ValueError("Размеры изображений не совпадают")

    result = {
        "method": method,
        "mse": calculate_mse(original, stego),
        "psnr": calculate_psnr(original, stego),
        "ssim": calculate_ssim(original, stego)
    }

    result.update(
        calculate_changed_components(original, stego)
    )

    result.update(
        calculate_lsb_distribution(stego)
    )

    result.update(
        calculate_chi_square(stego)
    )

    return result


def analyze_cover(image_path):
    image = load_image(image_path)

    result = {
        "method": "Cover",
        "mse": 0.0,
        "psnr": float("inf"),
        "ssim": 1.0,
        "changed_components": 0,
        "changed_ratio": 0.0
    }

    result.update(
        calculate_lsb_distribution(image)
    )

    result.update(
        calculate_chi_square(image)
    )

    return result


def save_results(results, output_path="results/metrics.csv"):
    df = pd.DataFrame(results)

    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nStatistical analysis:")
    print(df.to_string(index=False))

    print(f"\nResults saved to: {output_path}")
