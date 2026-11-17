
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.crypto import encrypt_message, decrypt_message
from src.steganography import embed_payload, extract_payload
from src.analysis import (
    load_image,
    analyze_images,
    analyze_cover,
)

IMAGE_PATH = Path("imgs/cover/gradient.png")
RESULTS_DIR = Path("results")
PLOTS_DIR = RESULTS_DIR / "plots"

PASSWORD = "MyStrongPassword2026"
FILL_LEVELS = [0.01, 0.05]

# AES-GCM payload overhead:
# salt (16) + nonce (12) + authentication tag (16)
ENCRYPTION_OVERHEAD = 44
HEADER_SIZE = 4


def create_payloads(image_path, fill_ratio):
    """
    Creates plaintext and encrypted payloads of exactly
    the same size.
    """
    image = load_image(image_path)
    total_components = image.size

    # Total embedded bits, including the 4-byte header.
    total_bits = int(total_components * fill_ratio)
    total_bytes = total_bits // 8

    payload_size = total_bytes - HEADER_SIZE

    if payload_size <= ENCRYPTION_OVERHEAD:
        raise ValueError("Image is too small for this fill level")

    plaintext_size = payload_size - ENCRYPTION_OVERHEAD

    message = "A" * plaintext_size
    message_bytes = message.encode("utf-8")

    encrypted_payload = encrypt_message(message, PASSWORD)

    # Pad plaintext so all methods embed exactly the same
    # number of bytes.
    plain_payload = message_bytes + (
        b"\x00" * (payload_size - len(message_bytes))
    )

    assert len(plain_payload) == len(encrypted_payload)

    return plain_payload, encrypted_payload, message


def create_difference_map(original_path, stego_path, output_path):
    """Creates an amplified absolute difference image."""
    original = load_image(original_path).astype(np.int16)
    stego = load_image(stego_path).astype(np.int16)

    difference = np.abs(original - stego)

    # Amplify differences so that changed pixels are visible.
    amplified = np.clip(difference * 255, 0, 255).astype(np.uint8)

    from PIL import Image
    Image.fromarray(amplified).save(output_path)


def create_histogram(original_path, stego_paths, output_path):
    """Compares RGB intensity distributions."""
    original = load_image(original_path)

    plt.figure(figsize=(12, 6))

    plt.hist(
        original.flatten(),
        bins=256,
        range=(0, 256),
        density=True,
        histtype="step",
        label="Cover",
    )

    for label, path in stego_paths:
        stego = load_image(path)

        plt.hist(
            stego.flatten(),
            bins=256,
            range=(0, 256),
            density=True,
            histtype="step",
            label=label,
        )

    plt.xlabel("Pixel intensity")
    plt.ylabel("Normalized frequency")
    plt.title("Cover and stego image intensity distributions")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def run_volume_experiment():
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    RESULTS_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)

    all_results = []

    print(f"Source image: {IMAGE_PATH}")

    for fill_ratio in FILL_LEVELS:
        percentage = int(fill_ratio * 100)

        print(f"\n--- Fill level: {percentage}% ---")

        plain_payload, encrypted_payload, message = create_payloads(
            IMAGE_PATH,
            fill_ratio,
        )

        print(f"Payload size: {len(encrypted_payload)} bytes")

        plain_path = RESULTS_DIR / f"gradient_{percentage}pct_plain.png"
        encrypted_path = RESULTS_DIR / f"gradient_{percentage}pct_encrypted.png"
        random_path = RESULTS_DIR / f"gradient_{percentage}pct_random.png"

        # 1. Plain LSB
        embed_payload(
            IMAGE_PATH,
            plain_path,
            plain_payload,
        )

        extracted = extract_payload(plain_path)
        assert extracted[:len(message)] == message.encode("utf-8")

        print("Plain LSB: SUCCESS")

        # 2. Encrypted LSB
        embed_payload(
            IMAGE_PATH,
            encrypted_path,
            encrypted_payload,
        )

        extracted = extract_payload(encrypted_path)
        assert decrypt_message(extracted, PASSWORD) == message

        print("Encrypted LSB: SUCCESS")

        # 3. Encrypted Random LSB
        embed_payload(
            IMAGE_PATH,
            random_path,
            encrypted_payload,
            password=PASSWORD,
        )

        extracted = extract_payload(
            random_path,
            password=PASSWORD,
        )
        assert decrypt_message(extracted, PASSWORD) == message

        print("Encrypted Random LSB: SUCCESS")

        # Analyze all three methods.
        experiments = [
            ("Plain LSB", plain_path),
            ("Encrypted LSB", encrypted_path),
            ("Encrypted Random LSB", random_path),
        ]

        for method, stego_path in experiments:
            result = analyze_images(
                IMAGE_PATH,
                stego_path,
                method,
            )

            result["fill_percent"] = percentage
            result["image"] = IMAGE_PATH.name
            result["payload_bytes"] = len(encrypted_payload)

            all_results.append(result)

        # Difference maps for each method and fill level.
        for method, stego_path in experiments:
            safe_method = method.lower().replace(" ", "_")

            difference_path = (
                PLOTS_DIR /
                f"gradient_{percentage}pct_{safe_method}_difference.png"
            )

            create_difference_map(
                IMAGE_PATH,
                stego_path,
                difference_path,
            )

        print(f"Analysis for {percentage}% completed.")

    # Save volume experiment results.
    df = pd.DataFrame(all_results)

    output_csv = RESULTS_DIR / "volume_metrics.csv"
    df.to_csv(
        output_csv,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nVolume experiment results:")
    print(df.to_string(index=False))
    print(f"\nSaved: {output_csv}")

    # Create histogram for the 5% experiment.
    histogram_paths = [
        (
            "Plain LSB 5%",
            RESULTS_DIR / "gradient_5pct_plain.png",
        ),
        (
            "Encrypted LSB 5%",
            RESULTS_DIR / "gradient_5pct_encrypted.png",
        ),
        (
            "Encrypted Random LSB 5%",
            RESULTS_DIR / "gradient_5pct_random.png",
        ),
    ]

    create_histogram(
        IMAGE_PATH,
        histogram_paths,
        PLOTS_DIR / "gradient_histogram_comparison.png",
    )

    print("\nAll visualizations created.")
    print(f"Plots directory: {PLOTS_DIR}")


if __name__ == "__main__":
    run_volume_experiment()
