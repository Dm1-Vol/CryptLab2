from pathlib import Path

from src.crypto import encrypt_message, decrypt_message
from src.steganography import embed_payload, extract_payload
from src.analysis import (
    load_image,
    analyze_images,
    save_results,
    calculate_lsb_distribution,
    calculate_chi_square,
)

MESSAGE = (
    "This is a secret message for my cryptography laboratory. "
    "It compares plain LSB, encrypted LSB and encrypted random LSB."
)
PASSWORD = "MyStrongPassword2026"

IMAGES = [
    Path("imgs/cover/gradient.png"),
    Path("imgs/cover/map.png"),
]

RESULTS_DIR = Path("results")


def analyze_cover(image_path):
    """Собирает статистику исходного изображения."""
    image = load_image(image_path)

    result = {
        "method": "Cover",
        "mse": 0.0,
        "psnr": float("inf"),
        "ssim": 1.0,
        "changed_components": 0,
        "changed_ratio": 0.0,
    }

    result.update(calculate_lsb_distribution(image))
    result.update(calculate_chi_square(image))
    return result


def run_for_image(image_path):
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    name = image_path.stem
    message_bytes = MESSAGE.encode("utf-8")

    # Шифруем один раз: оба зашифрованных варианта используют
    # один и тот же ciphertext.
    encrypted_payload = encrypt_message(MESSAGE, PASSWORD)

    # Выравниваем размер открытого сообщения до размера ciphertext.
    # Таким образом, все три метода встраивают одинаковое число байтов.
    if len(encrypted_payload) < len(message_bytes):
        raise ValueError("Encrypted payload is unexpectedly shorter")

    plain_payload = message_bytes + (
        b"\x00" * (len(encrypted_payload) - len(message_bytes))
    )

    print(f"\nImage: {image_path}")
    print(f"Plain payload size:     {len(plain_payload)} bytes")
    print(f"Encrypted payload size: {len(encrypted_payload)} bytes")

    plain_path = RESULTS_DIR / f"{name}_plain.png"
    encrypted_path = RESULTS_DIR / f"{name}_encrypted.png"
    random_path = RESULTS_DIR / f"{name}_random.png"

    # 1. Обычный LSB: открытый текст, последовательные позиции.
    embed_payload(
        image_path,
        plain_path,
        plain_payload,
    )

    extracted_plain = extract_payload(plain_path)
    assert extracted_plain[:len(message_bytes)] == message_bytes
    print("Plain LSB: SUCCESS")

    # 2. Зашифрованный текст, последовательные позиции.
    embed_payload(
        image_path,
        encrypted_path,
        encrypted_payload,
    )

    extracted_encrypted = extract_payload(encrypted_path)
    decrypted = decrypt_message(extracted_encrypted, PASSWORD)
    assert decrypted == MESSAGE
    print("Encrypted LSB: SUCCESS")

    # 3. Зашифрованный текст, псевдослучайные позиции.
    embed_payload(
        image_path,
        random_path,
        encrypted_payload,
        password=PASSWORD,
    )

    extracted_random = extract_payload(
        random_path,
        password=PASSWORD,
    )
    decrypted_random = decrypt_message(extracted_random, PASSWORD)
    assert decrypted_random == MESSAGE
    print("Encrypted Random LSB: SUCCESS")

    # Анализируем исходное изображение и три результата.
    results = [analyze_cover(image_path)]

    results.append(
        analyze_images(image_path, plain_path, "Plain LSB")
    )
    results.append(
        analyze_images(image_path, encrypted_path, "Encrypted LSB")
    )
    results.append(
        analyze_images(image_path, random_path, "Encrypted Random LSB")
    )

    return results


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []

    for image_path in IMAGES:
        all_results.extend(run_for_image(image_path))

    save_results(
        all_results,
        RESULTS_DIR / "metrics.csv",
    )

    print("\nAll experiments completed successfully!")


if __name__ == "__main__":
    main()