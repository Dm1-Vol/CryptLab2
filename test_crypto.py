from src.crypto import encrypt_message, decrypt_message
from src.steganography import embed_payload, extract_payload


MESSAGE = (
    "This is a secret message for my cryptography laboratory. "
    "Testing AES-256-GCM and LSB steganography."
)

PASSWORD = "MyStrongPassword2026"


images = [
    "imgs/cover/gradient.png",
    "imgs/cover/map.png"
]


for image in images:
    print(f"\nTesting image: {image}")

    # AES + последовательный LSB
    encrypted = encrypt_message(MESSAGE, PASSWORD)

    embed_payload(
        image,
        "results/encrypted.png",
        encrypted
    )

    extracted = extract_payload("results/encrypted.png")
    decrypted = decrypt_message(extracted, PASSWORD)

    assert decrypted == MESSAGE

    print("Encrypted LSB: SUCCESS")

    # AES + случайный LSB
    encrypted_random = encrypt_message(MESSAGE, PASSWORD)

    embed_payload(
        image,
        "results/random.png",
        encrypted_random,
        PASSWORD
    )

    extracted_random = extract_payload(
        "results/random.png",
        PASSWORD
    )

    decrypted_random = decrypt_message(
        extracted_random,
        PASSWORD
    )

    assert decrypted_random == MESSAGE

    print("Encrypted Random LSB: SUCCESS")

print("\nALL TESTS PASSED!")