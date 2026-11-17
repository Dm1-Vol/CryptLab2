import hashlib

import numpy as np
from PIL import Image

from src.lsb import bytes_to_bits, bits_to_bytes


HEADER_SIZE = 4


def get_positions(total: int, count: int, password=None):
    if count > total:
        raise ValueError("Недостаточно места для сообщения")

    if password is None:
        return np.arange(count)

    seed = int.from_bytes(
        hashlib.sha256(
            ("LSB-position:" + password).encode("utf-8")
        ).digest(),
        "big"
    )

    rng = np.random.default_rng(seed)

    return rng.permutation(total)[:count]


def embed_payload(
    image_path: str,
    output_path: str,
    payload: bytes,
    password=None
):
    image = Image.open(image_path).convert("RGB")
    pixels = np.array(image, dtype=np.uint8)

    flat = pixels.reshape(-1)

    data = len(payload).to_bytes(HEADER_SIZE, "big") + payload
    bits = bytes_to_bits(data)

    if len(bits) > len(flat):
        raise ValueError("Сообщение не помещается в изображение")

    positions = get_positions(
        len(flat),
        len(bits),
        password
    )

    flat[positions] = (
        (flat[positions] & 0xFE) | np.array(bits, dtype=np.uint8)
    )

    Image.fromarray(pixels).save(output_path, format="PNG")


def extract_payload(image_path: str, password=None) -> bytes:
    image = Image.open(image_path).convert("RGB")
    flat = np.array(image, dtype=np.uint8).reshape(-1)

    header_positions = get_positions(
        len(flat),
        HEADER_SIZE * 8,
        password
    )

    header_bits = [
        int(flat[pos]) & 1
        for pos in header_positions
    ]

    header = bits_to_bytes(header_bits)
    payload_length = int.from_bytes(header, "big")

    total_bits = (HEADER_SIZE + payload_length) * 8

    if total_bits > len(flat):
        raise ValueError("Некорректная длина сообщения")

    positions = get_positions(
        len(flat),
        total_bits,
        password
    )

    payload_bits = [
        int(flat[pos]) & 1
        for pos in positions[HEADER_SIZE * 8:]
    ]

    return bits_to_bytes(payload_bits)