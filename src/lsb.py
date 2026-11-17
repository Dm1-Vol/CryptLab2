from PIL import Image
import numpy as np


HEADER_SIZE = 4  # 4 байта для хранения длины сообщения


def bytes_to_bits(data: bytes) -> list[int]:
    return [
        (byte >> shift) & 1
        for byte in data
        for shift in range(7, -1, -1)
    ]


def bits_to_bytes(bits: list[int]) -> bytes:
    if len(bits) % 8 != 0:
        raise ValueError("Количество бит должно быть кратно 8")

    result = bytearray()

    for i in range(0, len(bits), 8):
        byte = 0
        for bit in bits[i:i + 8]:
            byte = (byte << 1) | bit
        result.append(byte)

    return bytes(result)


def embed_lsb(image_path: str, output_path: str, message: str):
    image = Image.open(image_path).convert("RGB")
    pixels = np.array(image, dtype=np.uint8)

    message_bytes = message.encode("utf-8")

    payload = (
        len(message_bytes).to_bytes(HEADER_SIZE, "big")
        + message_bytes
    )

    bits = bytes_to_bits(payload)

    flat = pixels.reshape(-1)

    if len(bits) > len(flat):
        raise ValueError(
            f"Сообщение слишком большое: требуется {len(bits)} бит, "
            f"доступно {len(flat)} бит"
        )

    for i, bit in enumerate(bits):
        flat[i] = (int(flat[i]) & 0xFE) | bit

    result = Image.fromarray(pixels)
    result.save(output_path, format="PNG")

    print(f"Сообщение встроено: {len(message_bytes)} байт")
    print(f"Изображение сохранено: {output_path}")


def extract_lsb(image_path: str) -> str:
    image = Image.open(image_path).convert("RGB")
    pixels = np.array(image, dtype=np.uint8).reshape(-1)

    # Сначала извлекаем заголовок
    header_bits = [
        int(pixel) & 1
        for pixel in pixels[:HEADER_SIZE * 8]
    ]

    header = bits_to_bytes(header_bits)
    message_length = int.from_bytes(header, "big")

    required_bits = (HEADER_SIZE + message_length) * 8

    if required_bits > len(pixels):
        raise ValueError("Некорректная длина сообщения")

    # Затем извлекаем сообщение
    message_bits = [
        int(pixel) & 1
        for pixel in pixels[HEADER_SIZE * 8:required_bits]
    ]

    message_bytes = bits_to_bytes(message_bits)

    return message_bytes.decode("utf-8")