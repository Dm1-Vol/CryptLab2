from pathlib import Path
import tempfile

from PIL import Image, ImageChops

from src.lsb import embed_lsb, extract_lsb


BASE_DIR = Path(__file__).resolve().parent
COVER_DIR = BASE_DIR / "imgs" / "cover"

TEST_IMAGES = [
    COVER_DIR / "gradient.png",
    COVER_DIR / "map.png",
]

TEST_MESSAGE = (
    "This is a test message for LSB steganography. "
    "The message should be embedded and extracted correctly."
)


def test_image(image_path: Path):
    print(f"\nTesting image: {image_path.name}")

    if not image_path.exists():
        raise FileNotFoundError(
            f"Cover image not found: {image_path}"
        )

    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "stego.png"

        # Embed the message into the image.
        embed_lsb(
            str(image_path),
            str(output_path),
            TEST_MESSAGE
        )

        # Check that the output image was created.
        assert output_path.exists(), (
            "Stego image was not created"
        )

        # Extract the message.
        extracted_message = extract_lsb(
            str(output_path)
        )

        # Verify the extracted message.
        assert extracted_message == TEST_MESSAGE, (
            "Extracted message does not match the original"
        )

        # Verify that the image was actually modified.
        original = Image.open(image_path).convert("RGB")
        stego = Image.open(output_path).convert("RGB")

        assert original.size == stego.size, (
            "Image dimensions changed"
        )

        difference = ImageChops.difference(
            original,
            stego
        )

        assert difference.getbbox() is not None, (
            "Image was not modified"
        )

        print("LSB embedding: SUCCESS")
        print("LSB extraction: SUCCESS")
        print("Image dimensions: OK")
        print("Image modification: CONFIRMED")


def main():
    print("LSB STEGANOGRAPHY TESTS")

    for image_path in TEST_IMAGES:
        test_image(image_path)

    print("\nALL LSB TESTS PASSED!")


if __name__ == "__main__":
    main()