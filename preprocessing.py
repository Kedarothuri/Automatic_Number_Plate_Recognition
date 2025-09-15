import cv2
import numpy as np
import os


def ensure_dir(path):
    """Ensure a folder exists, if not create it."""
    if not os.path.exists(path):
        os.makedirs(path)


def load_image(image_path):
    """Load an image from file."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    return image


def preprocess_image(image_path, save_dir="samples", show_steps=False):
    """Preprocess input image for plate detection."""
    image = load_image(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Noise removal
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Histogram equalization
    equalized = cv2.equalizeHist(blurred)

    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        equalized, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    # Ensure output directory exists
    ensure_dir(save_dir)

    # Save result
    output_path = os.path.join(save_dir, "preprocessed.jpg")
    cv2.imwrite(output_path, thresh)
    print(f"[INFO] Preprocessed image saved at {output_path}")

    if show_steps:
        cv2.imshow("Original", image)
        cv2.imshow("Gray", gray)
        cv2.imshow("Thresholded", thresh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return thresh


# Example usage
if __name__ == "__main__":
    test_image = "samples\img2.jpg"  # replace with your test image
    processed = preprocess_image(test_image, save_dir="samples", show_steps=False)
