"""
segmentation.py
---------------
"""
import cv2
import numpy as np
import os


def segment_characters(plate_image, save_dir="segmented_chars", show_steps=False):


    if plate_image is None:
        raise ValueError("Plate image is None, check input.")

    # Convert to grayscale
    gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Apply binary thresholding (invert: text is white, background is black)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological operations to clean noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Find contours (potential characters)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    char_regions = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = h / float(w)

        # Filter out non-character contours by size & shape
        if 1.2 < aspect_ratio < 6.0 and 100 > w > 10 and 200 > h > 20:
            char_regions.append((x, y, w, h))

    # Sort left-to-right
    char_regions = sorted(char_regions, key=lambda x: x[0])

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    segmented_chars = []
    for i, (x, y, w, h) in enumerate(char_regions):
        char_img = thresh[y:y + h, x:x + w]
        char_img = cv2.resize(char_img, (40, 80))  # Normalize size
        segmented_chars.append(char_img)

        # Save segmented character
        save_path = os.path.join(save_dir, f"char_{i+1}.jpg")
        cv2.imwrite(save_path, char_img)

    # Debug visualization
    if show_steps:
        cv2.imshow("Plate", plate_image)
        cv2.imshow("Thresholded", thresh)
        for i, ch in enumerate(segmented_chars):
            cv2.imshow(f"Char {i+1}", ch)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print(f"[INFO] Segmented {len(segmented_chars)} characters saved in '{save_dir}'")

    return segmented_chars


# Example usage
if __name__ == "__main__":
    test_plate = cv2.imread("samples/plate_1.jpg")  # cropped plate from detection.py
    chars = segment_characters(test_plate, save_dir="segmented_chars", show_steps=True)