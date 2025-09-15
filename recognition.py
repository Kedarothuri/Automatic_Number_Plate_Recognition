import os
import cv2
import numpy as np
import pytesseract
from typing import List, Tuple

# Update this path if Tesseract is in a different location on your machine
# On Windows for example:
# pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"
# Uncomment & edit below if necessary:
# pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"


# ----------------------------
# Preprocessing helpers
# ----------------------------
def preprocess_char(img: np.ndarray, scale: int = 6, pad: int = 10) -> np.ndarray:
    """
    Preprocess single character image for tesseract:
    - Ensure grayscale
    - Threshold
    - Invert if text is white-on-black (Tesseract prefers black text on white bg)
    - Resize (scale factor)
    - Add padding
    - Optional morphological clean
    """
    if img is None:
        raise ValueError("Empty image provided to preprocess_char")

    # Ensure grayscale
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # If already binary-like, keep; otherwise use OTSU
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    # Apply OTSU thresholding for robust binary
    _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Determine whether text is white-on-black or black-on-white
    white_ratio = np.count_nonzero(th) / th.size
    # if white_ratio is small -> likely white strokes on dark background (invert)
    if white_ratio < 0.5:
        th = cv2.bitwise_not(th)

    # Clean small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)

    # Resize up for OCR - enlarge by `scale`
    h, w = th.shape
    new_size = (max(32, w * scale), max(64, h * scale))
    th_resized = cv2.resize(th, new_size, interpolation=cv2.INTER_LINEAR)

    # Add border/padding
    th_padded = cv2.copyMakeBorder(th_resized, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)

    return th_padded


# ----------------------------
# OCR + correction helpers
# ----------------------------
def ocr_char_with_confidence(img: np.ndarray, whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789") -> Tuple[str, int]:
    """
    Returns (predicted_char, confidence_percent) using pytesseract image_to_data.
    If no text is found, returns ('', -1).
    """
    # PSM 8 = treat image as a single word (better for single char sometimes)
    config = f"--psm 8 --oem 3 -c tessedit_char_whitelist={whitelist}"

    # image_to_data gives confidence per box
    data = pytesseract.image_to_data(img, config=config, output_type=pytesseract.Output.DICT)

    texts = data.get("text", [])
    confs = data.get("conf", [])

    # Find the first non-empty text with best confidence
    best_char = ""
    best_conf = -1
    for txt, cf in zip(texts, confs):
        if txt and txt.strip():
            # sometimes conf is string or '-1' when not found
            try:
                cf_val = int(float(cf))
            except:
                cf_val = -1
            if cf_val > best_conf:
                best_conf = cf_val
                best_char = txt.strip()

    # Clean result to single character
    if best_char:
        # Take first character only
        best_char = best_char[0].upper()
    else:
        best_conf = -1

    return best_char, best_conf


# Simple mapping for common OCR confusions specific to Indian license plates
DIGIT_MAP = {
    'O': '0', 'D': '0', 'Q': '0', 'U': '0', 'V': '0', 'C': '0',
    'I': '1', 'L': '1', 'Z': '2', 'S': '5', 'B': '8', 'G': '6', 'F': '6', '1': '0'
}
LETTER_MAP = {
    '0': 'O', '1': 'I', '5': 'S', '8': 'B', '2': 'Z', '6': 'G', '7': 'T', 'I': 'H', 'F': 'E', 'L': 'H', '1': 'H'
}


def correct_char_by_pattern(ch: str, expected: str) -> str:
    """
    Correct a single character according to expected type:
    expected: 'N' for number, 'L' for letter, '?' for either
    """
    if not ch:
        # If no character recognized, try to provide a fallback based on pattern
        if expected == 'N':
            return '0'  # Default to 0 for numbers
        elif expected == 'L':
            return 'A'  # Default to A for letters
        else:
            return '?'
    if expected == 'N':
        corrected = DIGIT_MAP.get(ch, ch)
        if corrected.isdigit():
            return corrected
        return ch if ch.isdigit() else '0'
    elif expected == 'L':
        corrected = LETTER_MAP.get(ch, ch)
        if corrected.isalpha():
            return corrected
        return ch if ch.isalpha() else 'A'
    else:
        return ch


# ----------------------------
# Main recognition function
# ----------------------------
def recognize_plate_from_dir(segmented_dir: str = "segmented_chars", pattern: str = None,
                             whitelist: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                             debug_save_dir: str = "ocr_debug") -> Tuple[str, List[Tuple[str, int]]]:
    """
    Recognize full plate string from a directory containing segmented chars.
    Returns (plate_string, list_of_tuples[(char, confidence), ...])
    Optional:
      - pattern: str of same length as number of chars (L/N/?)
    """

    if not os.path.exists(segmented_dir):
        raise FileNotFoundError(f"Segmented directory not found: {segmented_dir}")

    # Collect files that look like char_1.jpg, char_2.jpg etc
    files = [f for f in os.listdir(segmented_dir) if os.path.isfile(os.path.join(segmented_dir, f))]
    if not files:
        raise ValueError("No files found in segmented directory.")

    # Sort by index extracted from filename (assumes format char_1.jpg)
    def idx_from_name(fn):
        parts = fn.split("_")
        for p in parts[::-1]:
            try:
                return int(''.join(filter(str.isdigit, p)))
            except:
                continue
        return 0

    files = sorted(files, key=idx_from_name)

    # Prepare debug canvas: we'll stitch the characters horizontally and annotate
    char_imgs = []
    results = []
    for i, fname in enumerate(files):
        path = os.path.join(segmented_dir, fname)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        proc = preprocess_char(img)
        pred, conf = ocr_char_with_confidence(proc, whitelist=whitelist)

        # Apply pattern correction if pattern provided
        if pattern and i < len(pattern):
            expected = pattern[i].upper()  # 'L' or 'N' or '?'
            pred_corrected = correct_char_by_pattern(pred, expected)
        else:
            pred_corrected = pred

        results.append((pred_corrected, conf))
        char_imgs.append(proc)

    # Build plate string
    plate_chars = [r[0] if r[0] else '?' for r in results]
    plate_string = "".join(plate_chars)

    # Save debug visualization
    os.makedirs(debug_save_dir, exist_ok=True)
    # create a wide image that shows each char and label
    heights = [img.shape[0] for img in char_imgs] if char_imgs else [0]
    max_h = max(heights) if heights else 0
    padded_imgs = []
    for img in char_imgs:
        h, w = img.shape
        if h < max_h:
            pad_v = max_h - h
            top = pad_v // 2
            bottom = pad_v - top
            img = cv2.copyMakeBorder(img, top, bottom, 10, 10, cv2.BORDER_CONSTANT, value=255)
        else:
            img = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
        # convert to BGR for annotation
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        padded_imgs.append(bgr)

    if padded_imgs:
        debug_canvas = cv2.hconcat(padded_imgs)
        # annotate below each char with recognized char + conf
        x_offset = 0
        for (pred_char, conf), img in zip(results, padded_imgs):
            h, w = img.shape[:2]
            label = f"{pred_char if pred_char else '?'} ({conf})"
            cv2.putText(debug_canvas, label, (x_offset + 5, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            x_offset += w
        debug_path = os.path.join(debug_save_dir, f"ocr_debug_{len(files)}_{plate_string}.jpg")
        cv2.imwrite(debug_path, debug_canvas)
        print(f"[DEBUG] OCR debug image saved to: {debug_path}")

    # Print detailed per-character output
    for i,(ch,conf) in enumerate(results):
        print(f"[CHAR {i+1}] -> '{ch}'  conf={conf}")

    print(f"[RESULT] Final plate: {plate_string}")
    return plate_string, results


# ----------------------------
# Quick test run (if executed directly)
# ----------------------------
if __name__ == "__main__":
    # Example usage
    # If your plate is like "KA01AB1234", you could pass a pattern like:
    pattern = "LLNNLLNNNN"  # L=letter, N=number, ?=either
    plate, details = recognize_plate_from_dir(segmented_dir="segmented_chars", pattern=pattern)
 