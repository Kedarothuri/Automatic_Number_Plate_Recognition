import cv2
import os
from ultralytics import YOLO


def ensure_dir(path):
    """Ensure a folder exists, if not create it."""
    if not os.path.exists(path):
        os.makedirs(path)


# Load YOLOv8 model (use your pretrained license plate weights)
MODEL_PATH = "best.pt"  # update if needed
model = YOLO(MODEL_PATH)


def detect_plate_yolo(image_path, save_dir="samples", show_result=False):
    """Detect license plates using YOLOv8."""
    results = model.predict(source=image_path, conf=0.25, save=False)

    cropped_plates = []
    image = cv2.imread(image_path)

    # Ensure save directory exists
    ensure_dir(save_dir)

    for result in results:
        for box in result.boxes:
            # Extract bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cropped_plate = image[y1:y2, x1:x2]
            cropped_plates.append(cropped_plate)

            # Draw rectangle on image
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Save detection result
    output_path = os.path.join(save_dir, "yolo_detection.jpg")
    cv2.imwrite(output_path, image)
    print(f"[INFO] Detection result saved at {output_path}")

    # Save cropped plates
    for i, plate in enumerate(cropped_plates):
        save_path = os.path.join(save_dir, f"plate_{i+1}.jpg")
        cv2.imwrite(save_path, plate)
        print(f"[INFO] Cropped plate saved as {save_path}")

    # Optionally show result
    if show_result:
        cv2.imshow("YOLOv8 Detection", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return cropped_plates


# Example usage
if __name__ == "__main__":
    test_image = "samples\preprocessed.jpg"  # replace with your test image
    plates = detect_plate_yolo(test_image, save_dir="samples", show_result=False)

    if not plates:
        print("[WARN] No plates detected.")
