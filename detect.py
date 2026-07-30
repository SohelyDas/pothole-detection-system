import os
from ultralytics import YOLO
import cv2

# Classes 1, 2, 6 are all "pothole" (duplicates from messy labels)
# Classes 0, 3, 4, 5 are junk/noise — ignore them
POTHOLE_CLASSES = {1, 2, 6}


def load_model(weights_path='models/best.pt'):
    return YOLO(weights_path)


def get_severity(area_ratio):
    if area_ratio > 0.15:
        return "Severe"
    elif area_ratio > 0.05:
        return "Moderate"
    else:
        return "Minor"


def detect_potholes(model, image_path, conf=0.25):
    """Run detection on a single image, return real pothole detections with severity."""
    results = model(image_path, conf=conf, imgsz=640)
    detections = []

    for r in results:
        if r.boxes is None:
            continue
        img_h, img_w = r.orig_shape

        for box in r.boxes:
            cls_id = int(box.cls)
            if cls_id not in POTHOLE_CLASSES:
                continue

            confidence = float(box.conf)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            box_area = (x2 - x1) * (y2 - y1)
            area_ratio = box_area / (img_w * img_h)

            detections.append({
                "confidence": round(confidence, 2),
                "severity": get_severity(area_ratio),
                "area_ratio": round(area_ratio, 4),
                "bbox": [round(x1), round(y1), round(x2), round(y2)]
            })

    return detections, results[0]


def draw_and_save(result, out_path):
    """Draw a solid translucent red segmentation mask + outline + label, and save."""
    img = result.orig_img.copy()

    if result.masks is not None and result.boxes is not None:
        for mask, box in zip(result.masks.xy, result.boxes):
            cls_id = int(box.cls)
            if cls_id not in POTHOLE_CLASSES:
                continue  # skip junk classes entirely — don't draw them

            confidence = float(box.conf)
            pts = mask.astype(int)

            # Draw filled translucent mask
            overlay = img.copy()
            cv2.fillPoly(overlay, [pts], color=(0, 255, 255))  # yellow mask (BGR)
            img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)

            # Draw mask outline
            cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 255), thickness=2)

            # Draw label with percentage
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            label = f"Pothole {confidence*100:.0f}%"
            cv2.putText(img, label, (int(x1), max(int(y1) - 10, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imwrite(out_path, img)


def annotate_image(result):
    """Same drawing as draw_and_save, but returns an RGB array instead of saving to disk.
    Used by the Streamlit app to display the annotated image directly."""
    img = result.orig_img.copy()

    if result.masks is not None and result.boxes is not None:
        for mask, box in zip(result.masks.xy, result.boxes):
            cls_id = int(box.cls)
            if cls_id not in POTHOLE_CLASSES:
                continue

            confidence = float(box.conf)
            pts = mask.astype(int)

            overlay = img.copy()
            cv2.fillPoly(overlay, [pts], color=(0, 255, 255))
            img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)

            cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 255), thickness=2)

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            label = f"Pothole {confidence*100:.0f}%"
            cv2.putText(img, label, (int(x1), max(int(y1) - 10, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def run_batch(image_dir="images", out_dir="runs/pothole_results", conf=0.25):
    os.makedirs(out_dir, exist_ok=True)
    model = load_model()

    image_files = [f for f in os.listdir(image_dir)
                   if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    if not image_files:
        print(f"No images found in '{image_dir}'.")
        return

    for fname in image_files:
        path = os.path.join(image_dir, fname)
        detections, result = detect_potholes(model, path, conf=conf)

        out_path = os.path.join(out_dir, fname)
        draw_and_save(result, out_path)

        print(f"\n{fname}:")
        if not detections:
            print("  No potholes detected.")
        else:
            for d in detections:
                print(f"  {d}")

    print(f"\nAnnotated results saved to: {out_dir}")


if __name__ == "__main__":
    run_batch()