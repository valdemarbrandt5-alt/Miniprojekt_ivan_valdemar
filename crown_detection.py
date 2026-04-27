import os
import cv2 as cv
import numpy as np

CROWN_TEMPLATE_FOLDER = "crown_templates"


def preprocess_for_crowns(image):
    if len(image.shape) == 3:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    gray = cv.GaussianBlur(gray, (3, 3), 0)
    return gray


def crop_tile_center(tile, margin_ratio=0.06):
    h, w = tile.shape[:2]
    margin_y = int(h * margin_ratio)
    margin_x = int(w * margin_ratio)

    y1 = margin_y
    y2 = h - margin_y
    x1 = margin_x
    x2 = w - margin_x

    return tile[y1:y2, x1:x2]


def safe_imread(path):
    try:
        file_bytes = np.fromfile(path, dtype=np.uint8)
        image = cv.imdecode(file_bytes, cv.IMREAD_COLOR)
        return image
    except Exception:
        return None


def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    matrix = cv.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv.warpAffine(image, matrix, (w, h), flags=cv.INTER_LINEAR)
    return rotated


def resize_template(template_img, scale):
    h, w = template_img.shape[:2]
    new_w = max(5, int(w * scale))
    new_h = max(5, int(h * scale))
    resized = cv.resize(template_img, (new_w, new_h), interpolation=cv.INTER_LINEAR)
    return resized


def load_crown_templates():
    templates = []

    if not os.path.exists(CROWN_TEMPLATE_FOLDER):
        print(f"Crown template mappe blev ikke fundet: {CROWN_TEMPLATE_FOLDER}")
        return templates

    rotation_angles = [0, 90, 180, 270]
    max_template_size = 24

    for file_name in os.listdir(CROWN_TEMPLATE_FOLDER):
        if not file_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            continue

        path = os.path.join(CROWN_TEMPLATE_FOLDER, file_name)
        image = safe_imread(path)

        if image is None:
            print(f"Kunde ikke læse template: {path}")
            continue

        processed = preprocess_for_crowns(image)
        h, w = processed.shape[:2]

        scale_factor = min(max_template_size / w, max_template_size / h, 1.0)
        new_w = max(5, int(w * scale_factor))
        new_h = max(5, int(h * scale_factor))

        processed = cv.resize(processed, (new_w, new_h), interpolation=cv.INTER_AREA)

        if processed.shape[0] < 5 or processed.shape[1] < 5:
            print(f"Springer meget lille template over: {file_name}")
            continue

        print(f"{file_name}: resized to {new_w}x{new_h}")

        for angle in rotation_angles:
            rotated = rotate_image(processed, angle)

            templates.append({
                "name": f"{file_name}_rot{angle}",
                "image": rotated
            })

    print(f"Loaded {len(templates)} crown template variants")
    return templates


def compute_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)

    inter_w = max(0, xb - xa)
    inter_h = max(0, yb - ya)
    inter_area = inter_w * inter_h

    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def center_distance(det1, det2):
    cx1 = det1["x"] + det1["w"] / 2
    cy1 = det1["y"] + det1["h"] / 2
    cx2 = det2["x"] + det2["w"] / 2
    cy2 = det2["y"] + det2["h"] / 2
    return ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5


def non_max_suppression(detections, iou_threshold=0.20, min_center_distance=12):
    if not detections:
        return []

    detections = sorted(detections, key=lambda d: d["score"], reverse=True)
    kept = []

    for det in detections:
        keep = True
        det_box = (det["x"], det["y"], det["w"], det["h"])

        for kept_det in kept:
            kept_box = (kept_det["x"], kept_det["y"], kept_det["w"], kept_det["h"])
            iou = compute_iou(det_box, kept_box)
            dist = center_distance(det, kept_det)

            if iou > iou_threshold or dist < min_center_distance:
                keep = False
                break

        if keep:
            kept.append(det)

    return kept


CROWN_TEMPLATES = load_crown_templates()


def detect_crowns_in_tile(
    tile,
    threshold=0.85,
    iou_threshold=0.20,
    min_center_distance=12,
    debug=False
):
    if not CROWN_TEMPLATES:
        return []

    tile_crop = crop_tile_center(tile, margin_ratio=0.06)
    tile_processed = preprocess_for_crowns(tile_crop)

    detections = []
    best_score_seen = -1.0

    scales = [0.75, 0.9, 1.0, 1.1, 1.25]

    for template in CROWN_TEMPLATES:
        base_template = template["image"]

        for scale in scales:
            template_img = resize_template(base_template, scale)
            th, tw = template_img.shape[:2]

            if tile_processed.shape[0] < th or tile_processed.shape[1] < tw:
                continue

            result = cv.matchTemplate(tile_processed, template_img, cv.TM_CCOEFF_NORMED)
            max_score = float(result.max())

            if max_score > best_score_seen:
                best_score_seen = max_score

            y_coords, x_coords = np.where(result >= threshold)

            for x, y in zip(x_coords, y_coords):
                score = float(result[y, x])

                detections.append({
                    "x": int(x),
                    "y": int(y),
                    "w": int(tw),
                    "h": int(th),
                    "score": score,
                    "template_name": template["name"],
                    "scale": scale
                })

    filtered = non_max_suppression(
        detections,
        iou_threshold=iou_threshold,
        min_center_distance=min_center_distance
    )

    if debug:
        print(
            f"Best score: {best_score_seen:.3f} | "
            f"Raw detections: {len(detections)} | "
            f"Filtered: {len(filtered)}"
        )

    return filtered


def get_terrain_adjusted_strong_threshold(terrain, base_threshold=0.90):
    """
    Terrain-aware tuning.
    Mine og Swamp laver ofte mere visuelt rod, så de får en lidt højere threshold.
    """
    if terrain in ["Mine", "Swamp"]:
        return base_threshold + 0.03
    return base_threshold


def predict_crown_count(
    tile,
    terrain_label=None,
    threshold=0.85,
    iou_threshold=0.20,
    min_center_distance=12,
    strong_score_threshold=0.90,
    max_crowns=3,
    debug=False
):
    if terrain_label in ["Home", "Empty"]:
        if debug:
            print(f"Terrain {terrain_label} -> forced crowns = 0")
        return 0

    adjusted_strong_threshold = get_terrain_adjusted_strong_threshold(
        terrain_label,
        base_threshold=strong_score_threshold
    )

    detections = detect_crowns_in_tile(
        tile=tile,
        threshold=threshold,
        iou_threshold=iou_threshold,
        min_center_distance=min_center_distance,
        debug=debug
    )

    blob_count = estimate_crown_blobs(tile, debug=debug)

    if not detections:
        if debug:
            print("No detections -> crowns = 0")
        return 0

    detections = sorted(detections, key=lambda d: d["score"], reverse=True)
    best_score = detections[0]["score"]

    if best_score < adjusted_strong_threshold:
        if debug:
            print(
                f"Best score: {best_score:.3f} < adjusted strong threshold "
                f"{adjusted_strong_threshold:.3f}"
            )
            print("Predicted crowns: 0")
        return 0

    strong_detections = [
        d for d in detections
        if d["score"] >= adjusted_strong_threshold
    ]

    template_count = min(len(strong_detections), max_crowns)

    # kombiner template count og blob count
    if blob_count == 0:
        crown_count = 0
    elif template_count == 0:
        crown_count = 0
    else:
        crown_count = min(template_count, blob_count, max_crowns)

    if debug:
        print(f"Terrain: {terrain_label}")
        print(f"Best score: {best_score:.3f}")
        print(f"Adjusted strong threshold: {adjusted_strong_threshold:.3f}")
        print(f"Strong detections: {len(strong_detections)}")
        print(f"Template count: {template_count}")
        print(f"Blob count: {blob_count}")
        print(f"Predicted crowns: {crown_count}")

    return crown_count

def estimate_crown_blobs(tile, debug=False):
    """
    Estimerer hvor mange crown-lignende blobs der findes i tilet.
    Returnerer et groft count signal.
    """
    tile_crop = crop_tile_center(tile, margin_ratio=0.06)
    hsv = cv.cvtColor(tile_crop, cv.COLOR_BGR2HSV)

    lower_yellow = np.array([15, 70, 90])
    upper_yellow = np.array([40, 255, 255])

    mask = cv.inRange(hsv, lower_yellow, upper_yellow)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    valid_blobs = []

    for contour in contours:
        area = cv.contourArea(contour)

        if area < 8 or area > 250:
            continue

        x, y, w, h = cv.boundingRect(contour)

        aspect_ratio = w / h if h > 0 else 0

        if aspect_ratio < 0.4 or aspect_ratio > 2.5:
            continue

        valid_blobs.append((x, y, w, h, area))

    blob_count = min(len(valid_blobs), 3)

    if debug:
        print(f"Estimated crown blobs: {blob_count}")

    return blob_count
def build_crown_grid(
    tiles,
    terrain_grid=None,
    threshold=0.85,
    iou_threshold=0.20,
    min_center_distance=12,
    strong_score_threshold=0.90,
    max_crowns=3,
    debug=False
):
    """
    Bygger et 5x5 crown grid ud fra tiles.
    Hvis terrain_grid gives, bruges terrain-aware crown detection.
    """
    crown_grid = []

    for row_index, row in enumerate(tiles):
        crown_row = []

        for col_index, tile in enumerate(row):
            terrain_label = None
            if terrain_grid is not None:
                terrain_label = terrain_grid[row_index][col_index]

            if debug:
                print(f"\nTile ({row_index}, {col_index}) terrain={terrain_label}")

            crown_count = predict_crown_count(
                tile=tile,
                terrain_label=terrain_label,
                threshold=threshold,
                iou_threshold=iou_threshold,
                min_center_distance=min_center_distance,
                strong_score_threshold=strong_score_threshold,
                max_crowns=max_crowns,
                debug=debug
            )
            crown_row.append(crown_count)

        crown_grid.append(crown_row)

    return crown_grid