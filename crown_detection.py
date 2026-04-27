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
    mx = int(w * margin_ratio)
    my = int(h * margin_ratio)
    return tile[my:h - my, mx:w - mx]


def safe_imread(path):
    try:
        file_bytes = np.fromfile(path, dtype=np.uint8)
        return cv.imdecode(file_bytes, cv.IMREAD_COLOR)
    except Exception:
        return None


def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    matrix = cv.getRotationMatrix2D(center, angle, 1.0)

    return cv.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv.INTER_LINEAR,
        borderMode=cv.BORDER_REPLICATE
    )


def resize_template(template_img, scale):
    h, w = template_img.shape[:2]
    new_w = max(5, int(w * scale))
    new_h = max(5, int(h * scale))
    return cv.resize(template_img, (new_w, new_h), interpolation=cv.INTER_LINEAR)


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
            print(f"Kunne ikke læse template: {path}")
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

        for angle in rotation_angles:
            rotated = rotate_image(processed, angle)
            templates.append({
                "name": f"{file_name}_rot{angle}",
                "image": rotated
            })

    print(f"Loaded {len(templates)} crown template variants")
    return templates


CROWN_TEMPLATES = load_crown_templates()


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


def non_max_suppression(detections, iou_threshold=0.20, min_center_distance=10):
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


def estimate_yellow_ratio(tile):
    tile_crop = crop_tile_center(tile, margin_ratio=0.06)
    hsv = cv.cvtColor(tile_crop, cv.COLOR_BGR2HSV)

    lower_yellow = np.array([15, 55, 75])
    upper_yellow = np.array([45, 255, 255])

    mask = cv.inRange(hsv, lower_yellow, upper_yellow)
    return float(mask.mean() / 255.0)


def estimate_crown_blobs(tile, debug=False):
    tile_crop = crop_tile_center(tile, margin_ratio=0.06)
    hsv = cv.cvtColor(tile_crop, cv.COLOR_BGR2HSV)

    lower_yellow = np.array([15, 55, 75])
    upper_yellow = np.array([45, 255, 255])

    mask = cv.inRange(hsv, lower_yellow, upper_yellow)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    valid_blobs = []

    for contour in contours:
        area = cv.contourArea(contour)

        if area < 6 or area > 280:
            continue

        x, y, w, h = cv.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0

        if aspect_ratio < 0.35 or aspect_ratio > 2.8:
            continue

        valid_blobs.append((x, y, w, h, area))

    blob_count = min(len(valid_blobs), 3)

    if debug:
        print(f"Yellow blobs: {blob_count}")

    return blob_count


def get_terrain_adjusted_thresholds(
    terrain,
    threshold=0.85,
    strong_score_threshold=0.90
):
    if terrain in ["Home", "Empty"]:
        return 1.0, 1.0

    if terrain in ["Mine", "Swamp"]:
        return threshold + 0.03, strong_score_threshold + 0.03

    return threshold, strong_score_threshold


def detect_crowns_in_tile(
    tile,
    threshold=0.85,
    iou_threshold=0.20,
    min_center_distance=10,
    debug=False
):
    if not CROWN_TEMPLATES:
        return []

    tile_crop = crop_tile_center(tile, margin_ratio=0.06)
    tile_processed = preprocess_for_crowns(tile_crop)

    detections = []
    best_score_seen = -1.0

    scales = [0.75, 0.90, 1.0, 1.10, 1.25]

    for template in CROWN_TEMPLATES:
        base_template = template["image"]

        for scale in scales:
            template_img = resize_template(base_template, scale)
            th, tw = template_img.shape[:2]

            if tile_processed.shape[0] < th or tile_processed.shape[1] < tw:
                continue

            result = cv.matchTemplate(tile_processed, template_img, cv.TM_CCOEFF_NORMED)
            max_score = float(result.max())
            best_score_seen = max(best_score_seen, max_score)

            y_coords, x_coords = np.where(result >= threshold)

            for x, y in zip(x_coords, y_coords):
                detections.append({
                    "x": int(x),
                    "y": int(y),
                    "w": int(tw),
                    "h": int(th),
                    "score": float(result[y, x]),
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
            f"Raw: {len(detections)} | "
            f"Filtered: {len(filtered)}"
        )

    return filtered


def predict_crown_count(
    tile,
    terrain_label=None,
    threshold=0.85,
    iou_threshold=0.20,
    min_center_distance=10,
    strong_score_threshold=0.90,
    max_crowns=3,
    debug=False
):
    if terrain_label in ["Home", "Empty"]:
        return 0

    yellow_ratio = estimate_yellow_ratio(tile)

    # Hvis der næsten ingen gul crown-farve er, er det næsten sikkert 0.
    # Men thresholden er lav, så vi ikke dræber rigtige crowns for hårdt.
    if yellow_ratio < 0.002:
        if debug:
            print(f"Yellow ratio too low: {yellow_ratio:.4f} -> 0")
        return 0

    adjusted_threshold, adjusted_strong_threshold = get_terrain_adjusted_thresholds(
        terrain_label,
        threshold=threshold,
        strong_score_threshold=strong_score_threshold
    )

    detections = detect_crowns_in_tile(
        tile=tile,
        threshold=adjusted_threshold,
        iou_threshold=iou_threshold,
        min_center_distance=min_center_distance,
        debug=debug
    )

    blob_count = estimate_crown_blobs(tile, debug=debug)

    if not detections:
        return 0

    detections = sorted(detections, key=lambda d: d["score"], reverse=True)
    best_score = detections[0]["score"]

    strong_detections = [
        d for d in detections
        if d["score"] >= adjusted_strong_threshold
    ]

    template_count = min(len(strong_detections), max_crowns)

    # Ny mere konservativ kombination:
    # Template er primært signal.
    # Blob count bruges til at stoppe åbenlyse overcounts.
    if template_count == 0:
        crown_count = 0
    elif blob_count == 0 and best_score < adjusted_strong_threshold + 0.05:
        crown_count = 0
    elif blob_count > 0:
        crown_count = min(template_count, max(blob_count, 1), max_crowns)
    else:
        crown_count = min(template_count, max_crowns)

    if debug:
        print(f"Terrain: {terrain_label}")
        print(f"Yellow ratio: {yellow_ratio:.4f}")
        print(f"Adjusted threshold: {adjusted_threshold:.3f}")
        print(f"Adjusted strong threshold: {adjusted_strong_threshold:.3f}")
        print(f"Best score: {best_score:.3f}")
        print(f"Template count: {template_count}")
        print(f"Blob count: {blob_count}")
        print(f"Predicted crowns: {crown_count}")

    return crown_count


def build_crown_grid(
    tiles,
    terrain_grid=None,
    threshold=0.85,
    iou_threshold=0.20,
    min_center_distance=10,
    strong_score_threshold=0.90,
    max_crowns=3,
    debug=False
):
    crown_grid = []

    for row_index, row in enumerate(tiles):
        crown_row = []

        for col_index, tile in enumerate(row):
            terrain_label = None

            if terrain_grid is not None:
                terrain_label = terrain_grid[row_index][col_index]

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