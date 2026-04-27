import os
import random
from collections import defaultdict

import cv2 as cv

from ground_truth_progress import GROUND_TRUTH
from image_processing import get_tiles, predict_terrain_grid_from_image
from crown_detection import build_crown_grid

GRID_SIZE = 5
BOARD_SIZE = 500
DATASET_FOLDER = "king_domino_dataset"


def get_image_files(dataset_folder):
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")

    image_files = [
        f for f in os.listdir(dataset_folder)
        if f.lower().endswith(valid_extensions)
    ]

    image_files.sort(
        key=lambda x: int(os.path.splitext(x)[0])
        if os.path.splitext(x)[0].isdigit()
        else x
    )

    return image_files


def split_dataset(image_files, train_ratio=0.8, seed=42):
    shuffled = image_files.copy()
    random.seed(seed)
    random.shuffle(shuffled)

    split_index = int(len(shuffled) * train_ratio)

    train_files = shuffled[:split_index]
    test_files = shuffled[split_index:]

    return train_files, test_files


def compare_crown_grids(predicted_grid, true_grid, confusion):
    correct = 0
    total = 0
    mistakes = []

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            predicted = predicted_grid[row][col]
            truth = true_grid[row][col]

            confusion[truth][predicted] += 1
            total += 1

            if predicted == truth:
                correct += 1
            else:
                mistakes.append((row, col, truth, predicted))

    accuracy = correct / total if total > 0 else 0.0

    return correct, total, accuracy, mistakes


def evaluate_crowns_return_accuracy(
    files_to_evaluate,
    threshold=0.85,
    iou_threshold=0.20,
    min_center_distance=12,
    strong_score_threshold=0.90,
    max_crowns=3,
    debug=False,
    home_template_threshold=0.80,
    home_hsv_max_distance=3.0,
    template_weight=1.0,
    hsv_weight=0.15
):
    total_correct = 0
    total_tiles = 0
    evaluated_boards = 0

    confusion = defaultdict(lambda: defaultdict(int))

    for file_name in files_to_evaluate:
        board_name = os.path.splitext(file_name)[0]

        if board_name not in GROUND_TRUTH:
            continue

        image_path = os.path.join(DATASET_FOLDER, file_name)
        image = cv.imread(image_path)

        if image is None:
            continue

        tiles = get_tiles(
            image,
            board_size=BOARD_SIZE,
            grid_size=GRID_SIZE
        )

        terrain_grid = predict_terrain_grid_from_image(
            image=image,
            board_size=BOARD_SIZE,
            grid_size=GRID_SIZE,
            debug=False,
            home_template_threshold=home_template_threshold,
            home_hsv_max_distance=home_hsv_max_distance,
            template_weight=template_weight,
            hsv_weight=hsv_weight
        )

        predicted_grid = build_crown_grid(
            tiles,
            terrain_grid=terrain_grid,
            threshold=threshold,
            iou_threshold=iou_threshold,
            min_center_distance=min_center_distance,
            strong_score_threshold=strong_score_threshold,
            max_crowns=max_crowns,
            debug=debug
        )

        true_grid = GROUND_TRUTH[board_name]["crowns"]

        correct, total, accuracy, mistakes = compare_crown_grids(
            predicted_grid,
            true_grid,
            confusion
        )

        total_correct += correct
        total_tiles += total
        evaluated_boards += 1

    if total_tiles == 0:
        return 0.0, 0, 0, confusion

    overall_accuracy = total_correct / total_tiles

    return overall_accuracy, total_correct, total_tiles, confusion


def print_confusion_matrix(confusion):
    labels = sorted(
        set(confusion.keys()) |
        {pred for row in confusion.values() for pred in row.keys()}
    )

    if not labels:
        print("Ingen confusion matrix data.")
        return

    print("\nConfusion Matrix")
    print("=" * 70)

    header = "true\\pred".ljust(12) + " ".join(str(label).ljust(8) for label in labels)
    print(header)

    for true_label in labels:
        row_values = []

        for pred_label in labels:
            row_values.append(confusion[true_label].get(pred_label, 0))

        row_text = " ".join(str(value).ljust(8) for value in row_values)
        print(str(true_label).ljust(12) + row_text)


def threshold_sweep(
    thresholds=None,
    train_ratio=0.8,
    seed=42,
    evaluate_on_test_only=True,
    iou_threshold=0.20,
    min_center_distance=12,
    strong_score_threshold=0.90,
    max_crowns=3,
    debug=False,
    home_template_threshold=0.80,
    home_hsv_max_distance=3.0,
    template_weight=1.0,
    hsv_weight=0.15
):
    if thresholds is None:
        thresholds = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

    if not os.path.exists(DATASET_FOLDER):
        print(f"Dataset mappe blev ikke fundet: {DATASET_FOLDER}")
        return

    image_files = get_image_files(DATASET_FOLDER)

    if not image_files:
        print("Ingen billeder fundet i dataset mappen.")
        return

    train_files, test_files = split_dataset(
        image_files,
        train_ratio=train_ratio,
        seed=seed
    )

    if evaluate_on_test_only:
        files_to_evaluate = test_files
        split_text = "test split"
    else:
        files_to_evaluate = image_files
        split_text = "hele datasættet"

    print("=" * 70)
    print("Threshold sweep for crown detection")
    print("=" * 70)
    print(f"Seed: {seed}")
    print(f"Train boards: {len(train_files)}")
    print(f"Test boards: {len(test_files)}")
    print(f"Evaluerer på: {split_text}")
    print(f"Thresholds: {thresholds}")
    print("=" * 70)

    results = []

    for threshold in thresholds:
        accuracy, correct, total, confusion = evaluate_crowns_return_accuracy(
            files_to_evaluate=files_to_evaluate,
            threshold=threshold,
            iou_threshold=iou_threshold,
            min_center_distance=min_center_distance,
            strong_score_threshold=strong_score_threshold,
            max_crowns=max_crowns,
            debug=debug,
            home_template_threshold=home_template_threshold,
            home_hsv_max_distance=home_hsv_max_distance,
            template_weight=template_weight,
            hsv_weight=hsv_weight
        )

        results.append({
            "threshold": threshold,
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "confusion": confusion
        })

        print(f"Threshold {threshold:.2f}: {correct}/{total} = {accuracy:.2%}")

    best_result = max(results, key=lambda r: r["accuracy"])

    print("\n" + "=" * 70)
    print("Threshold sweep result")
    print("=" * 70)

    for result in results:
        threshold = result["threshold"]
        accuracy = result["accuracy"]
        correct = result["correct"]
        total = result["total"]

        marker = "<-- BEDST" if result == best_result else ""

        print(
            f"Threshold {threshold:.2f}: "
            f"{correct}/{total} = {accuracy:.2%} {marker}"
        )

    print("\nBedste threshold:")
    print(f"Threshold = {best_result['threshold']:.2f}")
    print(f"Accuracy  = {best_result['accuracy']:.2%}")
    print(f"Korrekt   = {best_result['correct']}/{best_result['total']}")

    print_confusion_matrix(best_result["confusion"])


if __name__ == "__main__":
    threshold_sweep(
        thresholds=[
            0.70,
            0.75,
            0.80,
            0.85,
            0.90,
            0.95
        ],
        train_ratio=0.8,
        seed=42,
        evaluate_on_test_only=True,
        iou_threshold=0.20,
        min_center_distance=12,
        strong_score_threshold=0.90,
        max_crowns=3,
        debug=False,
        home_template_threshold=0.80,
        home_hsv_max_distance=3.0,
        template_weight=1.0,
        hsv_weight=0.15
    )