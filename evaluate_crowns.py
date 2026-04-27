import os
import random
from collections import defaultdict

import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.patches as patches

try:
    import pandas as pd
except (ModuleNotFoundError, ImportError):
    pd = None

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
        key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else x
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


def visualize_test_boards_with_crowns(test_results, board_size=500, grid_size=5):
    """
    Viser boards med predicted crowns ovenpå.
    Grøn kant = korrekt
    Rød kant = forkert
    """
    if not test_results:
        print("Ingen test results at visualisere.")
        return

    num_boards = len(test_results)
    cols = 4
    rows = (num_boards + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))

    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1:
        axes = list(axes)
    else:
        axes = axes.flatten()

    tile_size = board_size // grid_size

    for ax, result in zip(axes, test_results):
        board_name = result["board_name"]
        image = result["image"]
        predicted_grid = result["predicted_grid"]
        true_grid = result["true_grid"]

        image_resized = cv.resize(image, (board_size, board_size))
        image_rgb = cv.cvtColor(image_resized, cv.COLOR_BGR2RGB)

        ax.imshow(image_rgb)
        ax.set_title(f"Board {board_name}", fontsize=12)
        ax.axis("off")

        for row in range(grid_size):
            for col in range(grid_size):
                x = col * tile_size
                y = row * tile_size

                predicted = predicted_grid[row][col]
                truth = true_grid[row][col]

                is_correct = predicted == truth
                edge_color = "lime" if is_correct else "red"

                rect = patches.Rectangle(
                    (x, y),
                    tile_size,
                    tile_size,
                    linewidth=2.5,
                    edgecolor=edge_color,
                    facecolor="none"
                )
                ax.add_patch(rect)

                text = f"P:{predicted} T:{truth}"

                ax.text(
                    x + 4,
                    y + 18,
                    text,
                    color="white",
                    fontsize=8,
                    weight="bold",
                    bbox=dict(facecolor="black", alpha=0.65, pad=2)
                )

    for ax in axes[len(test_results):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def evaluate_crowns(
    train_ratio=0.8,
    seed=42,
    evaluate_on_test_only=True,
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
    if not os.path.exists(DATASET_FOLDER):
        print(f"Dataset mappe blev ikke fundet: {DATASET_FOLDER}")
        return

    image_files = get_image_files(DATASET_FOLDER)

    if not image_files:
        print("Ingen billeder fundet i dataset mappen.")
        return

    train_files, test_files = split_dataset(image_files, train_ratio=train_ratio, seed=seed)

    print("=" * 70)
    print("Evaluering af crown model")
    print("=" * 70)
    print(f"Seed: {seed}")
    print(f"Train boards: {len(train_files)}")
    print(f"Test boards: {len(test_files)}")
    print(f"Threshold: {threshold}")
    print(f"IOU threshold: {iou_threshold}")
    print(f"Min center distance: {min_center_distance}")
    print(f"Strong score threshold: {strong_score_threshold}")
    print(f"Max crowns per tile: {max_crowns}")

    if evaluate_on_test_only:
        files_to_evaluate = test_files
        print("Evaluerer kun på test split")
    else:
        files_to_evaluate = image_files
        print("Evaluerer på hele datasættet")

    total_correct = 0
    total_tiles = 0
    evaluated_boards = 0
    test_results = []

    confusion = defaultdict(lambda: defaultdict(int))

    for file_name in files_to_evaluate:
        board_name = os.path.splitext(file_name)[0]

        if board_name not in GROUND_TRUTH:
            print(f"Springer board {board_name} over, fordi det ikke findes i ground truth.")
            continue

        image_path = os.path.join(DATASET_FOLDER, file_name)
        image = cv.imread(image_path)

        if image is None:
            print(f"Kunne ikke læse billede: {image_path}")
            continue

        tiles = get_tiles(image, board_size=BOARD_SIZE, grid_size=GRID_SIZE)

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

        test_results.append({
            "board_name": board_name,
            "image": image.copy(),
            "predicted_grid": predicted_grid,
            "true_grid": true_grid,
            "terrain_grid": terrain_grid
        })

        correct, total, accuracy, mistakes = compare_crown_grids(predicted_grid, true_grid, confusion)

        total_correct += correct
        total_tiles += total
        evaluated_boards += 1

        print(f"\nBoard {board_name}")
        print(f"Accuracy: {correct}/{total} = {accuracy:.2%}")

        if mistakes:
            print("Fejl:")
            for row, col, truth, predicted in mistakes:
                print(f"  Felt ({row},{col}): true={truth}, predicted={predicted}")
        else:
            print("Ingen fejl. Perfekt board.")

    if total_tiles == 0:
        print("\nIngen boards blev evalueret.")
        return

    overall_accuracy = total_correct / total_tiles

    print("\n" + "=" * 70)
    print("Samlet resultat")
    print("=" * 70)
    print(f"Boards evalueret: {evaluated_boards}")
    print(f"Samlet korrekt: {total_correct}/{total_tiles}")
    print(f"Samlet accuracy: {overall_accuracy:.2%}")

    print("\n" + "=" * 70)
    print("Confusion Matrix")
    print("=" * 70)

    labels = sorted(set(confusion.keys()) | {pred for row in confusion.values() for pred in row.keys()})

    matrix = []
    for true_label in labels:
        row = []
        for pred_label in labels:
            row.append(confusion[true_label].get(pred_label, 0))
        matrix.append(row)

    if pd is not None:
        df = pd.DataFrame(matrix, index=labels, columns=labels)
        print(df)
    else:
        print("pandas kunne ikke importeres i dette miljø. Viser matrix som tekst:\n")
        header = "true\\pred".ljust(12) + " ".join(str(label).ljust(8) for label in labels)
        print(header)
        for true_label, row in zip(labels, matrix):
            row_text = " ".join(str(value).ljust(8) for value in row)
            print(str(true_label).ljust(12) + row_text)

    visualize_test_boards_with_crowns(
        test_results,
        board_size=BOARD_SIZE,
        grid_size=GRID_SIZE
    )


if __name__ == "__main__":
    evaluate_crowns(
        train_ratio=0.8,
        seed=42,
        evaluate_on_test_only=True,
        threshold=0.85,
        iou_threshold=0.20,
        min_center_distance=10,
        strong_score_threshold=0.90,
        max_crowns=3,
        debug=False,
        home_template_threshold=0.80,
        home_hsv_max_distance=3.0,
        template_weight=1.0,
        hsv_weight=0.15
    )