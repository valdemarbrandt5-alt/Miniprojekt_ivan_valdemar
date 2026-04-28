import os
import random
from collections import defaultdict

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

try:
    import pandas as pd
except (ModuleNotFoundError, ImportError):
    pd = None

try:
    from sklearn.metrics import f1_score, classification_report
except (ModuleNotFoundError, ImportError):
    f1_score = None
    classification_report = None

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from ground_truth_progress import GROUND_TRUTH
from image_processing import get_tiles
from evaluate_svm import build_dataset, get_tile_features
from crown_knn_detection import build_crown_knn_database, build_crown_grid_knn
from kingdomino_pointmodel import calculate_board_score


GRID_SIZE = 5
BOARD_SIZE = 500
DATASET_FOLDER = "king_domino_dataset"


def get_image_files(dataset_folder):
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")

    image_files = [
        file_name for file_name in os.listdir(dataset_folder)
        if file_name.lower().endswith(valid_extensions)
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


def train_terrain_svm(train_files):
    print("\nBygger SVM terrain train dataset...")
    X_train, y_train = build_dataset(train_files)

    if len(X_train) == 0:
        raise ValueError("Ingen SVM train data fundet.")

    print(f"SVM train samples: {len(X_train)}")
    print(f"Features pr tile: {X_train.shape[1]}")

    terrain_model = make_pipeline(
        StandardScaler(),
        SVC(
            kernel="rbf",
            C=10,
            gamma="scale"
        )
    )

    print("Træner SVM terrain model...")
    terrain_model.fit(X_train, y_train)

    return terrain_model


def predict_terrain_grid_svm(tiles, terrain_model):
    terrain_grid = []

    for row in range(GRID_SIZE):
        terrain_row = []

        for col in range(GRID_SIZE):
            tile = tiles[row][col]
            features = np.array(get_tile_features(tile)).reshape(1, -1)
            prediction = terrain_model.predict(features)[0]
            terrain_row.append(prediction)

        terrain_grid.append(terrain_row)

    return terrain_grid


def compare_crown_grids(predicted_grid, true_grid, confusion):
    correct = 0
    total = 0

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            predicted = predicted_grid[row][col]
            truth = true_grid[row][col]

            confusion[truth][predicted] += 1
            total += 1

            if predicted == truth:
                correct += 1

    return correct, total


def compare_terrain_grids(predicted_grid, true_grid, confusion):
    correct = 0
    total = 0

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            predicted = predicted_grid[row][col]
            truth = true_grid[row][col]

            confusion[truth][predicted] += 1
            total += 1

            if predicted == truth:
                correct += 1

    return correct, total


def print_confusion_matrix(confusion, labels, title):
    matrix = []

    for true_label in labels:
        row = []
        for pred_label in labels:
            row.append(confusion[true_label].get(pred_label, 0))
        matrix.append(row)

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    if pd is not None:
        df = pd.DataFrame(matrix, index=labels, columns=labels)
        print(df)
    else:
        header = "true\\pred".ljust(12) + " ".join(str(label).ljust(10) for label in labels)
        print(header)

        for true_label, row in zip(labels, matrix):
            row_text = " ".join(str(value).ljust(10) for value in row)
            print(str(true_label).ljust(12) + row_text)


def visualize_score_boards(results, board_size=500, grid_size=5):
    if not results:
        print("Ingen results at visualisere.")
        return

    num_boards = len(results)
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

    for ax, result in zip(axes, results):
        board_name = result["board_name"]
        image = result["image"]

        predicted_crowns = result["predicted_crowns"]
        true_crowns = result["true_crowns"]

        predicted_terrain = result["predicted_terrain"]
        true_terrain = result["true_terrain"]

        predicted_score = result["predicted_score"]
        true_score = result["true_score"]
        score_error = result["score_error"]

        image_resized = cv.resize(image, (board_size, board_size))
        image_rgb = cv.cvtColor(image_resized, cv.COLOR_BGR2RGB)

        ax.imshow(image_rgb)
        ax.set_title(
            f"Board {board_name}\n"
            f"Pred: {predicted_score} | True: {true_score} | Error: {score_error}",
            fontsize=11
        )
        ax.axis("off")

        for row in range(grid_size):
            for col in range(grid_size):
                x = col * tile_size
                y = row * tile_size

                crown_pred = predicted_crowns[row][col]
                crown_true = true_crowns[row][col]

                terrain_pred = predicted_terrain[row][col]
                terrain_true = true_terrain[row][col]

                is_correct = (
                    crown_pred == crown_true
                    and terrain_pred == terrain_true
                )

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

                text = (
                    f"C P:{crown_pred} T:{crown_true}\n"
                    f"{terrain_pred}"
                )

                ax.text(
                    x + 4,
                    y + 18,
                    text,
                    color="white",
                    fontsize=7,
                    weight="bold",
                    bbox=dict(facecolor="black", alpha=0.65, pad=2)
                )

    for ax in axes[len(results):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def evaluate_scores_svm_terrain_knn_crowns(
    train_ratio=0.8,
    seed=42,
    k=1,
    visualize=True,
    zero_crowns_count_as_one=False
):
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

    print("=" * 70)
    print("Evaluering af pointmodel med SVM terrain + KNN crowns")
    print("=" * 70)
    print(f"Seed: {seed}")
    print(f"K: {k}")
    print(f"Train boards: {len(train_files)}")
    print(f"Test boards: {len(test_files)}")
    print(f"Zero crowns count as one: {zero_crowns_count_as_one}")

    terrain_model = train_terrain_svm(train_files)

    print("\nBygger KNN crown database fra train split...")
    database_features, database_labels = build_crown_knn_database(train_files)

    if len(database_labels) == 0:
        print("KNN databasen er tom.")
        return

    total_boards = 0

    total_score_error = 0
    perfect_scores = 0

    total_crown_correct = 0
    total_crown_tiles = 0

    total_terrain_correct = 0
    total_terrain_tiles = 0

    all_true_crowns = []
    all_predicted_crowns = []

    crown_confusion = defaultdict(lambda: defaultdict(int))
    terrain_confusion = defaultdict(lambda: defaultdict(int))

    results = []

    for file_name in test_files:
        board_name = os.path.splitext(file_name)[0]

        if board_name not in GROUND_TRUTH:
            print(f"Springer board {board_name} over, fordi det ikke findes i ground truth.")
            continue

        image_path = os.path.join(DATASET_FOLDER, file_name)
        image = cv.imread(image_path)

        if image is None:
            print(f"Kunne ikke læse billede: {image_path}")
            continue

        tiles = get_tiles(
            image,
            board_size=BOARD_SIZE,
            grid_size=GRID_SIZE
        )

        predicted_terrain_grid = predict_terrain_grid_svm(
            tiles,
            terrain_model
        )

        predicted_crown_grid = build_crown_grid_knn(
            tiles,
            database_features,
            database_labels,
            k=k
        )

        true_terrain_grid = GROUND_TRUTH[board_name]["terrain"]
        true_crown_grid = GROUND_TRUTH[board_name]["crowns"]

        predicted_score, _ = calculate_board_score(
            predicted_terrain_grid,
            predicted_crown_grid,
            zero_crowns_count_as_one=zero_crowns_count_as_one
        )

        true_score, _ = calculate_board_score(
            true_terrain_grid,
            true_crown_grid,
            zero_crowns_count_as_one=zero_crowns_count_as_one
        )

        score_error = abs(predicted_score - true_score)

        terrain_correct, terrain_total = compare_terrain_grids(
            predicted_terrain_grid,
            true_terrain_grid,
            terrain_confusion
        )

        crown_correct, crown_total = compare_crown_grids(
            predicted_crown_grid,
            true_crown_grid,
            crown_confusion
        )

        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                all_true_crowns.append(true_crown_grid[row][col])
                all_predicted_crowns.append(predicted_crown_grid[row][col])

        total_terrain_correct += terrain_correct
        total_terrain_tiles += terrain_total

        total_crown_correct += crown_correct
        total_crown_tiles += crown_total

        total_score_error += score_error
        total_boards += 1

        if score_error == 0:
            perfect_scores += 1

        results.append({
            "board_name": board_name,
            "image": image.copy(),
            "predicted_terrain": predicted_terrain_grid,
            "true_terrain": true_terrain_grid,
            "predicted_crowns": predicted_crown_grid,
            "true_crowns": true_crown_grid,
            "predicted_score": predicted_score,
            "true_score": true_score,
            "score_error": score_error
        })

        print("\n" + "=" * 70)
        print(f"Board {board_name}")
        print("=" * 70)
        print(f"Terrain accuracy: {terrain_correct}/{terrain_total} = {terrain_correct / terrain_total:.2%}")
        print(f"Crown accuracy:   {crown_correct}/{crown_total} = {crown_correct / crown_total:.2%}")
        print(f"True score:       {true_score}")
        print(f"Predicted score:  {predicted_score}")
        print(f"Score error:      {score_error}")

    if total_boards == 0:
        print("Ingen boards evalueret.")
        return

    terrain_accuracy = total_terrain_correct / total_terrain_tiles
    crown_accuracy = total_crown_correct / total_crown_tiles

    exact_score_accuracy = perfect_scores / total_boards
    mean_absolute_score_error = total_score_error / total_boards

    print("\n" + "=" * 70)
    print("Samlet evaluering")
    print("=" * 70)
    print(f"Boards evalueret: {total_boards}")
    print(f"Terrain accuracy: {terrain_accuracy:.2%}")
    print(f"Crown accuracy:   {crown_accuracy:.2%}")
    print(f"Perfekte scores:  {perfect_scores}/{total_boards}")
    print(f"Exact score accuracy: {exact_score_accuracy:.2%}")
    print(f"Mean absolute score error: {mean_absolute_score_error:.2f}")

    if f1_score is not None:
        micro_f1 = f1_score(
            all_true_crowns,
            all_predicted_crowns,
            labels=[0, 1, 2, 3],
            average="micro",
            zero_division=0
        )

        macro_f1 = f1_score(
            all_true_crowns,
            all_predicted_crowns,
            labels=[0, 1, 2, 3],
            average="macro",
            zero_division=0
        )

        weighted_f1 = f1_score(
            all_true_crowns,
            all_predicted_crowns,
            labels=[0, 1, 2, 3],
            average="weighted",
            zero_division=0
        )

        print("\n" + "=" * 70)
        print("Crown F1 scores")
        print("=" * 70)
        print(f"Micro F1:    {micro_f1:.4f}")
        print(f"Macro F1:    {macro_f1:.4f}")
        print(f"Weighted F1: {weighted_f1:.4f}")

        print("\nClassification report:")
        print(
            classification_report(
                all_true_crowns,
                all_predicted_crowns,
                labels=[0, 1, 2, 3],
                zero_division=0
            )
        )
    else:
        print("\nsklearn metrics kunne ikke importeres.")
        print("Installer med: pip install scikit-learn")

    terrain_labels = sorted(
        set(terrain_confusion.keys()) |
        {pred for row in terrain_confusion.values() for pred in row.keys()}
    )

    crown_labels = [0, 1, 2, 3]

    print_confusion_matrix(
        terrain_confusion,
        terrain_labels,
        "Terrain confusion matrix"
    )

    print_confusion_matrix(
        crown_confusion,
        crown_labels,
        "Crown confusion matrix"
    )

    if visualize:
        visualize_score_boards(
            results,
            board_size=BOARD_SIZE,
            grid_size=GRID_SIZE
        )


if __name__ == "__main__":
    evaluate_scores_svm_terrain_knn_crowns(
        train_ratio=0.8,
        seed=67,
        k=1,
        visualize=True,
        zero_crowns_count_as_one=False
    )