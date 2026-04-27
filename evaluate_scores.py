import os
import random

import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from ground_truth_progress import GROUND_TRUTH
from image_processing import get_tiles, predict_terrain_grid_from_image
from crown_knn_detection import build_crown_knn_database
from crown_hybrid_detection import build_crown_grid_hybrid
from kingdomino_pointmodel import calculate_board_score


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

                predicted = predicted_crowns[row][col]
                truth = true_crowns[row][col]

                edge_color = "lime" if predicted == truth else "red"

                rect = patches.Rectangle(
                    (x, y),
                    tile_size,
                    tile_size,
                    linewidth=2.5,
                    edgecolor=edge_color,
                    facecolor="none"
                )
                ax.add_patch(rect)

                ax.text(
                    x + 4,
                    y + 18,
                    f"P:{predicted} T:{truth}",
                    color="white",
                    fontsize=8,
                    weight="bold",
                    bbox=dict(facecolor="black", alpha=0.65, pad=2)
                )

    for ax in axes[len(results):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def evaluate_scores(
    train_ratio=0.8,
    seed=42,
    evaluate_on_test_only=True,
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
    print("Evaluering af Kingdomino pointmodel")
    print("=" * 70)
    print(f"Seed: {seed}")
    print(f"K: {k}")
    print(f"Train boards: {len(train_files)}")
    print(f"Test boards: {len(test_files)}")
    print(f"Zero crowns count as one: {zero_crowns_count_as_one}")

    print("\nBygger KNN crown database...")
    database_features, database_labels = build_crown_knn_database(train_files)

    if len(database_labels) == 0:
        print("KNN databasen er tom.")
        return

    if evaluate_on_test_only:
        files_to_evaluate = test_files
        print("Evaluerer kun på test split")
    else:
        files_to_evaluate = image_files
        print("Evaluerer på hele datasættet")

    total_absolute_error = 0
    total_boards = 0
    perfect_scores = 0

    results = []

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

        tiles = get_tiles(
            image,
            board_size=BOARD_SIZE,
            grid_size=GRID_SIZE
        )

        predicted_terrain_grid = predict_terrain_grid_from_image(
            image=image,
            board_size=BOARD_SIZE,
            grid_size=GRID_SIZE,
            debug=False,
            home_template_threshold=0.80,
            home_hsv_max_distance=3.0,
            template_weight=1.0,
            hsv_weight=0.15
        )

        predicted_crown_grid = build_crown_grid_hybrid(
            tiles=tiles,
            database_features=database_features,
            database_labels=database_labels,
            terrain_grid=predicted_terrain_grid,
            k=k,
            template_threshold=0.85,
            strong_score_threshold=0.90,
            iou_threshold=0.20,
            min_center_distance=10,
            max_crowns=3,
            debug=False
        )

        true_terrain_grid = GROUND_TRUTH[board_name]["terrain"]
        true_crown_grid = GROUND_TRUTH[board_name]["crowns"]

        predicted_score, predicted_breakdown = calculate_board_score(
            predicted_terrain_grid,
            predicted_crown_grid,
            zero_crowns_count_as_one=zero_crowns_count_as_one
        )

        true_score, true_breakdown = calculate_board_score(
            true_terrain_grid,
            true_crown_grid,
            zero_crowns_count_as_one=zero_crowns_count_as_one
        )

        score_error = abs(predicted_score - true_score)

        total_absolute_error += score_error
        total_boards += 1

        if score_error == 0:
            perfect_scores += 1

        results.append({
            "board_name": board_name,
            "image": image.copy(),
            "predicted_terrain": predicted_terrain_grid,
            "predicted_crowns": predicted_crown_grid,
            "true_terrain": true_terrain_grid,
            "true_crowns": true_crown_grid,
            "predicted_score": predicted_score,
            "true_score": true_score,
            "score_error": score_error
        })

        print("\n" + "=" * 70)
        print(f"Board {board_name}")
        print("=" * 70)
        print(f"True score:      {true_score}")
        print(f"Predicted score: {predicted_score}")
        print(f"Error:           {score_error}")

    if total_boards == 0:
        print("Ingen boards evalueret.")
        return

    mean_absolute_error = total_absolute_error / total_boards
    perfect_score_accuracy = perfect_scores / total_boards

    print("\n" + "=" * 70)
    print("Samlet score evaluering")
    print("=" * 70)
    print(f"Boards evalueret: {total_boards}")
    print(f"Perfekte scores: {perfect_scores}/{total_boards}")
    print(f"Score accuracy: {perfect_score_accuracy:.2%}")
    print(f"Mean absolute error: {mean_absolute_error:.2f}")

    if visualize:
        visualize_score_boards(
            results,
            board_size=BOARD_SIZE,
            grid_size=GRID_SIZE
        )


if __name__ == "__main__":
    evaluate_scores(
        train_ratio=0.8,
        seed=42,
        evaluate_on_test_only=True,
        k=1,
        visualize=True,

        # Rigtige Kingdomino regler:
        # Områder med 0 kroner giver 0 point.
        zero_crowns_count_as_one=False

        # Hvis I absolut vil have 0 kroner til at tælle som 1:
        # zero_crowns_count_as_one=True
    )