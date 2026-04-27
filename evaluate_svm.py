import cv2 as cv
import numpy as np
import os

try:
    import pandas as pd
except (ModuleNotFoundError, ImportError):
    pd = None

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from ground_truth_progress import GROUND_TRUTH

from image_processing import (
    get_best_home_match_score,
    get_home_hsv_distance
)


GRID_SIZE = 5
BOARD_SIZE = 500
DATASET_FOLDER = "king_domino_dataset"


def get_tiles(image, board_size=500, grid_size=5):
    image = cv.resize(image, (board_size, board_size))
    tile_size = board_size // grid_size

    tiles = []

    for row in range(grid_size):
        tile_row = []

        for col in range(grid_size):
            tile = image[
                row * tile_size:(row + 1) * tile_size,
                col * tile_size:(col + 1) * tile_size
            ]

            tile_row.append(tile)

        tiles.append(tile_row)

    return tiles


def get_tile_features(tile):
    """
    Feature vector til SVM:
    1. median H
    2. median S
    3. median V
    4. Home template score
    5. Home HSV distance
    """
    hsv_tile = cv.cvtColor(tile, cv.COLOR_BGR2HSV)
    h, s, v = np.median(hsv_tile, axis=(0, 1))

    home_template_score = get_best_home_match_score(tile)
    home_hsv_distance = get_home_hsv_distance(tile)

    return [
        float(h),
        float(s),
        float(v),
        float(home_template_score),
        float(home_hsv_distance)
    ]


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

    rng = np.random.default_rng(seed)
    rng.shuffle(shuffled)

    split_index = int(len(shuffled) * train_ratio)

    train_files = shuffled[:split_index]
    test_files = shuffled[split_index:]

    return train_files, test_files


def build_dataset(file_list):
    X = []
    y = []

    for file_name in file_list:
        board_name = os.path.splitext(file_name)[0]

        if board_name not in GROUND_TRUTH:
            continue

        image_path = os.path.join(DATASET_FOLDER, file_name)
        image = cv.imread(image_path)

        if image is None:
            print(f"Kunne ikke læse billede: {image_path}")
            continue

        tiles = get_tiles(image, board_size=BOARD_SIZE, grid_size=GRID_SIZE)
        terrain_grid = GROUND_TRUTH[board_name]["terrain"]

        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                label = terrain_grid[row][col]

                if label == "" or label is None:
                    continue

                tile = tiles[row][col]
                features = get_tile_features(tile)

                X.append(features)
                y.append(label)

    return np.array(X), np.array(y)


def print_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    print("\nConfusion Matrix:")
    print("=" * 70)

    if pd is not None:
        df = pd.DataFrame(cm, index=labels, columns=labels)
        print(df)
    else:
        print("pandas kunne ikke importeres. Viser matrix som tekst:\n")

        header = "true\\pred".ljust(12) + " ".join(label.ljust(10) for label in labels)
        print(header)

        for true_label, row in zip(labels, cm):
            row_text = " ".join(str(value).ljust(10) for value in row)
            print(true_label.ljust(12) + row_text)


def evaluate_svm(train_ratio=0.8, seed=42):
    if not os.path.exists(DATASET_FOLDER):
        print(f"Dataset mappe blev ikke fundet: {DATASET_FOLDER}")
        return

    image_files = get_image_files(DATASET_FOLDER)

    if not image_files:
        print("Ingen billeder fundet i dataset mappen.")
        return

    train_files, test_files = split_dataset(
        image_files=image_files,
        train_ratio=train_ratio,
        seed=seed
    )

    print("=" * 70)
    print("SVM evaluering med board based split")
    print("=" * 70)
    print(f"Seed: {seed}")
    print(f"Train boards: {len(train_files)}")
    print(f"Test boards: {len(test_files)}")

    print("\nBygger train dataset...")
    X_train, y_train = build_dataset(train_files)

    print("Bygger test dataset...")
    X_test, y_test = build_dataset(test_files)

    if len(X_train) == 0:
        print("Ingen train data fundet.")
        return

    if len(X_test) == 0:
        print("Ingen test data fundet.")
        return

    print(f"\nTrain samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Antal features pr tile: {X_train.shape[1]}")

    model = make_pipeline(
        StandardScaler(),
        SVC(
            kernel="rbf",
            C=10,
            gamma="scale"
        )
    )

    print("\nTræner SVM...")
    model.fit(X_train, y_train)

    print("Tester SVM...")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 70)
    print("SVM Resultat med board based split")
    print("=" * 70)
    print(f"Accuracy: {accuracy:.2%}")

    labels = sorted(np.unique(np.concatenate([y_train, y_test])))
    print_confusion_matrix(y_test, y_pred, labels)


if __name__ == "__main__":
    evaluate_svm(
        train_ratio=0.8,
        seed=42
    )