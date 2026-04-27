import os
import cv2 as cv
import numpy as np

from ground_truth_progress import GROUND_TRUTH
from image_processing import get_tiles

DATASET_FOLDER = "king_domino_dataset"
BOARD_SIZE = 500
GRID_SIZE = 5


def preprocess_tile(tile, size=32):
    gray = cv.cvtColor(tile, cv.COLOR_BGR2GRAY)
    gray = cv.resize(gray, (size, size))
    gray = cv.GaussianBlur(gray, (3, 3), 0)

    vector = gray.astype(np.float32).flatten()

    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm

    return vector


def build_crown_knn_database(train_files):
    features = []
    labels = []

    for file_name in train_files:
        board_name = os.path.splitext(file_name)[0]

        if board_name not in GROUND_TRUTH:
            continue

        image_path = os.path.join(DATASET_FOLDER, file_name)
        image = cv.imread(image_path)

        if image is None:
            continue

        tiles = get_tiles(image, board_size=BOARD_SIZE, grid_size=GRID_SIZE)
        crown_grid = GROUND_TRUTH[board_name]["crowns"]

        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                tile = tiles[row][col]
                crown_count = crown_grid[row][col]

                features.append(preprocess_tile(tile))
                labels.append(crown_count)

    return np.array(features), np.array(labels)


def predict_crown_count_knn(tile, database_features, database_labels, k=1):
    tile_feature = preprocess_tile(tile)

    # cosine similarity, fordi features er normaliserede
    similarities = database_features @ tile_feature

    nearest_indices = np.argsort(similarities)[-k:]
    nearest_labels = database_labels[nearest_indices]

    if k == 1:
        return int(nearest_labels[0])

    counts = np.bincount(nearest_labels, minlength=4)
    return int(np.argmax(counts))


def build_crown_grid_knn(tiles, database_features, database_labels, k=1):
    crown_grid = []

    for row in range(GRID_SIZE):
        crown_row = []

        for col in range(GRID_SIZE):
            crown_count = predict_crown_count_knn(
                tiles[row][col],
                database_features,
                database_labels,
                k=k
            )
            crown_row.append(crown_count)

        crown_grid.append(crown_row)

    return crown_grid