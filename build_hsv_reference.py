import cv2 as cv
import numpy as np
import os
from pprint import pformat

from ground_truth_progress import GROUND_TRUTH

GRID_SIZE = 5
BOARD_SIZE = 500
TILE_SIZE = BOARD_SIZE // GRID_SIZE
DATASET_FOLDER = "king_domino_dataset"
OUTPUT_FILE = "hsv_reference.py"


def get_tiles(image, board_size=500, grid_size=5):
    image = cv.resize(image, (board_size, board_size))
    tile_size = board_size // grid_size

    tiles = []
    for y in range(grid_size):
        row = []
        for x in range(grid_size):
            tile = image[
                y * tile_size:(y + 1) * tile_size,
                x * tile_size:(x + 1) * tile_size
            ]
            row.append(tile)
        tiles.append(row)

    return tiles


def get_tile_median_hsv(tile):
    hsv_tile = cv.cvtColor(tile, cv.COLOR_BGR2HSV)
    h, s, v = np.median(hsv_tile, axis=(0, 1))
    return float(h), float(s), float(v)


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


def build_hsv_reference():
    class_hsv_values = {}

    image_files = get_image_files(DATASET_FOLDER)

    for file_name in image_files:
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
        terrain_grid = GROUND_TRUTH[board_name]["terrain"]

        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                terrain = terrain_grid[row][col]

                if terrain == "" or terrain is None:
                    continue

                tile = tiles[row][col]
                h, s, v = get_tile_median_hsv(tile)

                if terrain not in class_hsv_values:
                    class_hsv_values[terrain] = []

                class_hsv_values[terrain].append((h, s, v))

    hsv_reference = {}

    for terrain, hsv_list in sorted(class_hsv_values.items()):
        hsv_array = np.array(hsv_list)

        hsv_reference[terrain] = {
            "mean": np.mean(hsv_array, axis=0).round(2).tolist(),
            "median": np.median(hsv_array, axis=0).round(2).tolist(),
            "std": np.std(hsv_array, axis=0).round(2).tolist(),
            "min": np.min(hsv_array, axis=0).round(2).tolist(),
            "max": np.max(hsv_array, axis=0).round(2).tolist(),
            "count": int(len(hsv_list))
        }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("HSV_REFERENCE = ")
        f.write(pformat(hsv_reference, width=100, sort_dicts=False))
        f.write("\n")

    print(f"HSV reference gemt i {OUTPUT_FILE}")


if __name__ == "__main__":
    build_hsv_reference()