import os
import cv2 as cv

from ground_truth_progress import GROUND_TRUTH
from image_processing import get_tiles

DATASET_FOLDER = "king_domino_dataset"
OUTPUT_FOLDER = "crown_tiles"
BOARD_SIZE = 500
GRID_SIZE = 5


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


def extract_crown_tiles():
    if not os.path.exists(DATASET_FOLDER):
        print(f"Dataset mappe blev ikke fundet: {DATASET_FOLDER}")
        return

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_FOLDER, "1_crown"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_FOLDER, "2_crowns"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_FOLDER, "3_crowns"), exist_ok=True)

    image_files = get_image_files(DATASET_FOLDER)

    if not image_files:
        print("Ingen billeder fundet i dataset mappen.")
        return

    saved_count = 0

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
        crown_grid = GROUND_TRUTH[board_name]["crowns"]

        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                crown_count = crown_grid[row][col]

                if crown_count > 0:
                    tile = tiles[row][col]

                    if crown_count == 1:
                        subfolder = "1_crown"
                    elif crown_count == 2:
                        subfolder = "2_crowns"
                    else:
                        subfolder = "3_crowns"

                    filename = (
                        f"board_{board_name}_r{row}_c{col}_crowns_{crown_count}.png"
                    )
                    output_path = os.path.join(OUTPUT_FOLDER, subfolder, filename)

                    cv.imwrite(output_path, tile)
                    saved_count += 1

    print(f"Færdig. Gemte {saved_count} crown tiles i '{OUTPUT_FOLDER}'.")


if __name__ == "__main__":
    extract_crown_tiles()