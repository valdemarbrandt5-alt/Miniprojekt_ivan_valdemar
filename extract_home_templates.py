import cv2 as cv
import os

from ground_truth_progress import GROUND_TRUTH

GRID_SIZE = 5
BOARD_SIZE = 500
DATASET_FOLDER = "king_domino_dataset"
OUTPUT_FOLDER = "home_templates"


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


def extract_home_templates():
    if not os.path.exists(DATASET_FOLDER):
        print(f"Dataset mappe blev ikke fundet: {DATASET_FOLDER}")
        return

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    image_files = get_image_files(DATASET_FOLDER)
    saved_count = 0

    for file_name in image_files:
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
                if terrain_grid[row][col] == "Home":
                    tile = tiles[row][col]
                    output_name = f"home_{board_name}_r{row}_c{col}.png"
                    output_path = os.path.join(OUTPUT_FOLDER, output_name)
                    cv.imwrite(output_path, tile)
                    saved_count += 1

    print(f"Gemte {saved_count} Home templates i '{OUTPUT_FOLDER}'")


if __name__ == "__main__":
    extract_home_templates()