import cv2 as cv
import numpy as np
import os

from ground_truth_progress import GROUND_TRUTH


GRID_SIZE = 5
BOARD_SIZE = 500
TILE_SIZE = BOARD_SIZE // GRID_SIZE
DATASET_FOLDER = "king_domino_dataset"


def get_tiles(image, board_size=500, grid_size=5):
    """
    Resizer billedet til fast størrelse og splitter det i 5x5 tiles.
    Returnerer et 2D grid af tiles.
    """
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
    """
    Konverterer en tile til HSV og returnerer median H, S, V.
    Median er mere robust end mean overfor støj, kroner og små detaljer.
    """
    hsv_tile = cv.cvtColor(tile, cv.COLOR_BGR2HSV)
    h, s, v = np.median(hsv_tile, axis=(0, 1))
    return float(h), float(s), float(v)


def get_image_files(dataset_folder):
    """
    Finder og sorterer alle billedfiler i dataset-mappen.
    """
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")

    image_files = [
        f for f in os.listdir(dataset_folder)
        if f.lower().endswith(valid_extensions)
    ]

    image_files.sort(
        key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else x
    )
    return image_files


def calculate_hsv_stats():
    """
    Går gennem alle boards der findes både som billede og i ground truth.
    Samler median HSV pr tile for hver terrain-klasse.
    """
    if not os.path.exists(DATASET_FOLDER):
        print(f"Dataset mappe blev ikke fundet: {DATASET_FOLDER}")
        return

    class_hsv_values = {}

    image_files = get_image_files(DATASET_FOLDER)

    if not image_files:
        print("Ingen billeder fundet i dataset-mappen.")
        return

    used_boards = 0
    used_tiles = 0

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

                # Spring tomme labels over
                if terrain == "" or terrain is None:
                    continue

                tile = tiles[row][col]
                h, s, v = get_tile_median_hsv(tile)

                if terrain not in class_hsv_values:
                    class_hsv_values[terrain] = []

                class_hsv_values[terrain].append((h, s, v))
                used_tiles += 1

        used_boards += 1

    print(f"\nBoards brugt: {used_boards}")
    print(f"Tiles brugt: {used_tiles}")

    if not class_hsv_values:
        print("Ingen HSV data blev samlet.")
        return

    print("\n" + "=" * 70)
    print("HSV statistik pr terræntype")
    print("=" * 70)

    for terrain, hsv_list in sorted(class_hsv_values.items()):
        hsv_array = np.array(hsv_list)

        mean_hsv = np.mean(hsv_array, axis=0)
        median_hsv = np.median(hsv_array, axis=0)
        min_hsv = np.min(hsv_array, axis=0)
        max_hsv = np.max(hsv_array, axis=0)
        std_hsv = np.std(hsv_array, axis=0)

        print(f"\nTerræn: {terrain}")
        print(f"Antal tiles: {len(hsv_list)}")

        print(
            f"Mean HSV:   H={mean_hsv[0]:.2f}, S={mean_hsv[1]:.2f}, V={mean_hsv[2]:.2f}"
        )
        print(
            f"Median HSV: H={median_hsv[0]:.2f}, S={median_hsv[1]:.2f}, V={median_hsv[2]:.2f}"
        )
        print(
            f"Min HSV:    H={min_hsv[0]:.2f}, S={min_hsv[1]:.2f}, V={min_hsv[2]:.2f}"
        )
        print(
            f"Max HSV:    H={max_hsv[0]:.2f}, S={max_hsv[1]:.2f}, V={max_hsv[2]:.2f}"
        )
        print(
            f"Std HSV:    H={std_hsv[0]:.2f}, S={std_hsv[1]:.2f}, V={std_hsv[2]:.2f}"
        )

    print("\nFærdig.")


if __name__ == "__main__":
    calculate_hsv_stats()