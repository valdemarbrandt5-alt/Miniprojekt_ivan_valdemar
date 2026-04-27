import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from kingdomino_pointmodel import find_clusters


def get_tiles(image, board_size=500, grid_size=5):
    """
    Resizer billedet til board_size x board_size
    og splitter det i et 5x5 grid af tiles.
    Returnerer et 2D grid med tiles.
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


def get_terrain(tile, debug=False):
    """
    Bestemmer terrain type ud fra median HSV værdier.
    """
    hsv_tile = cv.cvtColor(tile, cv.COLOR_BGR2HSV)
    hue, saturation, value = np.median(hsv_tile, axis=(0, 1))

    if debug:
        print(f"H: {hue:.1f}, S: {saturation:.1f}, V: {value:.1f}")

    if 21 < hue < 28 and 218 < saturation < 256 and 136 < value < 208:
        return "Field"

    if 28 < hue < 80 and 67 < saturation < 226 and 24 < value < 74:
        return "Forest"

    if 103 < hue < 110 and 223 < saturation < 256 and 115 < value < 199:
        return "Lake"

    if 33 < hue < 48 and 159 < saturation < 249 and 75 < value < 165:
        return "Grassland"

    if 17 < hue < 27 and 34 < saturation < 181 and 72 < value < 145:
        return "Swamp"

    if 17 < hue < 26 and 39 < saturation < 156 and 23 < value < 72:
        return "Mine"

    if 16 < hue < 87 and 40 < saturation < 141 and 52 < value < 145:
        return "Home"

    return "Unknown"


def build_terrain_grid(tiles, debug=False):
    """
    Bygger et 5x5 terrain grid ud fra et 5x5 tile grid.
    """
    terrain_grid = []

    for row in tiles:
        terrain_row = []
        for tile in row:
            terrain_type = get_terrain(tile, debug=debug)
            terrain_row.append(terrain_type)
        terrain_grid.append(terrain_row)

    return terrain_grid


def visualize_board(terrain_grid, clusters, board_name="Board"):
    """
    Visualiserer terrain grid og markerer clusters.
    """
    terrain_colors = {
        "Field": "#ffe066",
        "Forest": "#228B22",
        "Lake": "#3399ff",
        "Grassland": "#98fb98",
        "Swamp": "#8fbc8f",
        "Mine": "#b0b0b0",
        "Home": "#ffb347",
        "Unknown": "#cccccc"
    }

    fig, ax = plt.subplots(figsize=(6, 6))

    for y, row in enumerate(terrain_grid):
        for x, terrain in enumerate(row):
            color = terrain_colors.get(terrain, "#cccccc")
            rect = plt.Rectangle((x, 4 - y), 1, 1, facecolor=color, edgecolor="black")
            ax.add_patch(rect)
            ax.text(
                x + 0.5,
                4 - y + 0.5,
                terrain[0],
                ha="center",
                va="center",
                fontsize=8,
                color="black"
            )

    for terrain, cluster in clusters:
        for (x, y) in cluster:
            ax.plot(x + 0.5, 4 - y + 0.5, "o", color="red", markersize=8, alpha=0.5)

    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.set_xticks(range(6))
    ax.set_yticks(range(6))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(f"Kingdomino board visualisering: {board_name}")
    ax.set_aspect("equal")

    legend_patches = [
        mpatches.Patch(color=color, label=terrain)
        for terrain, color in terrain_colors.items()
    ]
    plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


def get_image_files(dataset_folder):
    """
    Finder alle billedfiler i dataset mappen.
    """
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")

    image_files = [
        f for f in os.listdir(dataset_folder)
        if f.lower().endswith(valid_extensions)
    ]

    image_files.sort()
    return image_files


def process_board(image_path, debug=False):
    """
    Behandler ét board:
    læser billede, splitter i tiles, bygger terrain grid og finder clusters.
    """
    image = cv.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Kunne ikke læse billedet: {image_path}")

    tiles = get_tiles(image, board_size=500, grid_size=5)
    terrain_grid = build_terrain_grid(tiles, debug=debug)
    clusters = find_clusters(terrain_grid)

    return terrain_grid, clusters


def main():
    dataset_folder = "king_domino_dataset"

    if not os.path.exists(dataset_folder):
        print(f"Dataset mappen blev ikke fundet: {dataset_folder}")
        return

    image_files = get_image_files(dataset_folder)

    if not image_files:
        print("Ingen billeder fundet i dataset mappen.")
        return

    print(f"Antal boards fundet: {len(image_files)}")

    first_board_visualized = False

    for file_name in image_files:
        image_path = os.path.join(dataset_folder, file_name)
        board_name = os.path.splitext(file_name)[0]

        print("\n" + "=" * 50)
        print(f"Behandler board: {file_name}")

        terrain_grid, clusters = process_board(image_path, debug=False)

        print("\nTerrain grid:")
        for row in terrain_grid:
            print(row)

        print("\nSammenhængende områder:")
        for terrain, cluster in clusters:
            print(f"{terrain}: {len(cluster)} felter -> {cluster}")

        if not first_board_visualized:
            visualize_board(terrain_grid, clusters, board_name=board_name)
            first_board_visualized = True

    print("\nFærdig med at behandle alle boards.")


if __name__ == "__main__":
    main()