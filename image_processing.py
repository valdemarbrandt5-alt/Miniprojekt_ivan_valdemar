import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from kingdomino_pointmodel import find_clusters
from hsv_reference import HSV_REFERENCE

HOME_TEMPLATE_FOLDER = "home_templates"


def get_tiles(image, board_size=500, grid_size=5):
    """
    Resizer billedet til board_size x board_size
    og splitter det i et grid af tiles.
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


def save_tiles_if_needed(tiles, output_folder):
    """
    Gemmer kun tiles hvis output mappen ikke allerede findes.
    Hvis mappen findes, springes gemningen over.
    """
    if os.path.exists(output_folder):
        print(f"Tiles findes allerede i '{output_folder}' -> springer gemning over")
        return

    os.makedirs(output_folder, exist_ok=True)

    for y, row in enumerate(tiles):
        for x, tile in enumerate(row):
            filename = os.path.join(output_folder, f"tile_r{y}_c{x}.png")
            cv.imwrite(filename, tile)

    print(f"Tiles gemt i '{output_folder}'")


def load_home_templates():
    templates = []

    if not os.path.exists(HOME_TEMPLATE_FOLDER):
        print(f"Home template mappe blev ikke fundet: {HOME_TEMPLATE_FOLDER}")
        return templates

    for file_name in os.listdir(HOME_TEMPLATE_FOLDER):
        if file_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            path = os.path.join(HOME_TEMPLATE_FOLDER, file_name)
            img = cv.imread(path, cv.IMREAD_GRAYSCALE)

            if img is not None:
                img = cv.resize(img, (100, 100))
                templates.append(img)

    print(f"Loaded {len(templates)} Home templates")
    return templates


HOME_TEMPLATES = load_home_templates()


def is_home(tile, threshold=0.80, debug=False, home_templates=None):
    """
    Klassisk template matching for Home.
    Returnerer (detected, best_score)
    """
    templates = home_templates if home_templates is not None else HOME_TEMPLATES

    if not templates:
        return False, 0.0

    tile_gray = cv.cvtColor(tile, cv.COLOR_BGR2GRAY)
    tile_gray = cv.resize(tile_gray, (100, 100))

    best_score = -1.0

    for template in templates:
        result = cv.matchTemplate(tile_gray, template, cv.TM_CCOEFF_NORMED)
        score = result[0][0]

        if score > best_score:
            best_score = score

    if debug:
        print(f"Home best match score: {best_score:.3f}")

    return best_score >= threshold, best_score


def get_home_score(tile, debug=False, home_templates=None):
    """
    Returnerer kun Home template score.
    """
    _, score = is_home(tile, threshold=0.0, debug=debug, home_templates=home_templates)
    return score


def get_home_hsv_distance(tile, debug=False):
    """
    Returnerer normaliseret HSV distance til Home-klassen.
    Lav distance = ligner Home farvemæssigt.
    """
    if "Home" not in HSV_REFERENCE:
        return float("inf")

    hsv_tile = cv.cvtColor(tile, cv.COLOR_BGR2HSV)
    hue, saturation, value = np.median(hsv_tile, axis=(0, 1))

    tile_vector = np.array([hue, saturation, value], dtype=float)

    class_mean = np.array(HSV_REFERENCE["Home"]["mean"], dtype=float)
    class_std = np.array(HSV_REFERENCE["Home"]["std"], dtype=float)
    class_std[class_std == 0] = 1.0

    distance = np.linalg.norm((tile_vector - class_mean) / class_std)

    if debug:
        print(
            f"Home HSV distance: {distance:.3f} "
            f"(H={hue:.2f}, S={saturation:.2f}, V={value:.2f})"
        )

    return distance


def get_terrain_without_home(tile, debug=False):
    """
    Klassificerer terrain kun via HSV.
    Home springes over her, fordi Home vælges på board niveau senere.
    """
    hsv_tile = cv.cvtColor(tile, cv.COLOR_BGR2HSV)
    hue, saturation, value = np.median(hsv_tile, axis=(0, 1))

    tile_vector = np.array([hue, saturation, value], dtype=float)

    best_terrain = "Unknown"
    best_distance = float("inf")

    for terrain, stats in HSV_REFERENCE.items():
        if terrain == "Home":
            continue

        class_mean = np.array(stats["mean"], dtype=float)
        class_std = np.array(stats["std"], dtype=float)
        class_std[class_std == 0] = 1.0

        normalized_distance = np.linalg.norm((tile_vector - class_mean) / class_std)

        if normalized_distance < best_distance:
            best_distance = normalized_distance
            best_terrain = terrain

    if debug:
        print(f"Tile HSV: H={hue:.2f}, S={saturation:.2f}, V={value:.2f}")
        print(f"Chosen terrain without Home: {best_terrain}, norm_distance={best_distance:.2f}")

    return best_terrain


def get_terrain(tile, debug=False, home_threshold=0.80):
    """
    Beholdt for compatibility.
    Hvis du klassificerer ét tile alene.
    """
    home_detected, home_score = is_home(tile, threshold=home_threshold, debug=debug)

    if home_detected:
        if debug:
            print(f"Home detected with score {home_score:.3f}")
        return "Home"

    return get_terrain_without_home(tile, debug=debug)


def build_terrain_grid(
    tiles,
    debug=False,
    home_template_threshold=0.80,
    home_hsv_max_distance=3.0,
    template_weight=1.0,
    hsv_weight=0.15
):
    """
    Bygger terrain grid, hvor kun ét tile kan blive Home.

    Strategy:
    1. Alle tiles klassificeres først uden Home via HSV.
    2. For alle tiles beregnes:
       - template score for Home
       - HSV distance til Home
    3. En combined score beregnes:
       combined_score = template_weight * template_score - hsv_weight * hsv_distance
    4. Kun bedste kandidat kan blive Home, hvis:
       - template score >= home_template_threshold
       - hsv distance <= home_hsv_max_distance
    """
    terrain_grid = []
    home_template_scores = []
    home_hsv_distances = []
    combined_scores = []

    for row in tiles:
        terrain_row = []
        template_row = []
        hsv_row = []
        combined_row = []

        for tile in row:
            terrain_type = get_terrain_without_home(tile, debug=debug)
            template_score = get_home_score(tile, debug=False)
            hsv_distance = get_home_hsv_distance(tile, debug=False)

            combined_score = (
                template_weight * template_score
                - hsv_weight * hsv_distance
            )

            terrain_row.append(terrain_type)
            template_row.append(template_score)
            hsv_row.append(hsv_distance)
            combined_row.append(combined_score)

        terrain_grid.append(terrain_row)
        home_template_scores.append(template_row)
        home_hsv_distances.append(hsv_row)
        combined_scores.append(combined_row)

    best_row = 0
    best_col = 0
    best_combined_score = -float("inf")

    for row in range(len(combined_scores)):
        for col in range(len(combined_scores[row])):
            if combined_scores[row][col] > best_combined_score:
                best_combined_score = combined_scores[row][col]
                best_row = row
                best_col = col

    best_template_score = home_template_scores[best_row][best_col]
    best_hsv_distance = home_hsv_distances[best_row][best_col]

    terrain_grid[best_row][best_col] = "Home"

    if debug:
        print(
            f"Best Home candidate: ({best_row}, {best_col}) | "
            f"template={best_template_score:.3f} | "
            f"hsv_distance={best_hsv_distance:.3f} | "
            f"combined={best_combined_score:.3f}"
        )

    return terrain_grid


def predict_terrain_grid_from_image(
    image,
    board_size=500,
    grid_size=5,
    debug=False,
    home_template_threshold=0.80,
    home_hsv_max_distance=3.0,
    template_weight=1.0,
    hsv_weight=0.15
):
    """
    Fælles predictionfunktion som bruges af alle andre scripts.
    Det er den her, der sikrer at evaluering og visualisering bruger samme pipeline.
    """
    tiles = get_tiles(image, board_size=board_size, grid_size=grid_size)

    terrain_grid = build_terrain_grid(
        tiles,
        debug=debug,
        home_template_threshold=home_template_threshold,
        home_hsv_max_distance=home_hsv_max_distance,
        template_weight=template_weight,
        hsv_weight=hsv_weight
    )

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

    image_files.sort(
        key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else x
    )
    return image_files


def process_board(
    image_path,
    board_name,
    debug=False,
    save_tiles=True,
    tiles_root_folder="tiles",
    home_template_threshold=0.80,
    home_hsv_max_distance=3.0,
    template_weight=1.0,
    hsv_weight=0.15
):
    """
    Behandler ét board:
    læser billede, splitter i tiles, gemmer tiles hvis nødvendigt,
    bygger terrain grid og finder clusters.
    """
    image = cv.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Kunne ikke læse billedet: {image_path}")

    tiles = get_tiles(image, board_size=500, grid_size=5)

    if save_tiles:
        output_folder = os.path.join(tiles_root_folder, board_name)
        save_tiles_if_needed(tiles, output_folder)

    terrain_grid = predict_terrain_grid_from_image(
        image=image,
        board_size=500,
        grid_size=5,
        debug=debug,
        home_template_threshold=home_template_threshold,
        home_hsv_max_distance=home_hsv_max_distance,
        template_weight=template_weight,
        hsv_weight=hsv_weight
    )

    clusters = find_clusters(terrain_grid)

    return terrain_grid, clusters


def main():
    dataset_folder = "king_domino_dataset"
    run_all_boards = False
    single_board_file = "1.jpg"

    if not os.path.exists(dataset_folder):
        print(f"Dataset mappen blev ikke fundet: {dataset_folder}")
        return

    if run_all_boards:
        image_files = get_image_files(dataset_folder)
    else:
        image_files = [single_board_file]

    if not image_files:
        print("Ingen billeder fundet i dataset mappen.")
        return

    print(f"Antal boards der behandles: {len(image_files)}")

    first_board_visualized = False

    for file_name in image_files:
        image_path = os.path.join(dataset_folder, file_name)
        board_name = os.path.splitext(file_name)[0]

        print("\n" + "=" * 50)
        print(f"Behandler board: {file_name}")

        terrain_grid, clusters = process_board(
            image_path=image_path,
            board_name=board_name,
            debug=False,
            save_tiles=True,
            tiles_root_folder="tiles",
            home_template_threshold=0.80,
            home_hsv_max_distance=3.0,
            template_weight=1.0,
            hsv_weight=0.15
        )

        print("\nTerrain grid:")
        for row in terrain_grid:
            print(row)

        print("\nSammenhængende områder:")
        for terrain, cluster in clusters:
            print(f"{terrain}: {len(cluster)} felter -> {cluster}")

        if not first_board_visualized:
            visualize_board(terrain_grid, clusters, board_name=board_name)
            first_board_visualized = True

    print("\nFærdig.")


if __name__ == "__main__":
    main()