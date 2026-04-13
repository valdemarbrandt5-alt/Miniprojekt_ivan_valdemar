import cv2
import os
import ast
from pprint import pformat

GRID_SIZE = 5
BOARD_SIZE = 500
TILE_SIZE = BOARD_SIZE // GRID_SIZE

DATASET_FOLDER = "king_domino_dataset"
OUTPUT_FILE = "ground_truth_progress.py"

terrain_labels = {
    ord("f"): "Field",
    ord("r"): "Forest",
    ord("l"): "Lake",
    ord("m"): "Mine",
    ord("s"): "Swamp",
    ord("g"): "Grass",
    ord("h"): "Home",
    ord("e"): "Empty"
}

terrain_colors = {
    "Field": (0, 255, 255),
    "Forest": (0, 180, 0),
    "Lake": (255, 100, 0),
    "Mine": (130, 130, 130),
    "Swamp": (80, 120, 80),
    "Grass": (100, 255, 100),
    "Home": (0, 165, 255),
    "Empty": (180, 180, 180)
}


def create_empty_board():
    return {
        "terrain": [["" for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)],
        "crowns": [[None for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    }


def load_existing_ground_truth():
    if not os.path.exists(OUTPUT_FILE):
        return {}

    try:
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            content = f.read().strip()

        if not content.startswith("GROUND_TRUTH ="):
            return {}

        dict_text = content.split("=", 1)[1].strip()
        data = ast.literal_eval(dict_text)
        return data

    except Exception as e:
        print(f"Kunne ikke læse eksisterende ground truth fil: {e}")
        return {}


def save_ground_truth(data):
    formatted = pformat(data, width=100, indent=4)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("GROUND_TRUTH = ")
        f.write(formatted)
        f.write("\n")


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


def is_board_complete(board_data):
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if board_data["terrain"][r][c] == "":
                return False
            if board_data["crowns"][r][c] is None:
                return False
    return True


def count_completed_tiles(board_data):
    count = 0
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if board_data["terrain"][r][c] != "" and board_data["crowns"][r][c] is not None:
                count += 1
    return count


def get_next_tile(row, col):
    """
    Hopper til næste felt mod højre.
    Hvis vi er sidst på rækken, hopper den til næste række.
    Hvis vi er sidste felt på boardet, returneres None.
    """
    if col < GRID_SIZE - 1:
        return row, col + 1

    if row < GRID_SIZE - 1:
        return row + 1, 0

    return None


def draw_board(display, board_data, board_name, board_index, total_boards, current_tile):
    overlay_height = 90

    canvas = cv2.copyMakeBorder(
        display,
        overlay_height,
        0,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(40, 40, 40)
    )

    title = f"Board {board_name}   ({board_index + 1}/{total_boards})"
    progress = f"Udfyldt: {count_completed_tiles(board_data)}/25"
    help_text = "Klik felt -> terrain: f/r/l/m/s/g/h/e -> crowns: 0/1/2/3   N: næste   B: tilbage   ESC: afslut"

    cv2.putText(canvas, title, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(canvas, progress, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 2)
    cv2.putText(canvas, help_text, (15, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)

    for i in range(1, GRID_SIZE):
        cv2.line(canvas, (0, overlay_height + i * TILE_SIZE), (BOARD_SIZE, overlay_height + i * TILE_SIZE), (0, 0, 0), 2)
        cv2.line(canvas, (i * TILE_SIZE, overlay_height), (i * TILE_SIZE, overlay_height + BOARD_SIZE), (0, 0, 0), 2)

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            x1 = c * TILE_SIZE
            y1 = overlay_height + r * TILE_SIZE
            x2 = x1 + TILE_SIZE
            y2 = y1 + TILE_SIZE

            terrain = board_data["terrain"][r][c]
            crowns = board_data["crowns"][r][c]

            if terrain != "":
                color = terrain_colors.get(terrain, (255, 255, 255))
                cv2.rectangle(canvas, (x1 + 2, y1 + 2), (x2 - 2, y2 - 2), color, 3)

                cv2.putText(
                    canvas,
                    terrain[0],
                    (x1 + 10, y1 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 0),
                    2
                )

            if crowns is not None:
                cv2.putText(
                    canvas,
                    str(crowns),
                    (x1 + 65, y1 + 85),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2
                )

    if current_tile is not None:
        row, col = current_tile
        x1 = col * TILE_SIZE
        y1 = overlay_height + row * TILE_SIZE
        x2 = x1 + TILE_SIZE
        y2 = y1 + TILE_SIZE
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 255), 3)

    return canvas


def main():
    if not os.path.exists(DATASET_FOLDER):
        print(f"Dataset mappe blev ikke fundet: {DATASET_FOLDER}")
        return

    image_files = get_image_files(DATASET_FOLDER)

    if not image_files:
        print("Ingen billeder fundet i dataset mappen.")
        return

    ground_truth = load_existing_ground_truth()

    current_board_index = 0
    current_tile = None
    awaiting_crowns = False

    cv2.namedWindow("Annotator")

    def click_event(event, x, y, flags, param):
        nonlocal current_tile, awaiting_crowns

        overlay_height = 90

        if event == cv2.EVENT_LBUTTONDOWN:
            if y < overlay_height:
                return

            col = x // TILE_SIZE
            row = (y - overlay_height) // TILE_SIZE

            if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE:
                current_tile = (row, col)
                awaiting_crowns = False
                print(f"Valgt felt: ({row},{col}). Vælg terrain først.")

    cv2.setMouseCallback("Annotator", click_event)

    while True:
        file_name = image_files[current_board_index]
        board_name = os.path.splitext(file_name)[0]
        image_path = os.path.join(DATASET_FOLDER, file_name)

        image = cv2.imread(image_path)

        if image is None:
            print(f"Kunne ikke læse billede: {image_path}")
            current_board_index += 1

            if current_board_index >= len(image_files):
                break

            continue

        image = cv2.resize(image, (BOARD_SIZE, BOARD_SIZE))

        if board_name not in ground_truth:
            ground_truth[board_name] = create_empty_board()

        board_data = ground_truth[board_name]

        while True:
            canvas = draw_board(
                image.copy(),
                board_data,
                board_name,
                current_board_index,
                len(image_files),
                current_tile
            )

            cv2.imshow("Annotator", canvas)
            key = cv2.waitKey(20) & 0xFF

            if key == 27:
                save_ground_truth(ground_truth)
                print(f"Gemte fremskridt i {OUTPUT_FILE}")
                cv2.destroyAllWindows()
                return

            if key in (ord("n"), ord("N")):
                save_ground_truth(ground_truth)
                current_tile = None
                awaiting_crowns = False
                current_board_index = min(current_board_index + 1, len(image_files) - 1)
                break

            if key in (ord("b"), ord("B")):
                save_ground_truth(ground_truth)
                current_tile = None
                awaiting_crowns = False
                current_board_index = max(current_board_index - 1, 0)
                break

            if current_tile is not None:
                row, col = current_tile

                if not awaiting_crowns and key in terrain_labels:
                    board_data["terrain"][row][col] = terrain_labels[key]
                    awaiting_crowns = True

                    print(f"Terrain sat til {terrain_labels[key]} på ({row},{col})")
                    print("Vælg crowns 0-3")

                elif awaiting_crowns and key in (ord("0"), ord("1"), ord("2"), ord("3")):
                    board_data["crowns"][row][col] = int(chr(key))
                    print(f"Crowns sat til {board_data['crowns'][row][col]} på ({row},{col})")

                    awaiting_crowns = False
                    save_ground_truth(ground_truth)

                    if is_board_complete(board_data):
                        print(f"Board {board_name} er færdigt. Går videre til næste board.")

                        current_board_index += 1

                        if current_board_index >= len(image_files):
                            save_ground_truth(ground_truth)
                            print("Alle boards er annoteret færdige.")
                            cv2.destroyAllWindows()
                            return

                        current_tile = None
                        break

                    next_tile = get_next_tile(row, col)

                    if next_tile is not None:
                        current_tile = next_tile
                        next_row, next_col = next_tile
                        print(f"Går automatisk til næste felt: ({next_row},{next_col})")
                        print("Vælg terrain først.")
                    else:
                        current_tile = None


if __name__ == "__main__":
    main()