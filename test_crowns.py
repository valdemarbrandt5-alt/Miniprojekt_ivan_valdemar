import cv2 as cv

from image_processing import get_tiles
from crown_detection import build_crown_grid

image = cv.imread("king_domino_dataset/1.jpg")

if image is None:
    raise FileNotFoundError("Kunne ikke læse board billede")

tiles = get_tiles(image, board_size=500, grid_size=5)

crown_grid = build_crown_grid(
    tiles,
    threshold=0.9,
    iou_threshold=0.20,
    max_crowns=3,
    debug=True
)

print("\nPredicted crown grid:")
for row in crown_grid:
    print(row)