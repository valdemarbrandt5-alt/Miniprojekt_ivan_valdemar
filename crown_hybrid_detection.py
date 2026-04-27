from crown_knn_detection import predict_crown_count_knn
from crown_detection import predict_crown_count


def predict_crown_count_hybrid(
    tile,
    database_features,
    database_labels,
    terrain_label=None,
    k=1,
    template_threshold=0.85,
    strong_score_threshold=0.90,
    iou_threshold=0.20,
    min_center_distance=10,
    max_crowns=3,
    debug=False
):
    knn_pred = predict_crown_count_knn(
        tile,
        database_features,
        database_labels,
        k=k
    )

    template_pred = predict_crown_count(
        tile,
        terrain_label=terrain_label,
        threshold=template_threshold,
        iou_threshold=iou_threshold,
        min_center_distance=min_center_distance,
        strong_score_threshold=strong_score_threshold,
        max_crowns=max_crowns,
        debug=debug
    )

    if terrain_label in ["Home", "Empty"]:
        return 0

    # Hvis de er enige, easy win
    if knn_pred == template_pred:
        return knn_pred

    # Hvis KNN finder crowns men template ikke gør,
    # stoler vi på KNN, fordi dataset har gentagne tiles.
    if knn_pred > 0 and template_pred == 0:
        return knn_pred

    # Hvis template finder crowns men KNN siger 0,
    # er det ofte falsk positiv, så vi holder os til 0.
    if knn_pred == 0 and template_pred > 0:
        return 0

    # Hvis begge finder crowns men forskelligt antal,
    # tag det laveste for at undgå overcount.
    return min(knn_pred, template_pred)


def build_crown_grid_hybrid(
    tiles,
    database_features,
    database_labels,
    terrain_grid=None,
    k=1,
    template_threshold=0.85,
    strong_score_threshold=0.90,
    iou_threshold=0.20,
    min_center_distance=10,
    max_crowns=3,
    debug=False
):
    crown_grid = []

    for row_index, row in enumerate(tiles):
        crown_row = []

        for col_index, tile in enumerate(row):
            terrain_label = None

            if terrain_grid is not None:
                terrain_label = terrain_grid[row_index][col_index]

            crown_count = predict_crown_count_hybrid(
                tile=tile,
                database_features=database_features,
                database_labels=database_labels,
                terrain_label=terrain_label,
                k=k,
                template_threshold=template_threshold,
                strong_score_threshold=strong_score_threshold,
                iou_threshold=iou_threshold,
                min_center_distance=min_center_distance,
                max_crowns=max_crowns,
                debug=debug
            )

            crown_row.append(crown_count)

        crown_grid.append(crown_row)

    return crown_grid