def find_neighbors(x, y, rows, cols):
    neighbors = []

    directions = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1)
    ]

    for dx, dy in directions:
        nx = x + dx
        ny = y + dy

        if 0 <= nx < cols and 0 <= ny < rows:
            neighbors.append((nx, ny))

    return neighbors


def dfs(terrain_grid, x, y, terrain_type, visited):
    rows = len(terrain_grid)
    cols = len(terrain_grid[0])

    stack = [(x, y)]
    cluster = []

    while stack:
        cx, cy = stack.pop()

        if (cx, cy) in visited:
            continue

        if terrain_grid[cy][cx] != terrain_type:
            continue

        visited.add((cx, cy))
        cluster.append((cx, cy))

        neighbors = find_neighbors(cx, cy, rows, cols)

        for nx, ny in neighbors:
            if (nx, ny) not in visited and terrain_grid[ny][nx] == terrain_type:
                stack.append((nx, ny))

    return cluster


def find_clusters(terrain_grid):
    if not terrain_grid or not terrain_grid[0]:
        return []

    rows = len(terrain_grid)
    cols = len(terrain_grid[0])

    visited = set()
    clusters = []

    for y in range(rows):
        for x in range(cols):
            if (x, y) not in visited:
                terrain_type = terrain_grid[y][x]
                cluster = dfs(terrain_grid, x, y, terrain_type, visited)
                clusters.append((terrain_type, cluster))

    return clusters


def calculate_cluster_score(
    cluster,
    crown_grid,
    zero_crowns_count_as_one=False
):
    area_size = len(cluster)

    crown_count = 0
    for x, y in cluster:
        crown_count += crown_grid[y][x]

    if zero_crowns_count_as_one and crown_count == 0:
        score = area_size
    else:
        score = area_size * crown_count

    return score, area_size, crown_count


def calculate_board_score(
    terrain_grid,
    crown_grid,
    ignore_terrain_types=None,
    zero_crowns_count_as_one=False
):
    if ignore_terrain_types is None:
        ignore_terrain_types = {"Home", "Empty"}

    clusters = find_clusters(terrain_grid)

    total_score = 0
    score_breakdown = []

    for terrain_type, cluster in clusters:
        if terrain_type in ignore_terrain_types:
            continue

        cluster_score, area_size, crown_count = calculate_cluster_score(
            cluster,
            crown_grid,
            zero_crowns_count_as_one=zero_crowns_count_as_one
        )

        total_score += cluster_score

        score_breakdown.append({
            "terrain": terrain_type,
            "cells": cluster,
            "area_size": area_size,
            "crowns": crown_count,
            "score": cluster_score
        })

    return total_score, score_breakdown


def print_clusters(clusters):
    print("\nSammenhængende områder:")
    for terrain, cluster in clusters:
        print(f"{terrain}: {len(cluster)} felter -> {cluster}")


def print_score_breakdown(total_score, score_breakdown):
    print("\nPointberegning:")
    print("=" * 50)

    for item in score_breakdown:
        print(
            f"{item['terrain']}: "
            f"{item['area_size']} felter × "
            f"{item['crowns']} kroner = "
            f"{item['score']} point"
        )

    print("=" * 50)
    print(f"Samlet score: {total_score}")


if __name__ == "__main__":
    terrain_grid = [
        ["Grass", "Lake", "Forest", "Forest", "Forest"],
        ["Grass", "Forest", "Forest", "Forest", "Grass"],
        ["Grass", "Swamp", "Home", "Forest", "Grass"],
        ["Grass", "Swamp", "Lake", "Grass", "Grass"],
        ["Forest", "Lake", "Lake", "Grass", "Field"]
    ]

    crown_grid = [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0],
        [0, 2, 0, 2, 1],
        [0, 1, 0, 1, 0]
    ]

    total_score, breakdown = calculate_board_score(
        terrain_grid,
        crown_grid,
        zero_crowns_count_as_one=False
    )

    print_score_breakdown(total_score, breakdown)