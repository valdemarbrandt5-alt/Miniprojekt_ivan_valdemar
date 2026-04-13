def find_neighbors(x, y, rows, cols):
    """
    Returnerer gyldige naboer i de 4 hovedretninger:
    op, ned, venstre, højre.
    """
    neighbors = []

    directions = [
        (-1, 0),  # op
        (1, 0),   # ned
        (0, -1),  # venstre
        (0, 1)    # højre
    ]

    for dx, dy in directions:
        nx = x + dx
        ny = y + dy

        if 0 <= nx < cols and 0 <= ny < rows:
            neighbors.append((nx, ny))

    return neighbors


def dfs(terrain_grid, x, y, terrain_type, visited):
    """
    DFS der finder ét sammenhængende område af samme terræntype.
    Returnerer en liste af koordinater i clusteren.
    """
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
    """
    Finder alle sammenhængende områder i terrain_grid.

    Input:
        terrain_grid: 2D liste, fx 5x5 med terrain labels

    Returnerer:
        clusters: liste af tuples på formen
                  (terrain_type, [(x1, y1), (x2, y2), ...])
    """
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


def print_clusters(clusters):
    """
    Printer clusters på en pæn måde.
    """
    print("\nSammenhængende områder:")
    for terrain, cluster in clusters:
        print(f"{terrain}: {len(cluster)} felter -> {cluster}")


if __name__ == "__main__":
    # Lille test eksempel
    test_grid = [
        ["Forest", "Forest", "Lake", "Field", "Field"],
        ["Forest", "Grassland", "Lake", "Field", "Mine"],
        ["Grassland", "Grassland", "Lake", "Mine", "Mine"],
        ["Swamp", "Grassland", "Home", "Home", "Mine"],
        ["Swamp", "Swamp", "Home", "Field", "Field"]
    ]

    clusters = find_clusters(test_grid)
    print_clusters(clusters)