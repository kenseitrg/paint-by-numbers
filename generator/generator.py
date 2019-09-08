import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import Voronoi
from tqdm import tqdm


def get_voronoi_diagram(xs, ys, radius=None):
    assert xs.shape == ys.shape

    data = np.vstack((xs, ys)).T
    vor = Voronoi(data)

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()
    # Construct a map containing all ridges for a
    # given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points,
                                  vor.ridge_vertices):
        all_ridges.setdefault(
            p1, []).append((p2, v1, v2))
        all_ridges.setdefault(
            p2, []).append((p1, v1, v2))
    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]
        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue
        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]
        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue
            # Compute the missing endpoint of an
            # infinite ridge
            t = vor.points[p2] - \
                vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal
            midpoint = vor.points[[p1, p2]]. \
                mean(axis=0)
            direction = np.sign(
                np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + \
                direction * radius
            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())
        # Sort region counterclockwise.
        vs = np.asarray([new_vertices[v]
                         for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(
            vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[
            np.argsort(angles)]
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


def generate_image(n_digits: int = 9, min_font: int = 12, max_font: int = 20, max_digit: int = 10,
                   img_width: int = 416, img_height: int = 416):
    assert img_width // (2 * max_font) > n_digits
    assert img_height // (2 * max_font) > n_digits
    img = Image.new(mode="RGB", size=(img_width, img_height), color="#ffffff")
    draw = ImageDraw.Draw(img, mode="RGB")
    draw.rectangle([(0, 0), (img_width-1, img_height-1)], outline="#000000", fill="#ffffff")\

    font_size = np.random.randint(min_font, max_font)
    xs = np.random.randint(font_size, img_width-font_size, size=n_digits)
    ys = np.random.randint(font_size, img_height-font_size, size=n_digits)

    xs = np.array(range(font_size, img_width-font_size, 2*font_size))
    ys = np.array(range(font_size, img_height-font_size, 2*font_size))

    np.random.shuffle(xs)
    np.random.shuffle(ys)
    xs = xs[:n_digits]
    ys = ys[:n_digits]

    for i in range(n_digits):
        font = ImageFont.truetype("arial.ttf", font_size)
        x, y = xs[i], ys[i]
        text = str(np.random.randint(1, max_digit))
        draw.text((x, y), text, fill="#000000", font=font)

    new_regions, new_vertices = get_voronoi_diagram(xs, ys)

    for region in new_regions:
        region_to_draw = list(new_vertices[region].astype(np.int))
        region_to_draw = list(map(tuple, region_to_draw))
        draw.polygon(region_to_draw, outline="#000000")

    return img


print("Generating train set")
for i in tqdm(range(2500)):
    img = generate_image()
    img.save(f"images/train/train_{i}.png", "PNG")

print("Generating test set")
for i in tqdm(range(1000)):
    img = generate_image()
    img.save(f"images/train/test_{i}.png", "PNG")
