from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import Voronoi
import pandas as pd
from tqdm import tqdm
from utils.img_utils import display_image


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
    draw.rectangle([(0, 0), (img_width-1, img_height-1)], outline="#000000", fill="#ffffff")

    font_size = np.random.randint(min_font, max_font)
    xs = np.random.randint(font_size, img_width-font_size, size=n_digits)
    ys = np.random.randint(font_size, img_height-font_size, size=n_digits)

    xs = np.array(range(font_size, img_width-font_size, 2*font_size))
    ys = np.array(range(font_size, img_height-font_size, 2*font_size))

    np.random.shuffle(xs)
    np.random.shuffle(ys)
    xs = xs[:n_digits]
    ys = ys[:n_digits]

    targets = []

    for i in range(n_digits):
        font = ImageFont.truetype("arial.ttf", font_size)
        x, y = xs[i], ys[i]
        text = str(np.random.randint(1, max_digit))
        draw.text((x, y), text, fill="#000000", font=font)
        target = {"class": int(text), "box_x": x, "box_y": y, 
                    "box_width": font_size, "box_height": font_size,
                    "center_x": x+font_size//2, "center_y": y+font_size//2}
        targets.append(target)

    new_regions, new_vertices = get_voronoi_diagram(xs, ys)

    for region in new_regions:
        region_to_draw = list(new_vertices[region].astype(np.int))
        region_to_draw = list(map(tuple, region_to_draw))
        draw.polygon(region_to_draw, outline="#000000")

    return img, targets


if __name__ == "__main__":
    targets_train = []
    print("Generating train set")
    (Path.cwd() / "images" / "train").mkdir(parents=True, exist_ok=True)
    for i in tqdm(range(3000)):
        img, tgts = generate_image()
        for t in tgts:
            t["file"] = f"/images/train/train_{i}.png"
        targets_train += tgts
        img.save((Path.cwd() / "images" / "train" / f"train_{i}.png"), "PNG")
    train_df = pd.DataFrame(targets_train)
    train_df.to_excel((Path.cwd() / "images" / "train" / "train_target.xlsx"))
    
    targets_test = []
    print("Generating test set")
    (Path.cwd() / "images" / "test").mkdir(parents=True, exist_ok=True)
    for i in tqdm(range(500)):
        img, tgts = generate_image()
        for t in tgts:
            t["file"] = f"/images/test/test_{i}.png"
        targets_test += tgts
        img.save((Path.cwd() / "images" / "test" / f"test_{i}.png"), "PNG")
    test_df = pd.DataFrame(targets_test)
    test_df.to_excel((Path.cwd() / "images" / "test" / "test_target.xlsx"))
#img, targets = generate_image()
#display_image(img, targets)
