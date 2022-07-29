from __future__ import annotations
from bisect import bisect_left, bisect_right
from random import uniform
from typing import Callable
import random as rd
from shapely.geometry import Polygon
import scipy
from scipy.spatial import Voronoi, voronoi_plot_2d
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

import drawSvg as draw

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import pygame

from constants import *


def cum_sum(l: list):
    ret = l[:]
    for i in range(1, len(l)):
        ret[i] += ret[i - 1]
    return [0] + ret


def random_weighted(cum_sums: list[float]) -> float:
    r = uniform(0, cum_sums[-1])
    x1 = bisect_right(cum_sums, r)
    x0 = x1 - 1
    # print(r, x0, len(cum_sums), cum_sums[-3:])
    y0 = cum_sums[x0]
    y1 = cum_sums[x1] if x0 != -1 else 0

    prop = (r - y0) / (y1 - y0)
    # probability is prop to: pdf = (y0 + prop * (y1 - y0)) / area
    area = (x1 - x0) * (y0 + y1) / 2
    # cdf = lambda x: (x * y0 + x ** 2 * (y1 - y0) / 2) / area
    # we want the x such that cdf(x) = prop
    a = (y1 - y0) / 2 / area
    b = y0 / area
    c = -prop
    # print('r x0 y0 y1', r, x0, y0, y1)
    # print('a b c', a, b, c)
    # we solve a*x**2 + b*x + c == 0
    x = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)
    # print('x', x)

    return x0 + x


def gen_points(density: Callable[[float, float], float] = lambda x, y: 1,
               grid_size: int = 200,
               rect: tuple[float, float, float, float]=(0, 0, 1, 1)):

    t, l, w, h = rect

    line_weights: np.ndarray = np.array([
        cum_sum([
            density(l + w * x / grid_size, t + h * y / grid_size)
            for y in range(grid_size + 1)
        ]) for x in range(grid_size + 1)
    ])

    col_weight = cum_sum([line_weights[x, -1] for x in range(grid_size)])

    while True:
        x = random_weighted(col_weight)
        p = x - int(x)
        y_weights = line_weights[int(x)] * (1 - p) + p * line_weights[int(x) +
                                                                      1]
        y = random_weighted(y_weights)
        point = np.array([l + w * x / grid_size, t + h * y / grid_size])
        print(point)
        yield point


def get_centroid(cell):
    return np.array(Polygon(cell).centroid.coords)[0]


def get_relaxed_points(density, square_side):

    points = []
    N = 800*(2 * square_side)**2
    MAX_ITERATION = 5  #10
    rho = 0.5  #0
    SAMPLING_MARGIN = 0.2
    MAX_MOUV = 0.002

    pts_gen = gen_points(density=density, rect=(-square_side, -square_side, 2*square_side, 2*square_side))

    for i in range(N):
        p = np.random.uniform(-SAMPLING_MARGIN-square_side, square_side + SAMPLING_MARGIN, 2)

        if -square_side < p[0] and p[0] < square_side and -square_side < p[1] and p[1] < square_side:
            points.append(next(pts_gen))
        else:
            points.append(p)

    points = np.array(points)

    for it in range(MAX_ITERATION):

        vor = Voronoi(points, qhull_options="Qbb Qz Qc")

        centroids = []

        for k in range(N):

            if -1 in vor.regions[vor.point_region[k]]:
                center = points[k]
            else:
                cell = vor.vertices[vor.regions[vor.point_region[k]]]
                center = get_centroid(cell)
            centroids.append(center)

        ## move toward centroids
        for k in range(N):
            vect = centroids[k] - points[k]
            if np.linalg.norm(vect) > MAX_MOUV:
                points[k] = points[k] + MAX_MOUV * vect / np.linalg.norm(vect)
            else:
                points[k] = (1 - rho) * points[k] + rho * centroids[k]

    vor = Voronoi(points, qhull_options="Qbb Qz Qc")

    inner_circle_radii = []
    for k in range(N):
        p3 = centroids[k]
        reg_corners = vor.regions[vor.point_region[k]]
        rad = 100
        for corner in range(len(reg_corners)):
            p1 = vor.vertices[reg_corners[corner - 1]]
            p2 = vor.vertices[reg_corners[corner]]
            #print(p1, p2, p3)
            rad = min(
                rad,
                np.linalg.norm(np.cross(p2 - p1, p1 - p3)) /
                np.linalg.norm(p2 - p1))
        inner_circle_radii.append(rad)

    return centroids, inner_circle_radii, vor


def getLine(cx, cy, angle, length, sf):
    l = length / 2
    x1 = cx + np.cos(angle) * l
    y1 = cy + np.sin(angle) * l

    x2 = cx - np.cos(angle) * l
    y2 = cy - np.sin(angle) * l
    return x1, y1, x2, y2


@dataclass
class ProtectedZones:
    logo_xy: np.ndarray
    logo_radius: float
    is_face: bool
    line_rects: list[pygame.Rect]
    line_width: float

    RECT_INFLATION = 30

    def collide(self, shape_dict, shape_type) -> bool:
        if shape_type == "circle":
            p = np.array([shape_dict["cx"], shape_dict["cy"]])
            r = shape_dict["r"]

            if self.is_face:
                for rect in self.line_rects:
                    inflation = self.RECT_INFLATION + r * 500
                    if rect.inflate(inflation,
                                    inflation).collidepoint(self.to_pygame(p)):
                        return True
                return False
            else:
                return np.linalg.norm(self.logo_xy - p) <= (self.logo_radius +
                                                            r)

        else:  # shape_type == "line":
            p1 = np.array([shape_dict["x1"], shape_dict["y1"]])
            p2 = np.array([shape_dict["x2"], shape_dict["y2"]])
            if self.is_face:
                inflation = self.RECT_INFLATION + self.line_width
                for rect in self.line_rects:
                    if rect.inflate(inflation, inflation).clipline(
                            self.to_pygame(p1), self.to_pygame(p2)):
                        return True
                return False
            else:
                return np.linalg.norm(self.logo_xy - p1) < self.logo_radius or \
                   np.linalg.norm(self.logo_xy - p2) < self.logo_radius

    def to_pygame(self, point):
        return (point[0] * 500, 500 - point[1] * 500)


Field2D = Callable[[float, float], float]


def is_in_square(shape, offset, l, m):

    def is_point_in_square(x, y):
        if x > offset[0]-m and x < offset[0]+l+m and \
            y > offset[1]-m and y < offset[1]+l+m:
            return True
        else:
            return False

    if shape["t"] == "c":
        return is_point_in_square(shape["cx"], shape["cy"])
    elif shape["t"] == "l":
        return is_point_in_square(shape["x1"], shape["y1"]) or \
                is_point_in_square(shape["x2"], shape["y2"])
    else:
        raise ValueError("Undefined shape type.")


def dist_to_rect(pos, rect: pygame.Rect):
    nearest = (
        np.clip(pos[0], rect.left, rect.right),
        np.clip(pos[1], rect.top, rect.bottom)
    )
    return np.linalg.norm(pos - nearest)


def draw_card(
    card_type: Category,
    shapes: list,
    card_position: tuple,
    text_metrics: list[tuple[str, tuple[int, int], pygame.Rect]],
    is_face: bool = False,
):

    # TODO seed the card

    S = 500

    shrink_factor = SHRINK_FACTOR
    offset = (card_position[0], card_position[1]) #position of the bottom left corner


    NoMansLand = ProtectedZones(
        logo_xy=np.array([0.5, 0.5]),
        logo_radius=LOGO_RADIUS,
        is_face=is_face,
        line_rects=[r for _, _, r in text_metrics],
        line_width=LINE_WIDTH,
    )

    d = draw.Drawing(S, S, origin=(0, 0), displayInline=False)

    d.append(
        draw.Rectangle(0,
                       0,
                       S,
                       S,
                       fill=COLOR_PALETTES[card_type]['background'],
                       rx=40,
                       ry=40))

    bg = draw.ClipPath()
    bg.append(
        draw.Rectangle(0,
                       0,
                       S,
                       S,
                       fill=COLOR_PALETTES[card_type]['background'],
                       rx=40,
                       ry=40))

    if is_face:
        # font size 1000 = 1 en coo
        for line, (x, y), rect in text_metrics:
            d.append(
                draw.Text(
                    [line],
                    30,
                    x,
                    500 - y,
                    fill='white',
                    style=
                    "font-style:normal;font-variant:normal;font-weight:normal;font-stretch:normal;font-family:'Arbutus Slab';-inkscape-font-specification:'Arbutus Slab'"
                ))

    else:
        #d.append(draw.Circle(NoMansLand.LOGO_XY[0]*S, NoMansLand.LOGO_XY[1]*S, NoMansLand.LOGO_RADIUS*S,
        #            fill="violet", stroke='none', style="opacity:0.5"))

        # d.append(draw.Image(NoMansLand.logo_xy[0]*S-LOGO_W/2,
        #                     NoMansLand.logo_xy[1]*S-LOGO_H/2, LOGO_W, LOGO_H,embed=True,
        #                     path=f"logo/logo-{card_type}.svg"))

        with open(card_type.logo_name, "r") as f:
            logo_svg = f.read()
        d.append(draw.Raw(logo_svg))

        #d.append(draw.Circle(NoMansLand.LOGO_XY[0]*S-LOGO_W/2,
        #                    NoMansLand.LOGO_XY[1]*S-LOGO_H/2, 0.01*S,
        #           fill="red", stroke='none'))

        #d.append(draw.Rectangle(NoMansLand.LOGO_XY[0]*S-LOGO_W/2,
        #                    NoMansLand.LOGO_XY[1]*S-LOGO_H/2,LOGO_W,LOGO_H, fill="white", style="opacity:0.5"))

    for i, shape in enumerate(shapes):
        if is_in_square(shape, offset, 1, 1):

            if shape["t"] == "c":  #if circle
                cx = shape["cx"] - offset[0]
                cy = shape["cy"] - offset[1]
                if is_face:  ## flip the face
                    cx = 1 - cx

                shape_color = rd.choice(COLOR_PALETTES[card_type]['dots'])
                if not NoMansLand.collide({
                        "cx": cx,
                        "cy": cy,
                        "r": shape["r"]
                }, "circle"):
                    d.append(
                        draw.Circle(cx * S,
                                    cy * S,
                                    shape["r"] * S * shrink_factor,
                                    fill=shape_color,
                                    stroke='none',
                                    clip_path=bg))

            elif shape["t"] == "l":
                shape_color = rd.choice(COLOR_PALETTES[card_type]['lines'])
                x1 = shape["x1"] - offset[0]
                y1 = shape["y1"] - offset[1]
                x2 = shape["x2"] - offset[0]
                y2 = shape["y2"] - offset[1]

                if is_face:  ## flip the face
                    x1 = 1 - x1
                    x2 = 1 - x2

                if not NoMansLand.collide(
                    {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2
                    }, "line"):
                    d.append(
                        draw.Line(x1 * S,
                                  y1 * S,
                                  x2 * S,
                                  y2 * S,
                                  fill='none',
                                  stroke_width=LINE_WIDTH,
                                  stroke=shape_color,
                                  stroke_linecap="round",
                                  clip_path=bg))
            else:
                raise ValueError("Undefined shape type.")

    d.setPixelScale(1.5)  # Set number of pixels per geometry unit

    return d.asSvg()


def generate_all_shapes(angles: Field2D = lambda x, y: 0,
                        intensity: Field2D = lambda x, y: 0,
                        square_side: int = 4) -> list[dict[str, float]]:
    np.random.seed(42)

    pts, radii, vor = get_relaxed_points(intensity, square_side)
    shape_file = []

    for i, p in enumerate(pts):

        if np.random.random() < PROP_CIRCLE:
            shape_file.append({
                "cx": p[0],
                "cy": p[1],
                "r": radii[i],
                "t": "c"
            })
        else:
            angle = angles(p[0], p[1])
            x1, y1, x2, y2 = getLine(p[0], p[1], angle, radii[i],
                                        SHRINK_FACTOR)
            shape_file.append(
                {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "t":"l"
                })

    return shape_file
