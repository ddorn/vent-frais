from __future__ import annotations
from bisect import bisect_left
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
    return ret


def gen_points(density):
    W = 1
    H = 1
    GRID_SIZE = 0.002

    line_weights = [
        cum_sum([
            density(x * GRID_SIZE, y * GRID_SIZE)
            for y in range(int(H // GRID_SIZE))
        ]) for x in range(int(W // GRID_SIZE))
    ]

    col_weight = cum_sum(
        [line_weights[x][-1] for x in range(int(W // GRID_SIZE))])
    deja_vu = set()

    while True:
        r = uniform(0, col_weight[-1])
        x = bisect_left(col_weight, r)
        y = bisect_left(line_weights[x], uniform(0, line_weights[x][-1]))
        if not ((x, y) in deja_vu):
            yield (np.array([x, y]) * GRID_SIZE)
            deja_vu.add((x, y))


def get_centroid(cell):
    return np.array(Polygon(cell).centroid.coords)[0]


def get_relaxed_points():

    def density(x, y):
        return 1.

    points = []
    N = 1000
    MAX_ITERATION = 5  #10
    rho = 0.5  #0
    SAMPLING_MARGIN = 0.2
    MAX_MOUV = 0.002

    pts_gen = gen_points(density=density)

    for i in range(N):
        p = np.random.uniform(-SAMPLING_MARGIN, 1 + SAMPLING_MARGIN, 2)

        if 0 < p[0] and p[0] < 1 and 0 < p[1] and p[1] < 1:
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


def draw_card(
    card_type: Category,
    text_metrics: list[tuple[str, tuple[int, int], pygame.Rect]],
    is_face: bool = False,
    angles: Field2D = lambda x, y: 0,
    intensity: Field2D = lambda x, y: 0,
):

    # TODO seed the card

    S = 500
    shrink_factor = 0.85

    NoMansLand = ProtectedZones(
        logo_xy=np.array([0.5, 0.5]),
        logo_radius=0.128,
        is_face=is_face,
        line_rects=[r for _, _, r in text_metrics],
        line_width=5,
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

        with open(f"logo/logo-{card_type}-raw.svg", "r") as f:
            logo_svg = f.read()
        d.append(draw.Raw(logo_svg))

        #d.append(draw.Circle(NoMansLand.LOGO_XY[0]*S-LOGO_W/2,
        #                    NoMansLand.LOGO_XY[1]*S-LOGO_H/2, 0.01*S,
        #           fill="red", stroke='none'))

        #d.append(draw.Rectangle(NoMansLand.LOGO_XY[0]*S-LOGO_W/2,
        #                    NoMansLand.LOGO_XY[1]*S-LOGO_H/2,LOGO_W,LOGO_H, fill="white", style="opacity:0.5"))

    pts, radii, vor = get_relaxed_points()

    for i, p in enumerate(pts):
        if SHOW_DESIGN:
            if np.random.random() < PROP_CIRCLE:
                shape_color = rd.choice(COLOR_PALETTES[card_type]['dots'])
                if not NoMansLand.collide(
                    {
                        "cx": p[0],
                        "cy": p[1],
                        "r": radii[i]
                    }, "circle"):
                    d.append(
                        draw.Circle(p[0] * S,
                                    p[1] * S,
                                    radii[i] * S * shrink_factor,
                                    fill=shape_color,
                                    stroke='none',
                                    clip_path=bg))

            else:
                shape_color = rd.choice(COLOR_PALETTES[card_type]['lines'])
                angle = angles(p[0], p[1])
                x1, y1, x2, y2 = getLine(p[0], p[1], angle, radii[i],
                                         shrink_factor)
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

        if SHOW_VOR:
            #print(p, radii)

            d.append(
                draw.Circle(p[0] * S,
                            p[1] * S,
                            radii[i] * S,
                            fill='none',
                            stroke_width=0.5,
                            stroke='blue'))
            d.append(draw.Circle(p[0] * S, p[1] * S, 1., fill='black'))
            reg_corners = vor.regions[vor.point_region[i]]
            for corner in range(len(reg_corners)):
                if reg_corners[corner - 1] != -1 and reg_corners[corner] != -1:
                    p1 = vor.vertices[reg_corners[corner - 1]]
                    p2 = vor.vertices[reg_corners[corner]]
                    d.append(
                        draw.Line(p1[0] * S,
                                  p1[1] * S,
                                  p2[0] * S,
                                  p2[1] * S,
                                  stroke='red',
                                  stroke_width=0.5,
                                  fill='none'))
    d.setPixelScale(1.5)  # Set number of pixels per geometry unit

    return d.asSvg()
