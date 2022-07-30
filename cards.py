#!/bin/env python3

from __future__ import annotations
import itertools

import json
import os
from typing import Callable
import click
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

from tqdm import tqdm
import pygame
import pikepdf
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from constants import *
from card_draw import draw_card, generate_all_shapes

COLORS = [[-1, -1, -1], [255, 165, 0], [230, 240, 250], [65, 160, 100],
          [40, 67, 120]]


def thue_gen():
    n = 0
    while True:
        n += 1
        yield n.bit_count() % 2


def thue_morse_random(bound):
    t = thue_gen()

    yield
    while True:
        n = sum(2**i * next(t) for i in range(bound.bit_count() * 2)) % bound
        bound = yield n


def iterate_in_squares(bound=-1):
    """Yield all the points in Z², ordered by distance to the origin (in |·|_{sup})"""
    r = 0
    while r != bound:
        for d in range(-r, r + 1):
            yield r, d
            yield -r, d
            if abs(d) != r:
                yield d, r
                yield d, -r
        r += 1


def print_square(color: int, txt='', fg=False):
    r, g, b = COLORS[int(color)]
    txt = (str(txt) + '  ')[:2]
    code = 38 if fg else 48
    print(f'\033[{code};2;{r};{g};{b}m{txt}\033[0m', end='')


def get_board(radius: int):
    """
    Return a (2r+1)×(2r+1) board of the types of the tiles.
    The center is in (r, r).
    Larger boards are identical with smaller boards on their intersection.
    """

    # Deterministic random generator
    t = thue_morse_random(4)
    next(t)

    board = -np.ones((2 * radius + 1, 2 * radius + 1))
    for dx, dy in iterate_in_squares(radius + 1):
        x = radius + dx
        y = radius + dy

        # N/S/E/W neighbours if they exist
        taken = {
            board[x - 1, y] if x - 1 >= 0 else None,
            board[x + 1, y] if x + 1 < board.shape[0] else None,
            board[x, y - 1] if y - 1 >= 0 else None,
            board[x, y + 1] if y + 1 < board.shape[1] else None,
        }
        options = {1, 2, 3, 4} - set(taken)
        # tile = random.choice(list(options))
        tile = sorted(list(options))[t.send(len(options))]
        board[x, y] = tile

    return board


@dataclass(frozen=True)
class Question:
    statement: str
    position: tuple[int, int]
    category: Category

    def name(self, is_front: bool, pdf: bool = False):
        side = 'front' if is_front else 'back'
        ext = 'pdf' if pdf else 'svg'
        return f'card_{self.position[0]}_{self.position[1]}_{side}.{ext}'

    def gen_svg(self, shapes, is_face: bool = False):

        def heatmap(f):
            v = [[f(x / 100, y / 100) for x in range(100)] for y in range(100)]
            plt.imshow(v)

        # heatmap(angle)
        # plt.show()
        # heatmap(speed)
        # plt.show()

        metrics = get_text_metrics(self.statement)
        return draw_card(self.category, shapes, self.position, metrics,
                         is_face)

    def to_dict(self):
        return {
            'statement': self.statement,
            'position': self.position,
            'category': self.category.name,
        }

    @classmethod
    def from_dict(cls, d: dict):
        category = Category[d['category']]
        return cls(
            d['statement'],
            (d['position'][0], d['position'][1]),
            category,
        )


@dataclass
class Deck:
    cards: list[Question]
    shapes: list[dict[str, float]]

    @classmethod
    def load(cls, path: Path = DECK_PATH) -> Deck:
        d = json.loads(path.read_text())
        return cls(
            [Question.from_dict(q) for q in d['cards']],
            d['shapes'],
        )

    def to_json(self) -> str:
        s = json.dumps({
            'cards': [q.to_dict() for q in self.cards],
            'shapes': self.shapes,
        })
        return s

    def new_card(self, category: Category, prompt: str) -> None:
        used_positions = {q.position for q in self.cards}
        radius = self.radius
        board = get_board(radius + 2)

        for pos in iterate_in_squares(radius + 2):
            x = pos[0] + radius + 2
            y = pos[1] + radius + 2
            if pos not in used_positions and board[x, y] == category.value:
                # We found the new position
                self.cards.append(Question(prompt, pos, category))
                break

    def at(self, x, y):
        for q in self.cards:
            if q.position == (x, y):
                return q
        raise IndexError(f'No card at position {x},{y}.')

    @property
    def radius(self) -> int:
        if not self.cards:
            return 0
        return max((max(abs(p[0]), abs(p[1]))
                    for p in (q.position for q in self.cards)),
                   default=0)

    def show(self):
        by_pos = {tuple(q.position): q for q in self.cards}

        ids = []
        radius = self.radius + 1
        board = get_board(radius)
        for v in range(2 * radius + 1):
            for u in range(2 * radius + 1):
                q = by_pos.get((u - radius, v - radius), None)
                if q is None:
                    print_square(board[u, v], '. ', True)
                else:
                    print_square(q.category.value, len(ids))
                    ids.append(q)
            print()
        print()

        for i, q in enumerate(ids):
            print_square(q.category.value, i)
            x, y = q.position
            p = f' ({x},{y}):\t'
            print(p + q.statement)


class WindMap:

    def __init__(self, wind: np.ndarray, scale: float = 1.0) -> None:
        self.scale = scale
        u = gaussian_filter(wind[..., 0], WIND_BLUR)
        v = gaussian_filter(wind[..., 1], WIND_BLUR)
        self.wind = np.stack((u, v), axis=-1)

    def gps_to_index(self, lat: float, lon: float) -> tuple[float, float]:
        """
        Convert GPS coordinates to index in the wind array.
        """
        # print('lat, lon', lat, lon, self.wind.shape)
        x = (180 + lon) % 360 / 360 * self.wind.shape[1]
        y = (90 + lat) % 180 / 180 * self.wind.shape[0]
        # print('xy', x, y)
        return x, y

    @staticmethod
    def bilinear_interpolation(array: np.ndarray, x: float, y: float):
        """
        Interpolate a 2D array using bilinear interpolation.
        """
        x0 = int(x)
        x1 = x0 + 1
        y0 = int(y)
        y1 = y0 + 1
        x_diff = x - x0
        y_diff = y - y0
        return ((1 - x_diff) * (1 - y_diff) * array[y0, x0] + x_diff *
                (1 - y_diff) * array[y0, x1] +
                (1 - x_diff) * y_diff * array[y1, x0] +
                x_diff * y_diff * array[y1, x1])

    @staticmethod
    def gps_from_equal_earth(x, y):
        iterations = 20
        limit = 1e-8
        A1 = 1.340264
        A2 = -0.081106
        A3 = 0.000893
        A4 = 0.003796
        A23 = A2 * 3.
        A37 = A3 * 7.
        A49 = A4 * 9.
        M = np.sqrt(3.) / 2.
        # Use Newtons Method, where:
        #   fy is the function you need the root of
        #   dy is the derivative of the function
        #   dp is fy/dy or the change in estimate.
        # p = y.copy()    # initial estimate for parametric latitude
        p = y  # we don't use arrays
        # Note y is a reference, so as p changes, so would y,
        # so make local copy, otherwise the changed y affects results
        dp = 0.  # no change at start
        for i in range(iterations):
            p -= dp
            p2 = p**2
            p6 = p**6
            fy = p * (A1 + A2 * p2 + p6 * (A3 + A4 * p2)) - y
            dy = A1 + A23 * p2 + p6 * (A37 + A49 * p2)
            dp = fy / dy
            print('p', i, p, np.sin(p), np.sin(p) / M)
            if (np.abs(dp) < limit): break
        long = M * x * dy / np.cos(p)
        lat = np.arcsin(np.sin(p) / M)
        return long, lat

    def wind_at(self, x, y) -> tuple[float, float]:
        # lat, lon = self.gps_from_equal_earth(x * self.scale, y * self.scale)
        lat, lon = x * self.scale, y * self.scale
        x, y = self.gps_to_index(lat, lon)

        try:
            u = self.bilinear_interpolation(self.wind[..., 0], x, y)
            v = self.bilinear_interpolation(self.wind[..., 1], x, y)
        except IndexError:
            return 0, 0

        return u, v

    def angle_at(self, x, y):
        u, v = self.wind_at(x, y)
        return np.arctan2(v, u)

    def speed_at(self, x, y):
        u, v = self.wind_at(x, y)
        return u**2 + v**2


_TEXT = "Quel événement de ton enfance à eu le plus d'impact sur ce que tu fais aujourd'hui ?"


def get_text_metrics(
        text: str = _TEXT,
        font_size=FONT_SIZE,
        top_margin=TOP_MARGIN,
        margin=MARGIN,
        line_spacing=LINE_SPACING,
        canvas_size=CANVAS_SIZE,
        font_file=FONT_FILE) -> list[tuple[str, tuple[int, int], pygame.Rect]]:

    pygame.init()
    font = pygame.font.Font(font_file, font_size)

    def wrapped_text(txt: str, max_width):
        words = txt.split(' ')
        lines = []
        while words:
            line: list[str] = []
            while font.size(" ".join(line))[0] < max_width:
                if not words:
                    break
                line.append(words.pop(0))
            else:
                words.insert(0, line.pop())
            lines.append(' '.join(line))
        return lines

    def center_text(midtop: tuple[int, int], *lines: str):
        rects = []
        y = midtop[1]
        for line in lines:
            r = pygame.Rect(0, 0, *font.size(line))
            r.midtop = midtop[0], y
            rects.append(r)
            y += r.height + line_spacing

        return rects

    lines = wrapped_text(text, canvas_size - margin * 2)
    rects = center_text((canvas_size // 2, top_margin), *lines)

    descent = font.get_descent()
    ascent = font.get_ascent()
    return [(line, (rect.left, rect.top + ascent), rect)
            for line, rect in zip(lines, rects)]


# def load_questions() -> list[Question]:
#     questions = csv.reader(QUESTION_PATH.open('r'))
#     return [
#         Question(
#             statement=row[0],
#             tags=row[1].split(', '),
#         ) for row in questions
#     ]

# ------------------------ #
#  Command line interface  #
# ------------------------ #


@click.group()
def cli():
    """Utility to manage decks of Vent Frais."""
    pass


@cli.command(name='convert')
@click.argument('file', type=click.Path(exists=True))
@click.argument('out',
                type=click.Path(writable=True, dir_okay=False),
                default=WIND_PATH)
def convert_gfs_data(file: str, out: str):
    """
    Convert GFS forcasts into numpy arrays with wind velocities.

    Forcasts can be found here:
    https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl
    One needs to select:
     - the file ending in f000,
     - The desired level (eg. 10 m above ground)
     - The variables UGRD and VGRD
    """

    import xarray as xr

    ds = xr.load_dataset(file, engine='cfgrib')
    u = ds.u10.data
    v = ds.v10.data
    velocities = np.stack((u, v), axis=-1)
    np.save(out, velocities)

    return velocities


@cli.command(name='shapes')
@click.argument('file', type=click.Path(exists=True))
@click.argument('out', type=click.File('w'))
@click.option('-s', '--scale', type=float, default=20.0)
@click.option('-d', '--min-density', type=float, default=20.0)
@click.option('-S', '--size', type=int, default=4)
def generate_shapes(file, out, scale, min_density, size):
    """Convert a wind .npy file into a shapefile."""

    wind = WindMap(np.load(file), scale)
    shapes = generate_all_shapes(
        wind.angle_at, lambda x, y: min_density + wind.speed_at(x, y), size)

    out.write(json.dumps(shapes))


@cli.command('add')
@click.argument('category', type=click.Choice(list(Category.__members__)))
@click.argument('prompt', type=click.STRING)
@click.option('-d',
              '--deck-path',
              type=click.Path(exists=True, path_type=Path),
              default=DECK_PATH)
def new_card(category, prompt, deck_path):
    """Create a new card and add it to the deck."""

    deck = Deck.load(deck_path)
    deck.new_card(Category[category], prompt)
    deck_path.parent.mkdir(parents=True, exist_ok=True)
    deck_path.write_text(deck.to_json())


@cli.command('new')
@click.argument('wind',
                type=click.Path(exists=True, path_type=Path),
                default=WIND_PATH)
@click.option('-s', '--scale', type=click.FLOAT, default=1)
@click.option('-o', '--output', type=click.File('w'), default='-')
def new_deck(wind, scale, output):
    """
    Create a new deck.
    """

    deck = Deck([], WindMap(np.load(wind), scale=scale))
    output.write(deck.to_json())


@cli.command('show')
@click.argument('deck', type=click.Path(exists=True, path_type=Path))
def show_deck(deck: Path):
    """
    Show a deck.
    """
    the_deck = Deck.load(deck)
    the_deck.show()


@cli.command('edit')
@click.argument('deck', type=click.Path(exists=True, path_type=Path))
@click.option('-s', '--shapefile', type=click.File())
def edit_deck(deck: Path, shapefile=None):
    """Edit a deck."""
    the_deck = Deck.load(deck)
    if shapefile is not None:
        the_deck.shapes = json.load(shapefile)
    deck.write_text(the_deck.to_json())


@cli.command('gen')
@click.argument('deck', type=click.Path(exists=True, path_type=Path))
@click.argument('x', type=int)
@click.argument('y', type=int)
@click.option('-s',
              '--show',
              is_flag=True,
              help='Directly show the image afterwards.')
@click.option('-b',
              '--back',
              is_flag=True,
              help='Generate the back of the card.')
@click.option('-o', '--output', type=click.Path(path_type=Path))
def gen_svg(deck, x, y, show, back, output: Path = None):
    deck = Deck.load(deck)
    card = deck.at(x, y)
    svg = card.gen_svg(deck.shapes, not back)

    if output is None:
        output = Path('out') / card.svg_name(not back)
    elif output.is_dir():
        output = output / card.svg_name(not back)
    assert output is not None  # for mypy
    output.parent.mkdir(parents=True, exist_ok=True)

    output.write_text(svg)
    click.secho(f'Saved to {output.absolute()}', fg='green')

    if show:
        os.system('firefox ' + str(output))


@cli.command('pdf')
@click.argument('deck', type=click.Path(exists=True, path_type=Path))
@click.option('-s', '--show', is_flag=True, help='Open the pdf afterwards.')
@click.option('-o',
              '--output',
              type=click.Path(allow_dash=False, dir_okay=False, writable=True),
              default='output.pdf')
@click.option('-C',
              '--cache-dir',
              type=click.Path(path_type=Path, file_okay=False),
              default='out')
# @click.option('-c', '--cards', type=list[str])
def generate_pdf(deck, show, output, cache_dir: Path):
    """Generate a PDF of the deck."""

    the_deck = Deck.load(deck)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Make sure all svg and pdf are in cache
    for card, is_face in tqdm(
            list(itertools.product(the_deck.cards, [True, False]))):
        pdf_path = cache_dir / card.name(is_face, pdf=True)
        svg_path = cache_dir / card.name(is_face, pdf=False)

        if not pdf_path.exists():
            # Ensure svg exists
            if not svg_path.exists():
                # click.secho(f'{progress} Generating {svg_path}: {card.statement}')
                svg = card.gen_svg(the_deck.shapes, is_face)
                svg_path.write_text(svg)
            else:
                svg = svg_path.read_text()

            # use inkscape to convert svg to pdf
            # click.secho(f'{progress} Generating {pdf_path}: {card.statement}')
            ret_code = os.system(
                f'inkscape "{svg_path}" --export-filename "{pdf_path}" 2> /dev/null'
            )
            assert ret_code == 0
        # else:  # pdf exists
        #     click.secho(f'{progress} Using cached pdf {pdf_path}', fg='yellow')

    # group cards four by four
    pages = [the_deck.cards[i:i + 4] for i in range(0, len(the_deck.cards), 4)]

    # Compute the positions of each cards
    margin = 0.5  # In centimeters
    card_size = 10.0
    page_width = 21
    page_height = 29.7
    positions = np.array([
        (margin, margin),
        (margin + card_size, margin),
        (margin, page_height - margin - card_size),
        (margin + card_size, page_height - margin - card_size),
    ])

    # convert from cm to 1/72 inch (unit of pdfs)
    unit = 0.39370079 * 72
    positions *= unit
    card_size *= unit

    # create two new pages for each group
    pdf = pikepdf.new()
    for page in tqdm(pages):
        recto = pdf.add_blank_page(page_size=(595, 842))  # A4
        verso = pdf.add_blank_page(page_size=(595, 842))

        for i, card in enumerate(page):
            front = pikepdf.open(cache_dir / card.name(True, pdf=True))
            recto.add_overlay(
                front.pages[0],
                pikepdf.Rectangle(*positions[i], *positions[i] + card_size))

            back = pikepdf.open(cache_dir / card.name(False, pdf=True))
            idx = i ^ 1  # horizontal flip
            verso.add_overlay(
                back.pages[0],
                pikepdf.Rectangle(*positions[idx],
                                  *positions[idx] + card_size))

    pdf.save(output)

    if show:
        os.system('firefox ' + str(output))


@cli.group('plot')
def plot():
    """Display various data."""
    pass


@plot.command('wind')
@click.argument('wind_file', type=click.Path(exists=True), default=WIND_PATH)
def plot_wind(wind_file: str):
    wind = np.load(wind_file)
    x = wind[..., 0]
    y = wind[..., 1]
    angle = np.arctan2(y, x)
    speed = x**2 + y**2
    # plt.streamplot(
    #     np.linspace(-90, 90, x.shape[1]),
    #     np.linspace(0, 360, x.shape[0]),
    #     x, y,
    #     density=10,
    # )
    plt.subplot(2, 1, 1)
    plt.imshow(angle, cmap='hsv')
    plt.subplot(2, 1, 2)
    plt.imshow(speed)
    plt.show()


@plot.command('shapefile')
@click.argument('shapefile', type=click.File())
def plot_shapefile(shapefile):
    shapes = json.load(shapefile)

    minx = float('inf')
    miny = minx
    maxx = -minx
    maxy = maxx
    for shape in shapes:
        # print(shape)
        if 'r' in shape:
            attrs = [('cx', 'cy')]
        else:
            attrs = ('x1', 'y1'), ('x2', 'y2')

        for ax, ay in attrs:
            minx = min(minx, shape[ax])
            maxx = max(maxx, shape[ax])
            miny = min(miny, shape[ay])
            maxy = max(maxy, shape[ay])

    print(f"{minx=} {maxx=} {miny=} {maxy=}")

    W, H = 1920, 1080

    def to_screen(*pos):
        if len(pos) == 1:
            pos = pos[0]

        pos: pygame.Vector2
        pos -= center
        pos = pos.elementwise() / extent
        pos = pos.elementwise() * W
        pos += (W / 2, H / 2)

        # rx = (x - minx) / (maxx - minx)
        # ry = (y - miny) / (maxy - miny)
        return pos

    center = pygame.Vector2()
    extent = max(maxx - minx, maxy - miny) / 10

    screen = pygame.display.set_mode((W, H))

    def redraw():
        screen.fill('#01132C')
        pygame.draw.line(screen, 'grey', to_screen(0, 100), to_screen(0, -100))
        pygame.draw.line(screen, 'grey', to_screen(100, 0), to_screen(-100, 0))
        pygame.draw.line(screen, 'grey', to_screen(1, 100), to_screen(1, -100))

        for shape in shapes:
            if 'r' in shape:
                p = shape['cx'], shape['cy']
                r = max(2, shape['r'] * SHRINK_FACTOR / extent * W)
                pygame.draw.circle(screen, '#F86624', to_screen(p), int(r))
            else:
                p1 = shape['x1'], shape['y1']
                p2 = shape['x2'], shape['y2']
                w = max(1, 0.01 / extent * W)
                pygame.draw.line(screen, '#CEE076', to_screen(p1),
                                 to_screen(p2), int(w))
                if w > 2:
                    pygame.draw.circle(screen, '#CEE076', to_screen(p1),
                                       int(w / 2) - 1)
                    pygame.draw.circle(screen, '#CEE076', to_screen(p2),
                                       int(w / 2) - 1)

    redraw()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            # quit on escape
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return
                elif event.key == pygame.K_j:
                    extent *= 1.5
                    redraw()
                elif event.key == pygame.K_k:
                    extent /= 1.5
                    redraw()
            elif event.type == pygame.MOUSEWHEEL:
                center.x += event.x * extent / 100
                center.y += event.y * extent / 100
                redraw()

        pygame.time.wait(10)
        pygame.display.flip()


@plot.command('squares')
@click.argument('radius', type=int, default=10)
def plot_squares(radius):
    board = get_board(radius)

    for y, row in enumerate(board):
        for x, tile in enumerate(row):
            print_square(tile, '**' if x == y == radius else '')
        print('\033[0m')


@cli.command()
@click.argument('text', type=click.STRING, default=_TEXT)
def text_test(text):
    W = 500
    TEXT_COLOR = (255, 255, 255)

    screen = pygame.display.set_mode((W, W))
    screen.fill((0, 0, 0))
    metrics = get_text_metrics(text)
    font = pygame.font.Font(FONT_FILE, FONT_SIZE)

    for line, _, rect in metrics:
        t = font.render(line, True, TEXT_COLOR)
        screen.blit(t, rect)
        pygame.draw.rect(screen, (0, 255, 0), rect, 1)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            # quit on escape
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return

        pygame.time.wait(100)
        pygame.display.flip()


if __name__ == '__main__':
    cli()
