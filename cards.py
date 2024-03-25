#!/bin/env python3

from __future__ import annotations

import csv
from functools import partial
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal, Optional

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

import click
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pikepdf
import pygame
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from card_draw import draw_card, generate_all_shapes
from constants import *

# For printing the cards in the terminal
COLORS = [[-1, -1, -1], [255, 165, 0], [230, 240, 250], [65, 160, 100],
          [40, 67, 120]]


def thue_gen() -> Iterator[int]:
    n = 0
    while True:
        n += 1
        yield n.bit_count() % 2


def thue_morse_random(bound):
    """Generate deterministic random-like numbers in [0, bound[ using the Thue-Morse sequence.

    The new bound must be sent every time to the generator in order to optain a new random number.

    Yields:
        int: A random number in [0, bound[
    Send:
        int: The new bound
    Usage:
        t = thue_morse_random(4)
        next(t)  # Skip the first yield
        for i in range(5, 10):
            print(t.send(i))
    """
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
    """Print a colored square with a text of at most two characters in it.

    Args:
        color (int): the index of the color in COLORS
        txt (str, optional): text two show (truncated to the first 2 chars). Defaults to ''.
        fg (bool, optional): Whether the color is in the forground or background. Defaults to False.
    """

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

    def gen_svg(self,
                shapes,
                is_face: bool = False,
                rounding: int = CARD_ROUNDING):

        metrics = get_text_metrics(self.statement)
        return draw_card(self.category, shapes, self.position, metrics,
                         is_face, rounding)

    def gen_pdf(self,
                shapes,
                is_face: bool,
                pdf_path: Path,
                rounding: int = CARD_ROUNDING):
        """Generate a pdf file for the card. Overwrites the file if it already exists."""
        svg_path = pdf_path.with_suffix('.svg')

        svg = self.gen_svg(shapes, is_face, rounding)
        svg_path.write_text(svg)

        # use inkscape to convert svg to pdf
        ret_code = os.system(
            f'inkscape "{svg_path}" --export-filename "{pdf_path}" 2> /dev/null'
        )
        assert ret_code == 0

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
    file: Optional[Path] = None

    @classmethod
    def load(cls, path: Path = DECK_PATH) -> Deck:
        d = json.loads(path.read_text())
        return cls(
            [Question.from_dict(q) for q in d['cards']],
            d['shapes'],
            path,
        )

    def save(self, path: Optional[Path] = None) -> None:
        if path is None:
            path = self.file
        if path is None:
            raise ValueError('No path given')

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())

    def to_json(self) -> str:
        s = json.dumps({
            'cards': [q.to_dict() for q in self.cards],
            'shapes': self.shapes,
        })
        return s

    def new_card(self, category: Category, prompt: str) -> None:
        """Find a fitting position for a new card and add it to the deck."""
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

        if not ids:
            print('No cards in deck.')


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
        return np.sqrt(u**2 + v**2)


_TEXT = "Quel événement de ton enfance à eu le plus d'impact sur ce que tu fais aujourd'hui ?"

@dataclass
class TextMetrics:
    text: str
    position: tuple[int, int]
    rect: pygame.Rect
    font_size: int

def get_text_metrics(
        text: str = _TEXT,
        font_size=FONT_SIZE,
        top_margin=TOP_MARGIN,
        margin=MARGIN,
        line_spacing=LINE_SPACING,
        canvas_size=CANVAS_SIZE,
        font_file=FONT_FILE) -> list[TextMetrics]:
    r"""Return each line of formated text with its position and rect.

    Lines starting with <big> will be rendered with a bigger font size until the next \n newline.
    <big> tags always start a new line.
    Consecutive spaces are replaced by a single space.
    """

    pygame.init()
    font_sizes = font_size, int(font_size * 1.5)
    fonts = [pygame.font.Font(font_file, f) for f in font_sizes]

    def wrapped_text(txt: str, max_width) -> list[tuple[str, bool]]:
        new_line_marker = '<NEWLINE>'
        # Make sure the <big> tags are on their own word
        txt = txt.replace('<big>', ' <big> ')
        txt = txt.strip().replace('\n', f' {new_line_marker} ')
        words = txt.split()
        lines = []
        big = False
        next_big = False
        while words:
            line: list[str] = []
            while fonts[big].size(" ".join(line))[0] < max_width:
                if not words:
                    break
                elif words[0] == new_line_marker:
                    words.pop(0)
                    next_big = False
                    break
                elif words[0] == '<big>':
                    # If we set big directly, and start a new line,
                    # then we forgot the size of the previous line
                    next_big = True
                    if line:
                        # The <big> tag starts a new line
                        break
                    big = next_big
                    words.pop(0)
                    continue

                # We tentatively add the word
                line.append(words.pop(0))
            else:
                # And put it back if it was too long
                words.insert(0, line.pop())

            lines.append([' '.join(line), big])
            big = next_big

        return lines

    def center_text(midtop: tuple[int, int], *lines: tuple[str, bool]):
        rects = []
        y = midtop[1]
        for line, big in lines:
            r = pygame.Rect(0, 0, *fonts[big].size(line))
            r.midtop = midtop[0], y
            rects.append(r)
            y += r.height + line_spacing

        return rects

    lines = wrapped_text(text, canvas_size - margin * 2)
    rects = center_text((canvas_size // 2, top_margin), *lines)

    return [TextMetrics(
        text=line,
        position=(rect.left, rect.top + fonts[big].get_ascent()),
        rect=rect,
        font_size=font_sizes[big],
    )
            for (line, big), rect in zip(lines, rects, strict=True)
            if rect.width and rect.height]  # Ignore empty lines


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

# Common arguments and options


class DeckParam(click.ParamType):
    name = 'deck'

    def convert(self, value, param, ctx):
        if value is None:
            return None
        if isinstance(value, Deck):
            return value
        if isinstance(value, str):
            path = Path(value)
            return Deck.load(path)
        self.fail(f'Invalid deck: {value!r}', param, ctx)


deck_argument = click.argument('deck', type=DeckParam())
output_file_option = click.option('-o',
                                  '--output',
                                  type=click.File('w'),
                                  default='-')
output_binary_file_option = partial(click.option,
                                    '-o',
                                    '--output',
                                    type=click.File('wb'))
cache_dir_option = click.option('-C',
                                '--cache-dir',
                                type=click.Path(path_type=Path,
                                                file_okay=False),
                                default='out')
overwrite_option = click.option('--overwrite/--no-overwrite',
                                default=False,
                                help='Overwrite existing pdfs/svg in cache.')
rounding_option = click.option('-r',
                               '--rounding',
                               type=int,
                               default=CARD_ROUNDING,
                               help="Rounding of the cards' corner.")
show_option = click.option('-s',
                           '--show',
                           is_flag=True,
                           help='Open the file for viewing.')

# For CSVs
has_header_option = click.option('-h', '--has-header', is_flag=True)
question_col_option = click.option('-q', '--question-col', type=int, default=0)
category_col_option = click.option('-c', '--category-col', type=int, default=1)
category_map_option = click.option(
    '-C',
    '--category-map',
    type=str,
    default=None,
    help=
    'Name of the categories. Fmt: name-perso-easy;name-perso-hard;name-word;name-vision'
)


@click.group()
def cli():
    """Utility to manage decks of Vent Frais."""
    pass


@cli.command(name='convert')
@click.argument('file', type=click.Path(exists=True))
@output_binary_file_option(default='wind.npy')
def convert_gfs_data(file: str, output):
    """
    Convert GFS forcasts into numpy arrays with wind velocities.

    Outputs a numpy array with shape (lat, lon, 2) and saves it.

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
    np.save(output, velocities)

    return velocities


@cli.command(name='shapes')
@click.argument('wind-file', type=click.Path(exists=True))
@output_file_option
@click.option('-s', '--scale', type=float, default=20.0)
@click.option('-d', '--min-density', type=float, default=20.0)
@click.option('-S',
              '--size',
              type=int,
              default=4,
              help='Generates shapes in [-size, size]^2')
def generate_shapes(wind_file, output, scale, min_density, size):
    """Convert a wind .npy file into a shapefile."""

    wind = WindMap(np.load(wind_file), scale)
    shapes = generate_all_shapes(
        wind.angle_at, lambda x, y: min_density + wind.speed_at(x, y), size)

    json.dump(shapes, output)


@cli.command('add')
@deck_argument
@click.argument('category', type=click.Choice(list(Category.__members__)))
@click.argument('prompt', type=click.STRING)
def new_card(deck: Deck, category, prompt):
    """Create a new card and add it to the deck."""

    deck.new_card(Category[category], prompt)
    deck.save()


# Passes a dictionnary of Question -> Category to subcommands
@cli.group('csv')
@click.argument('csv-file', type=click.File())
@click.option('-s', '--skip-first', type=int, default=0, help='Skip the first N lines of the file.')
@click.option('-q', '--question-col', type=int, default=0)
@click.option('-c', '--category-col', type=int, default=1)
@click.option(
    '-C',
    '--category-map',
    type=str,
    default=None,
    help=
    'Name of the categories. Fmt: name-perso-easy;name-perso-hard;name-word;name-vision'
)
@click.pass_context
def csv_group(ctx, csv_file, skip_first, question_col, category_col,
              category_map):
    """Commands to import cards from csv files."""
    reader = csv.reader(csv_file)

    if category_map is not None:
        category_map = category_map.split(';')
        assert len(category_map) == len(Category.__members__)

        def get_cat(name):
            """Get the category from the name according to the map."""
            for cat, enum_member in zip(category_map, Category):
                if name == cat:
                    return enum_member
            raise ValueError(
                f'Unknown category {name!r}. Valid categories are: {category_map}'
            )
    else:

        def get_cat(name):
            """Get the category from the name according to the map."""
            cat = name.upper().replace(' ', '_').replace('-_', '')
            try:
                return Category[name]
            except KeyError:
                raise ValueError(
                    f'Unknown category {cat}. Valid categories are: {list(Category.__members__.keys())}'
                ) from None

    for _ in range(skip_first):
        next(reader)

    questions = {
        row[question_col]: get_cat(row[category_col])
        for row in reader
    }
    ctx.obj = questions


@csv_group.command('add-to-deck')
@deck_argument
@click.pass_context
def new_card_from_csv(ctx, deck: Deck):
    """Add all the cards from a csv file to the deck.

    Does not check for duplicates and keep all the existing cards."""

    for question, category in ctx.obj.items():
        deck.new_card(category, question)
    deck.save()


@csv_group.command('sync-to-deck')
@deck_argument
@click.pass_context
@click.option('-f',
              '--force',
              is_flag=True,
              help='Do not ask for confirmation')
@show_option
@cache_dir_option
def sync_deck(ctx, deck: Deck, force: bool, show: bool, cache_dir: Path):
    """Sync the questions in the deck with those in the csv file.

    Remove all the cards in the deck that are not in the csv file.
    Then add all the cards from the csv file that are not in the deck.
    Removes the generated pdfs/svg from the cache if they are not in the deck anymore.
    """

    questions: dict[str, Category] = ctx.obj
    # Check for both the category and the prompt
    card_kept = [
        card for card in deck.cards
        if questions.get(card.statement, None) == card.category
    ]
    cards_to_add = [(name, category) for name, category in questions.items()
                    if all(card.statement != name or card.category != category
                           for card in deck.cards)]
    cards_to_remove = [
        card for card in deck.cards
        if questions.get(card.statement, None) != card.category
    ]

    if not cards_to_add and not cards_to_remove:
        print("No changes needed.")
        return

    # Get confirmation if needed
    if not force:
        if cards_to_remove:
            print('The following cards will be removed from the deck:')
            for card in cards_to_remove:
                print(f'  - {card.statement} ({card.category.name})')
        if cards_to_add:
            print('The following cards will be added to the deck:')
            for name, category in cards_to_add:
                print(f'  - {name} ({category.name})')
        if not click.confirm('Do you want to proceed?'):
            return

    deck.cards = card_kept
    for name, category in cards_to_add:
        deck.new_card(category, name)

    if show:
        deck.show()

    deck.save()

    paths_to_remove = [
        cache_dir / card.name(is_face, is_pdf)
        for card in cards_to_remove
        for is_face in [True, False]
        for is_pdf in [True, False]
    ]
    print('The following files should not be in the cache anymore:')
    print('\n - '.join(map(str, paths_to_remove)))
    if not force and click.confirm('Do you want to remove them?'):
        for path in paths_to_remove:
            (cache_dir / path).unlink(missing_ok=True)



@cli.command('new')
# @click.argument('wind',
#                 type=click.Path(exists=True, path_type=Path),
#                 default=WIND_PATH)
@click.argument('shapefile', type=click.File())
# @click.option('-s', '--scale', type=click.FLOAT, default=1)
@output_file_option
def new_deck(shapefile, output):
    """Create a new deck."""

    shapes = json.load(shapefile)

    deck = Deck([], shapes)
    output.write(deck.to_json())


@cli.command('show')
@deck_argument
def show_deck(deck: Deck):
    """Show a deck nicely."""
    deck.show()


@cli.command('edit')
@deck_argument
@click.option('-s', '--shapefile', type=click.File())
def edit_deck(deck: Deck, shapefile=None):
    """Edit a deck."""
    if shapefile is not None:
        deck.shapes = json.load(shapefile)
    deck.save()


@cli.command('gen')
@deck_argument
@click.argument('x', type=int)
@click.argument('y', type=int)
@show_option
@click.option('-b',
              '--back',
              is_flag=True,
              help='Generate the back of the card.')
@click.option('-o', '--output', type=click.Path(path_type=Path))
def gen_svg(deck: Deck, x, y, show, back, output: Optional[Path] = None):
    """Generate a single svg of a card at position X, Y in DECK."""
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
@deck_argument
@show_option
@output_binary_file_option(default='collage.pdf')
@cache_dir_option
@overwrite_option
@rounding_option
@click.option(
    '-p',
    '--pattern',
    type=click.Choice(['a4', 'single', 'collage-front', 'collage-back', 'dont-merge'],
                      case_sensitive=False),
    default='single',
    help=
    'The pattern to use to generate the pdf. A4 is 8 cards/page. Single is one page per card. Collage is a single page with all the cards in a grid.'
)
@click.option('-j', '--jobs', "n_jobs", type=int, default=-1)
def generate_pdf(deck: Deck, show, output, cache_dir: Path, overwrite: bool,
                 pattern: Literal['a4', 'single', 'collage-front',
                                  'collage-back'], n_jobs: int, rounding: int):
    """Generate a PDF of the deck."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    pdf_paths = {(is_face, card): cache_dir / card.name(is_face, pdf=True)
                 for card in deck.cards for is_face in [True, False]}

    # Make sure all svg and pdf are in cache
    # TODO: We could also check the integrity of the files (i.e. they are the right questions)
    need_to_generate = [
        key for key, path in pdf_paths.items()
        if not path.exists() or overwrite
    ]
    if not need_to_generate:
        click.secho('All pdfs are in cache.', fg='green')
    else:
        click.secho(
            f'{len(pdf_paths) - len(need_to_generate)} pdfs are in cache.',
            fg='green')
        joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(card.gen_pdf)(deck.shapes, is_face, pdf_paths[(
                is_face, card)], rounding)
            for is_face, card in tqdm(need_to_generate,
                                      desc='Generating svg+pdfs'))

    # Merge all pdfs into one
    if pattern == 'dont-merge':
        return
    elif pattern == 'single':
        cmd = f'pdftk {" ".join(map(str, pdf_paths.values()))} cat output {output.name} verbose'
        click.secho("$ " + cmd, fg='yellow')
        ret = os.system(cmd)
        assert ret == 0, f'pdftk failed with code {ret}'

    elif pattern == 'a4':
        # Compute the positions of each cards
        margin = 0.5  # In centimeters
        card_size = 7.0
        page_width = 21.0
        page_height = 29.7
        nb_cards_per_row = int((page_width - 2 * margin) // card_size)
        nb_cards_per_col = int((page_height - 2 * margin) // card_size)
        positions = np.empty((0, 2))
        for x in range(nb_cards_per_row):
            for y in range(nb_cards_per_col):
                pos = (x * card_size + margin, y * card_size + margin)
                positions = np.append(positions, [pos], axis=0)

        # group cards per page
        per_page = nb_cards_per_row * nb_cards_per_col
        pages = [
            deck.cards[i:i + per_page]
            for i in range(0, len(deck.cards), per_page)
        ]

        # positions = np.array([
        #     (margin, margin),
        #     (margin + card_size, margin),
        #     (margin, page_height - margin - card_size),
        #     (margin + card_size, page_height - margin - card_size),
        # ])

        # convert from cm to 1/72 inch (unit of pdfs)
        positions *= CM_TO_PDF_UNIT
        card_size *= CM_TO_PDF_UNIT
        # page_width *= unit
        page_width = 595  # exact size of A4 (askip)
        # page_height *= unit
        page_height = 842

        # create two new pages for each group
        pdf = pikepdf.new()
        for page in tqdm(pages, desc='Collecting pdfs'):
            recto = pdf.add_blank_page(page_size=(595, 842))  # A4
            verso = pdf.add_blank_page(page_size=(595, 842))

            for i, card in enumerate(page):
                front = pikepdf.open(cache_dir / card.name(True, pdf=True))
                recto.add_overlay(
                    front.pages[0],
                    pikepdf.Rectangle(*positions[i],
                                      *positions[i] + card_size))

                back = pikepdf.open(cache_dir / card.name(False, pdf=True))
                # idx = i ^ 1  # horizontal flip (when using the layout with 1 card in each angle)
                p = [page_width - margin - card_size, margin]
                verso.add_overlay(
                    back.pages[0],
                    pikepdf.Rectangle(page_width - positions[i][0] - card_size,
                                      positions[i][1],
                                      page_width - positions[i][0],
                                      positions[i][1] + card_size))
        pdf.save(output)

    elif pattern in ('collage-front', 'collage-back'):
        is_front = pattern == 'collage-front'

        # Output is a single page of size 2 * radius + 1
        scale = 10 * CM_TO_PDF_UNIT
        radius = deck.radius
        page_size = (2 * radius + 1) * scale
        pdf = pikepdf.new()
        page = pdf.add_blank_page(page_size=(page_size, page_size))

        # To add a background we generate a svg and convert it to pdf (not sure how to do it directly with pikepdf)
        bg_color = COLOR_PALETTES[Category.PERSO_EASY]['background']
        svg = f'<svg width="{page_size}" height="{page_size}"><rect width="{page_size}" height="{page_size}" style="fill:{bg_color}"/></svg>'
        svg_path = cache_dir / 'background.svg'
        svg_path.write_text(svg)
        ret_code = os.system(
            f'inkscape "{svg_path}" --export-filename "{cache_dir / "background.pdf"}" 2> /dev/null'
        )
        assert ret_code == 0
        background = pikepdf.open(cache_dir / 'background.pdf')
        page.add_overlay(background.pages[0],
                         pikepdf.Rectangle(0, 0, page_size, page_size))

        # Add all cards
        for card in tqdm(deck.cards, desc='Collecting pdfs'):
            front = pikepdf.open(cache_dir / card.name(is_front, pdf=True))
            if is_front:
                # The edges of the back match, but we need to flip everything
                # for the front to work match, symetry axis is the center of the collage
                x = (radius - card.position[0]) * scale
            else:
                x = (card.position[0] + radius) * scale
            y = (card.position[1] + radius) * scale
            page.add_overlay(front.pages[0],
                             pikepdf.Rectangle(x, y, x + scale, y + scale))
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
    pygame.init()
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


@plot.command()
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
