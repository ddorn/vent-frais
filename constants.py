import enum
from pathlib import Path

DATA = Path(__file__).parent / 'data'
QUESTION_PATH = DATA / 'questions.csv'
DECK_PATH = DATA / 'deck.json'
WIND_PATH = DATA / 'wind.npy'

FONT_FILE = 'font/ArbutusSlab-Regular.ttf'
FONT_SIZE = 30
TOP_MARGIN = 100
MARGIN = 30
LINE_SPACING = 0
CANVAS_SIZE = 500

SHOW_VOR = False
SHOW_DESIGN = True
PROP_CIRCLE = 0.5
LINE_WIDTH = 5
SHRINK_FACTOR = 0.85

LOGO_H = 353*0.3
LOGO_W = 273*0.3

class Category(enum.Enum):
    PERSO_EASY = 1
    PERSO_HARD = 2
    ABOUT_THE_WORLD = 3
    REGARD_SUR_LE_MONDE = 4

    @property
    def logo_name(self):
        base = {
            self.PERSO_EASY: 'perso-easy',
            self.PERSO_HARD: 'perso-hard',
            self.ABOUT_THE_WORLD: 'about-the-world',
            self.REGARD_SUR_LE_MONDE: 'view-about-the-world',
        }[self]
        return f'logo/logo-{base}-raw.svg'

COLOR_PALETTES = {
    Category.PERSO_EASY: {"background": "#01132c", "dots" : ["#fed9b7", "#ffffff", "#f07167"], "lines" : ["#ffffff", "#00afb9", "#0081a7"]},
    Category.PERSO_HARD: {"background": "#01132c", "dots" : ["#f86624", "#f9c80e", "#ea3546"], "lines" : ["#cee076", "#43bccd", "#662e9b"]},
    Category.ABOUT_THE_WORLD: {"background": "#01132c", "dots" : ["#a4574d", "#43bccd", "#cee076"], "lines" : ["#a4574d", "#43bccd", "#cee076"]},
    Category.REGARD_SUR_LE_MONDE: {"background": "#01132c", "dots" : ["#41ead4", "#fbff12", "#ff206e"], "lines" : ["#41ead4", "#fbff12", "#ff206e"]}
}
