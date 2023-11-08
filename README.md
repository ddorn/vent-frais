# Vent Frais

You can download a copy of the cards on [the release page](https://github.com/ddorn/vent-frais/releases).
If you want to print it, I recommend using thick paper (≥300 g/m²) and a professional quality printer. Cards are best with a side of 10cm or 8cm.

![A sample card back](./sample.png)

## Installation
All dependencies can be installed via poetry:
```bash
poetry install
```

## Usage
```bash
poetry shell
./cards.py
```

```
$ ./cards.py
Usage: cards.py [OPTIONS] COMMAND [ARGS]...

  Utility to manage decks of Vent Frais.

Options:
  --help  Show this message and exit.

Commands:
  add      Create a new card and add it to the deck.
  convert  Convert GFS forcasts into numpy arrays with wind velocities.
  edit     Edit a deck.
  gen
  new      Create a new deck.
  pdf      Generate a PDF of the deck.
  plot     Display various data.
  shapes   Convert a wind .npy file into a shapefile.
  show     Show a deck.
```

To see an example of usage, see [script.sh](./script.sh).

## Types of files

- `grib2` files are the raw data files from the GFS model, which can then be converted to windfiles using `cards.py convert`.
    Forcasts can be found at
    https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl.
    One needs to select:
     - the file ending in f000,
     - The desired level (eg. 10 m above ground)
     - The variables UGRD and VGRD
- `windfile`s are numpy arrays of shape `(lat, lon, 2)` which contain the wind velocity at each point of the grid.
    They can be converted to shapefiles using `cards.py shapes`.
- `shapefile`s are json files which contain a single list of shapes. Shapes are
    dictionaries with a key "t" associated with the type of the shape (`c` for circle,
    `l` for line) and more parameters depending on the type (c: `cx`, `cy`, `r`; l: `x1`, `y1`, `x2`, `y2`).
- `deck`s are json files which contain a list of cards and a shapefile.


<!-- To tweak colors and categories, configuration is found in [`constants.py`](constants.py). -->
