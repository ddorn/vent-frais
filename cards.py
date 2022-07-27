#!/bin/env python3

import csv
import click
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint

import numpy as np
import matplotlib.pyplot as plt

QUESTION_PATH = Path(__file__).parent / 'data' / 'questions.csv'

@dataclass
class Question:
    statement: str
    tags: list[str]


@click.group()
def cli():
    pass

@cli.command(name='convert')
@click.argument('file', type=click.Path())
@click.argument('out', type=click.Path(), default='data/wind.npy')
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


@cli.command(name='plot')
@click.argument('wind_file', type=click.Path(), default='data/wind.npy')
def plot(wind_file: str):
    wind = np.load(wind_file)
    x = wind[..., 0]
    y = wind[..., 1]
    angle = np.arctan2(y, x)
    speed = x ** 2 + y ** 2
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


class WindMap:
    def __init__(self, wind: np.ndarray) -> None:
        self.wind = wind
        self.u = wind[..., 0]
        self.v = wind[..., 1]
        self.angle = np.arctan2(self.v, self.u)
        self.speed = self.u ** 2 + self.v ** 2

    def gps_to_index(self, lat: float, lon: float) -> tuple[float, float]:
        """
        Convert GPS coordinates to index in the wind array.
        """
        x = lon / 360 * self.wind.shape[1]
        y = (90 - lat) / 180 * self.wind.shape[0]
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
        return (
            (1 - x_diff) * (1 - y_diff) * array[y0, x0]
            + x_diff * (1 - y_diff) * array[y0, x1]
            + (1 - x_diff) * y_diff * array[y1, x0]
            + x_diff * y_diff * array[y1, x1]
        )

    @staticmethod
    def gps_from_equal_earth(self, x, y):
        iterations = 20
        limit = 1e-8
        A1 = 1.340264
        A2 = -0.081106
        A3 = 0.000893
        A4 = 0.003796
        A23 = A2 * 3.
        A37 = A3 * 7.
        A49 = A4 * 9.
        M = np.sqrt(3.)/2.
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
            fy = p*(A1 + A2*p2 + p6*(A3 + A4*p2)) - y
            dy = A1 + A23*p2 + p6*(A37 + A49*p2)
            dp = fy/dy
            if (np.abs(dp) < limit).all(): break
        long = M * x * dy/np.cos(p)
        lat = np.arcsin(np.sin(p)/M)
        return long, lat

    def angle_at(self, x, y, scale = 1):
        lat, lon = self.gps_from_equal_earth(x, y)
        x, y = self.gps_to_index(lat, lon)
        return self.bilinear_interpolation(self.angle, x * scale, y * scale)

    def speed_at(self, x, y, scale = 1):
        lat, lon = self.gps_from_equal_earth(x, y)
        x, y = self.gps_to_index(lat, lon)
        return self.bilinear_interpolation(self.speed, x * scale, y * scale)


def load_questions() -> list[Question]:
    questions = csv.reader(QUESTION_PATH.open('r'))
    return [
        Question(
            statement=row[0],
            tags=row[1].split(', '),
        )
        for row in questions
    ]

# pprint(load_questions())


if __name__ == '__main__':
    cli()