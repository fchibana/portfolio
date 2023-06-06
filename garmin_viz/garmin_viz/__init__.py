import math
from datetime import time
from typing import List, Tuple

import gpxpy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from geopy import distance
from gpxpy.gpx import GPXTrackPoint
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import FuncFormatter
from timezonefinder import TimezoneFinder


def load_gpx(fname: str) -> dict:

    # open gpx activity
    with open(fname) as f:
        gpx = gpxpy.parse(f)

    track = gpx.tracks[0]
    res = {"name": track.name, "type": track.type, "points": track.segments[0].points}
    return res


def get_df(points: List[GPXTrackPoint]) -> pd.DataFrame:
    def parse_extension(point: GPXTrackPoint) -> tuple:
        """Extract the Garmin Track Point Extensions info"""

        # track point extension
        tpe = point.extensions[0]
        atemp = tpe[0].text
        hr = tpe[1].text
        cad = tpe[2].text

        return float(atemp), int(hr), int(cad)

    data = []
    for point_idx, point in enumerate(points):
        atemp, hr, cad = parse_extension(point)
        data.append(
            [
                point.latitude,
                point.longitude,
                point.elevation,
                point.time,
                atemp,
                hr,
                cad,
            ]
        )

    columns = ["latitude", "longitude", "elevation", "time", "atemp", "hr", "cad"]
    return pd.DataFrame(data, columns=columns)


def load_activity(fname: str) -> Tuple[str, pd.DataFrame]:

    raw_data = load_gpx(fname)
    activity_name = raw_data["name"]
    activity_data = get_df(raw_data["points"])

    return activity_name, activity_data


def extend_dataframe(df_in: pd.DataFrame) -> pd.DataFrame:
    """Extend DataFrame with additional data.

    Additional data:
    - Time interval between points [s]
    - 2D and 3D (cumulative) distance [m]
    - Distance interval between points [m]
    - Pace [s/m]
    - Cadence [spm]

    Args:
        df_in (pd.DataFrame): DataFrame with raw data

    Returns:
        pd.DataFrame: DataFrame with additional data
    """

    df_out = df_in.copy()
    distance_2d = [0.0]
    distance_3d = [0.0]

    for index in range(1, df_out.shape[0]):

        prev = df_out.iloc[index - 1]
        curr = df_out.iloc[index]

        # distance using geodesic method (2D)
        d_diff_2d = distance.distance(
            (prev.latitude, prev.longitude), (curr.latitude, curr.longitude)
        ).meters
        # difference in elevation
        h_diff = curr.elevation - prev.elevation
        #  3D Euclidean distance
        d_diff_3d = math.sqrt(d_diff_2d ** 2 + h_diff ** 2)

        # 2d accumulated distance
        d_2d = distance_2d[-1] + d_diff_2d
        distance_2d.append(d_2d)

        # 3d accumulated distance
        d_3d = distance_3d[-1] + d_diff_3d
        distance_3d.append(d_3d)

    # fix timezone
    tz_str = TimezoneFinder().timezone_at(
        lng=df_out["longitude"].iloc[0], lat=df_out["latitude"].iloc[0]
    )
    df_out["time"] = df_out["time"].dt.tz_convert(tz_str)

    df_out["dt"] = df_out["time"].diff() / np.timedelta64(1, "s")  # type: ignore
    df_out["distance_2d"] = distance_2d
    df_out["distance_3d"] = distance_3d

    # add point to point difference in distance
    df_out["ds_2d"] = df_out["distance_2d"].diff()

    # add pace dt/ds in seconds/meters
    df_out["pace"] = df_out["dt"] / df_out["ds_2d"]

    # add cadence in steps per min
    df_out["cadence"] = 2.0 * df_out["cad"]

    return df_out


def remove_stop_segments(df: pd.DataFrame) -> pd.DataFrame:

    # garmin's auto pause threshold: 12:30/km
    stop_pace_threshold = 12.5 * (60.0 / 1000.0)  # s/m

    # print("Rows to be ignored")
    # print(df[df["pace"] > stop_pace_threshold])
    mask = df["pace"] < stop_pace_threshold

    return df[mask].copy()  # type: ignore


def convert_pace_to_str(pace_in_min_per_km: float) -> str:
    t = time(minute=int(pace_in_min_per_km), second=int(pace_in_min_per_km % 1 * 60))
    # return f"{t.minute}:{t.second}"
    return t.strftime("%M:%S")


def bin_data(df_in: pd.DataFrame, step_km: float) -> pd.DataFrame:
    max_distance_m = df_in["distance_2d"].max()
    max_bin = int(max_distance_m + 1000)
    max_bin_km = max_bin / 1000
    labels = np.arange(start=0.0, stop=max_bin_km, step=step_km)
    bins = pd.Series(labels * 1000)
    labels = pd.Series(labels[1:])

    # get a tmp df
    tmp_df = df_in.copy()

    # start the split
    tmp_df["splits"] = pd.cut(
        df_in["distance_2d"], bins=bins, labels=labels
    )  # type:ignore

    data = []
    for split in labels:
        tmp_split = tmp_df[tmp_df["splits"] == split]

        split_avg_pace = tmp_split["pace"].mean() * (1000.0 / 60.0)  # min / km
        split_avg_cadence = tmp_split["cadence"].mean()  # spm
        split_avg_hr = tmp_split["hr"].mean()  # bpm
        split_avg_pace_str = convert_pace_to_str(split_avg_pace)

        data.append(
            [
                int(split),
                split_avg_pace,
                split_avg_pace_str,
                round(split_avg_cadence),
                round(split_avg_hr),
            ]
        )

    return pd.DataFrame(data, columns=["lap", "pace_float", "pace", "cadence", "hr"])


def pretty_pace_plot(lap: pd.Series, pace: pd.Series) -> Figure:
    def format_func(v, tick_number):
        value = 1 / v
        minutes = int(value)
        seconds = int(value % 1 * 60)
        if seconds == 0:
            seconds = "00"
        return f"{minutes}:{seconds}"

    fig, ax = plt.subplots()
    ax.plot(lap, 1.0 / pace)

    ax.grid(True)

    ax.set_ylabel("pace (/km)")
    ax.set_xlabel("lap (km)")

    ax.yaxis.set_major_formatter(FuncFormatter(format_func))
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(2.5))

    return fig
