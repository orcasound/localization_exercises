from datetime import datetime
import numpy as np
import soundfile as sf
import os
import pyproj
#import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import minimize
from typing import Any
import time
import math
import random
from pathlib import Path


class annotation:
    def __init__(self, localizer, node, source_type, date, time, latitude, longitude, depth, wav_file):
        self.node = node
        self.source_type = source_type
        self.date = date
        self.time = time
        self.latitude = float(latitude)
        self.longitude = float(longitude)
        self.depth = float(depth)
        self.wav_file = wav_file
        self.sample_rate = 0
        self.x, self.y = localizer.convert_to_local_xy(latitude, longitude)
        self.z = float(depth)
        self.wav_t1 = 0
        self.wav_t2 = 0
        self.annot_type = ''


def calculate_bearing(hydrophone_positions: np.ndarray) -> tuple[float, float]:
    """
    Calculate bearingal angle in the x,y plane between hydrophones.

    Args:
        hydrophone_positions: Array of shape (2, 3) with [x, y, z] for each hydrophone

    Returns:
        tuple: (bearing_degrees, baseline_angle_degrees)
        - bearing_degrees: Angle from H1 to H2 in degrees (0° = East, 90° = North)
        - baseline_angle_degrees: Angle of the baseline vector from H1 to H2
    """
    h1, h2 = hydrophone_positions

    # Calculate the vector from H1 to H2
    dx = h2[0] - h1[0]
    dy = h2[1] - h1[1]

    # Calculate bearing angle (0° = North, 90° = East)
    bearing_rad = math.atan2(dx, dy)
    bearing_deg = math.degrees(bearing_rad)
    # Normalize to 0-360 degrees
    if bearing_deg < 0:
        bearing_deg += 360
    return bearing_deg, bearing_rad


def hyperbola_params(h1_xy: np.ndarray, h2_xy: np.ndarray, delta_r: float):
    """
    Core geometry of the TDOA hyperbola branch for a pair of hydrophones.

    delta_r is the path difference r1 - r2 ("extra distance" = tdoa * speed_of_sound).
    Its sign selects the branch nearer the hydrophone with the shorter path.

    Returns (center, R, a, b), or None if delta_r is non-physical (its
    magnitude can't exceed the hydrophone separation).
    """
    h1_xy = np.asarray(h1_xy, dtype=float)
    h2_xy = np.asarray(h2_xy, dtype=float)
    baseline = h2_xy - h1_xy
    distance = np.linalg.norm(baseline)

    c = distance / 2.0
    a = delta_r / 2.0
    if abs(a) >= c:
        return None

    b = math.sqrt(c ** 2 - a ** 2)
    center = (h1_xy + h2_xy) / 2.0
    cos_ang, sin_ang = baseline / distance
    R = np.array([[cos_ang, -sin_ang], [sin_ang, cos_ang]])
    return center, R, a, b


def hyperbola_branch_points(h1_xy: np.ndarray, h2_xy: np.ndarray, delta_r: float,
                             xlim: tuple[float, float] = None, ylim: tuple[float, float] = None,
                             n: int = 3000):
    """
    Points along the TDOA hyperbola branch for a pair of hydrophones, extended
    to reach and clipped to the box (xlim, ylim) -- typically the plot's
    current view -- rather than a fixed range of the cosh(t)/sinh(t) parameter.

    Returns (branch_pts, center, R, a, b), or None if delta_r is non-physical.
    """
    params = hyperbola_params(h1_xy, h2_xy, delta_r)
    if params is None:
        return None
    center, R, a, b = params

    if xlim is not None and ylim is not None:
        # cosh(t)/sinh(t) grow exponentially, so rather than guessing a fixed
        # parameter range, size t to the box's own extent: pick the farthest
        # corner from the branch center and solve x=scale*sinh(t) for t.
        x_min, x_max = xlim
        y_min, y_max = ylim
        corners = np.array([[x_min, y_min], [x_min, y_max], [x_max, y_min], [x_max, y_max]])
        reach = np.max(np.linalg.norm(corners - center, axis=1))
        scale = max(b, abs(a), 1e-6)
        t_max = math.asinh(reach / scale) + 1.0  # small margin so it overshoots the box
    else:
        t_max = 7.0

    t = np.linspace(-t_max, t_max, n)
    local = np.vstack([a * np.cosh(t), b * np.sinh(t)])
    branch_pts = (R @ local).T + center

    if xlim is not None and ylim is not None:
        mask = ((branch_pts[:, 0] >= x_min) & (branch_pts[:, 0] <= x_max) &
                (branch_pts[:, 1] >= y_min) & (branch_pts[:, 1] <= y_max))
        branch_pts = branch_pts[mask]

    return branch_pts, center, R, a, b


def closest_point_on_branch(source_xy: np.ndarray, center: np.ndarray, R: np.ndarray,
                             a: float, b: float, n: int = 4000) -> np.ndarray:
    """Closest point on the hyperbola branch x=a*cosh(t), y=b*sinh(t) (in the
    center/R local frame) to a source point, found by a dense scan over t."""
    local_source = R.T @ (np.asarray(source_xy, dtype=float) - center)

    # Size the search range to comfortably bracket the source's own distance
    # from the branch center -- the closest point won't lie much farther out.
    scale = max(b, abs(a), 1e-6)
    t_max = math.asinh(np.linalg.norm(local_source) / scale) + 2.0

    t = np.linspace(-t_max, t_max, n)
    x_local = a * np.cosh(t)
    y_local = b * np.sinh(t)
    dist_sq = (x_local - local_source[0]) ** 2 + (y_local - local_source[1]) ** 2

    best = np.argmin(dist_sq)
    closest_local = np.array([x_local[best], y_local[best]])
    return R @ closest_local + center


def rms_hyperbola_error(annotations: list, h1_xy: np.ndarray, h2_xy: np.ndarray,
                         speed_of_sound: float) -> float:
    """
    RMS distance between each signal's location and the closest point on its
    own TDOA hyperbola, for the current hydrophone geometry and speed of sound.
    A measure of how well the current bearing/separation fit explains all the
    observed TDOAs at once.

    Returns None if no annotation has a usable TDOA.
    """
    squared_errors = []
    for src in annotations:
        if src.tdoa == -99:  # error sentinel set when TDOA calculation failed
            continue
        params = hyperbola_params(h1_xy, h2_xy, src.tdoa * speed_of_sound)
        if params is None:
            continue
        center, R, a, b = params
        closest = closest_point_on_branch((src.x, src.y), center, R, a, b)
        squared_errors.append((src.x - closest[0]) ** 2 + (src.y - closest[1]) ** 2)

    if not squared_errors:
        return None
    return math.sqrt(sum(squared_errors) / len(squared_errors))


class binaural_array:
    def __init__(self, localizer, initial_hydrophone_positions, hydrophone_separation):  # [40, -1, 336, 1.6]
        self.initial_hydrophone_positions = initial_hydrophone_positions
        self.hydrophone_separation = hydrophone_separation
        self.latitude, self.longitude = convert_hydrophone_positions_to_lat_lon(localizer, initial_hydrophone_positions)
        self.fitted_hydrophone_positions = initial_hydrophone_positions  # placeholder
        self.fitted_error = 0
        self.fitted_bearing_deg = self.set_bearing_deg()

    def get_lat_lon(self):
        self.latitude, self.longitude = convert_hydrophone_positions_to_lat_lon(self.localizer,
                                                                                self.fitted_hydrophone_positions)

    def set_bearing_deg(self):
        # Calculate the vector from H1 to H2

        dx = self.fitted_hydrophone_positions[1][0] - self.fitted_hydrophone_positions[0][0]
        dy = self.fitted_hydrophone_positions[1][1] - self.fitted_hydrophone_positions[0][1]

        bearing_deg, bearing_rad = calculate_bearing(self.fitted_hydrophone_positions)
        # Calculate bearing angle (0° = North, 90° = East)
        return bearing_deg

    def set_fitted_positions(self, bearing_deg: float, separation: float):
        """Recompute H1/H2 positions for a live bearing/separation, keeping the
        same midpoint as the initial guess."""
        center = np.mean(self.initial_hydrophone_positions[:, :2], axis=0)
        z = self.initial_hydrophone_positions[0][2]

        bearing_rad = math.radians(bearing_deg)
        direction = np.array([math.sin(bearing_rad), math.cos(bearing_rad)])  # 0deg=N, 90deg=E
        offset = direction * (separation / 2)

        self.fitted_hydrophone_positions = np.array([
            [center[0] - offset[0], center[1] - offset[1], z],
            [center[0] + offset[0], center[1] + offset[1], z],
        ])
        self.hydrophone_separation = separation
        self.fitted_bearing_deg = self.set_bearing_deg()


class HydrophoneLocalizer:
    def __init__(self, local_ref_lat_lon: list[float], c: float = 1485):
        self.local_ref_lat_lon = local_ref_lat_lon
        self.c = c  # speed of sound in m/s
        self.transformer = self._setup_coordinate_transformer()
        self.ref_utm_x, self.ref_utm_y = self._get_reference_utm()
        self.loss_history = []

    def _setup_coordinate_transformer(self) -> pyproj.Transformer:
        """Set up coordinate transformer from lat/lon to UTM"""
        source_crs = pyproj.CRS("EPSG:4326")  # WGS84
        target_crs = pyproj.CRS("EPSG:32610")  # UTM Zone 10N
        return pyproj.Transformer.from_crs(source_crs, target_crs)

    def _get_reference_utm(self) -> tuple[float, float]:
        """Convert reference point to UTM coordinates"""
        lat, lon = self.local_ref_lat_lon
        return self.transformer.transform(lat, lon)

    def convert_to_local_xy(self, latitude: float, longitude: float) -> tuple[float, float]:
        """Convert latitude/longitude to local x,y coordinates in meters"""
        utm_x, utm_y = self.transformer.transform(latitude, longitude)
        return utm_x - self.ref_utm_x, utm_y - self.ref_utm_y


def convert_local_to_lat_lon(localizer: HydrophoneLocalizer, x: float, y: float) -> tuple[float, float]:
    """
    Convert local (x,y) coordinates back to (latitude, longitude).

    Args:
        localizer: HydrophoneLocalizer instance with reference coordinates
        x: Local x coordinate in meters
        y: Local y coordinate in meters

    Returns:
        tuple: (latitude, longitude) in degrees
    """
    # Convert back to UTM coordinates by adding the reference UTM offsets
    utm_x = x + localizer.ref_utm_x
    utm_y = y + localizer.ref_utm_y

    # Create reverse transformer (UTM to lat/lon)
    target_crs = pyproj.CRS("EPSG:4326")  # WGS84
    source_crs = pyproj.CRS("EPSG:32610")  # UTM Zone 10N
    reverse_transformer = pyproj.Transformer.from_crs(source_crs, target_crs)

    # Convert UTM to latitude/longitude
    latitude, longitude = reverse_transformer.transform(utm_x, utm_y)

    return latitude, longitude


def convert_hydrophone_positions_to_lat_lon(localizer: HydrophoneLocalizer,
                                            hydrophone_positions: np.ndarray) -> list[tuple[float, float]]:
    """
    Convert all hydrophone positions from local coordinates to (lat, lon).

    Args:
        localizer: HydrophoneLocalizer instance
        hydrophone_positions: Array of shape (N, 3) with [x, y, z] for each hydrophone

    Returns:
        list: List of (latitude, longitude) tuples for each hydrophone
    """
    positions_lat_lon = []
    for pos in hydrophone_positions:
        lat, lon = convert_local_to_lat_lon(localizer, pos[0], pos[1])
        positions_lat_lon.append((lat, lon))
    return positions_lat_lon


def read_annotations(filepath: str, wav_start_time: datetime, initial_index: int) -> list[list]:
    """Read annotation file and convert times to Unix seconds"""
    annotations = []
    with open(filepath, 'r') as f:
        for line in f:
            items = line.strip().split()
            if items and items[0] != '\\':
                secs_start = float(items[0])
                secs_end = float(items[1])
                # ID = int(items[2])
                ID = int(items[2].split('_')[-1]) + initial_index
                unix_start = wav_start_time.timestamp() + secs_start
                unix_end = wav_start_time.timestamp() + secs_end
                # Convert the timestamp to a datetime object
                dt_object = datetime.fromtimestamp(unix_start)
                formatted_dt = dt_object.strftime('%Y-%m-%d %H:%M:%S')
                print(ID, formatted_dt)
                annotations.append([unix_start, unix_end, ID])
    return annotations


def get_audio_segments(wav_filename: str, start_stop_times: list[tuple[float, float]]) -> tuple[list[np.ndarray], int]:
    """Extract audio segments from WAV file based on start/stop times"""
    with sf.SoundFile(wav_filename, 'r') as f:
        sample_rate = f.samplerate

    audio_segments = []
    for start_time, stop_time in start_stop_times:
        start_frame = int(start_time * sample_rate)
        stop_frame = int(stop_time * sample_rate)

        segment, _ = sf.read(wav_filename, start=start_frame, stop=stop_frame, always_2d=True)
        audio_segments.append(segment)

    return audio_segments, sample_rate


def calculate_time_delay(audio_data: np.ndarray, sample_rate: int) -> float:
    """Calculate time delay between stereo channels using cross-correlation"""
    if audio_data.shape[1] < 2:
        raise ValueError("Audio data must have at least 2 channels")

    left_channel = audio_data[:, 0]
    right_channel = audio_data[:, 1]

    # Normalize signals to improve correlation
    left_channel = (left_channel - np.mean(left_channel)) / np.std(left_channel)
    right_channel = (right_channel - np.mean(right_channel)) / np.std(right_channel)

    correlation = signal.correlate(left_channel, right_channel, mode='full')
    delay_in_samples = np.argmax(correlation) - (len(left_channel) - 1)
    time_delay_seconds = delay_in_samples / sample_rate

    return time_delay_seconds


def calculate_tdoas(annotations: list[list], wav_filename: str, wav_start_time: datetime) -> list[list]:
    """Calculate Time Difference of Arrival for each annotation"""
    # Convert annotation times to relative times within WAV file
    wav_start_stop = []
    for annotation in annotations:
        start_rel = annotation[0] - wav_start_time.timestamp()
        stop_rel = annotation[1] - wav_start_time.timestamp()
        wav_start_stop.append((start_rel, stop_rel))

    audio_segments, sample_rate = get_audio_segments(wav_filename, wav_start_stop)

    # Calculate TDOA for each segment
    for i, (segment, annotation) in enumerate(zip(audio_segments, annotations)):
        try:
            tdoa = calculate_time_delay(segment, sample_rate)
            annotation.append(tdoa)  # TDOA in seconds
            print(f"Annotation {i}: TDOA = {tdoa:.6f} seconds  {tdoa * annotation.sample_rate:.3f} samples")
        except Exception as e:
            print(f"Error calculating TDOA for annotation {i}: {e}")
            annotation.append(0.0)  # Default value

    return annotations


def read_source_locations(filepath: str, localizer: HydrophoneLocalizer) -> list[list]:
    """Read source location data and convert to local coordinates"""
    source_locations = []
    with open(filepath, 'r') as f:
        for line in f:
            if not line.strip():
                continue

            items = line.strip().split(',')
            if len(items) >= 6:
                try:
                    # Parse timestamp (assuming format with timezone)
                    time_str = items[2].strip()

                    # Handle timezone offset - improved parsing
                    if time_str[-6] == '+' or time_str[-6] == '-':
                        # String has timezone offset in format ±HH:MM at position -6
                        # Keep the string as is, fromisoformat can handle it
                        pass
                    elif ':' in time_str and ('+' in time_str or '-' in time_str):
                        # Handle other timezone formats if needed
                        pass

                    dt = datetime.fromisoformat(time_str)

                    longitude = float(items[3])
                    latitude = float(items[4])
                    depth = float(items[5])

                    x, y = localizer.convert_to_local_xy(latitude, longitude)
                    source_locations.append([dt.timestamp(), [x, y, depth]])
                    print(f"Source: t={dt}, lat={latitude}, lon={longitude}, local=({x:.2f}, {y:.2f}), depth {depth}")

                except (ValueError, IndexError) as e:
                    print(f"Skipping malformed line: {line.strip()} - Error: {e}")
                    continue

    return source_locations
def move_to_start_of_ith_in_string(my_list, sep, ith):
    ith_item = my_list[0]
    new_list = [my_list[1]]
    for i in range(2,len(my_list)):
        new_list.append(my_list[i])
        if i == ith:
            new_list.append(ith_item)
    new_string = f"{sep}".join(new_list)
    return new_string