import numpy as np
from datetime import datetime
import os
import pyproj
import pandas as pd
import matplotlib.pyplot as plt
import soundfile as sf
from scipy import signal
from scipy.optimize import minimize
from typing import Any
import time
import math
import random

#####from olderCodes.main import label_filename, loc_data_filename


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

def get_start_time_from_filename(filename: str) -> datetime:
    """Extract start time from filename (format: month_day_year_hour_minute_second)"""
    basename = os.path.basename(filename)
    name_without_ext = os.path.splitext(basename)[0]
    items = name_without_ext.split('_')
    
    if len(items) < 6:
        raise ValueError(f"Filename format incorrect: {filename}")
    
    month, day, year, hour, minute, second = items[:6]
    return datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))

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
            print(f"Annotation {i}: TDOA = {tdoa:.6f} seconds")
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

def match_sources_to_annotations(source_locations: list[list], annotations: list[list], 
                               time_tolerance: float = 5) -> list[dict[str, Any]]:
    """Match source locations to annotations based on timing"""
    matches = []
    
    for i, annotation in enumerate(annotations):
        annotation_time = annotation[0]  # Start time
        best_match = None
        min_time_diff = float('inf')
        
        for source in source_locations:
            source_time = source[0]
            time_diff = abs(annotation_time - source_time)
            
            if time_diff < time_tolerance and time_diff < min_time_diff:
                min_time_diff = time_diff
                best_match = source
        
        if best_match is not None:
            matches.append({
                'annotation': annotation,
                'source': best_match,
                'time_diff': min_time_diff,
                'index': i
            })
            print(f"Matched annotation {i} (t={annotation_time}) with source (t={best_match[0]}, diff={min_time_diff:.2f}s)")
    
    return matches

def calculate_azimuth(hydrophone_positions: np.ndarray) -> tuple[float, float]:
    """
    Calculate azimuthal angle in the x,y plane between hydrophones.
    
    Args:
        hydrophone_positions: Array of shape (2, 3) with [x, y, z] for each hydrophone
        
    Returns:
        tuple: (azimuth_degrees, baseline_angle_degrees)
        - azimuth_degrees: Angle from H1 to H2 in degrees (0° = East, 90° = North)
        - baseline_angle_degrees: Angle of the baseline vector from H1 to H2
    """
    h1, h2 = hydrophone_positions
    
    # Calculate the vector from H1 to H2
    dx = h2[0] - h1[0]
    dy = h2[1] - h1[1]
    
    # Calculate azimuth angle (0° = East, 90° = North)
    azimuth_rad = math.atan2(dy, dx)
    azimuth_deg = math.degrees(azimuth_rad)
    
    # Normalize to 0-360 degrees
    if azimuth_deg < 0:
        azimuth_deg += 360
    
    # Calculate baseline orientation (angle of the line between hydrophones)
    baseline_angle_deg = azimuth_deg
    
    return azimuth_deg, baseline_angle_deg

def loss_function_fixed_separation(params: np.ndarray, matches: list[dict], c: float = 1485, 
                                 hydrophone_separation: float = 1.7, separation_weight: float = 100.0) -> float:
    """
    Loss function with fixed separation between hydrophones and same depth constraint.
    
    Args:
        params: [x1, y1, theta, z] where:
            - (x1, y1, z) is position of first hydrophone
            - theta is the azimuth angle (radians) from H1 to H2
        matches: List of matched source-annotation pairs
        c: Speed of sound
        hydrophone_separation: Fixed distance between hydrophones (1.7m)
        separation_weight: Weight for separation constraint penalty
    """
    x1, y1, theta, z = params
    
    # Calculate second hydrophone position based on fixed separation and angle
    x2 = x1 + hydrophone_separation * math.cos(theta)
    y2 = y1 + hydrophone_separation * math.sin(theta)
    
    hydrophone_positions = np.array([
        [x1, y1, z],  # H1
        [x2, y2, z]   # H2
    ])
    
    # Calculate TDOA prediction error
    data_error = 0.0
    for match in matches:
        source_pos = np.array(match['source'][1])
        measured_tdoa = match['annotation'][3]
        
        # Calculate distances
        dist1 = np.linalg.norm(hydrophone_positions[0] - source_pos)
        dist2 = np.linalg.norm(hydrophone_positions[1] - source_pos)
        
        predicted_tdoa = (dist1 - dist2) / c
        error = measured_tdoa - predicted_tdoa
        data_error += error ** 2
    
    data_error /= len(matches)
    
    # Calculate actual separation (should be very close to 1.7m due to our parameterization)
    actual_separation = np.linalg.norm(hydrophone_positions[0] - hydrophone_positions[1])
    separation_error = (actual_separation - hydrophone_separation) ** 2
    
    # Add small regularization for depth
    depth_error = 0.0
    if abs(z) > 20:  # deeper than 20m
        depth_error = (abs(z) - 20) ** 2
    
    total_loss = data_error + separation_weight * separation_error + 0.1 * depth_error
    
    return total_loss

def grid_search(initial_params_scaled, matches, c, hydrophone_separation):
    # params are ([x1, y1, theta_initial, z_initial])
    min_loss = loss_function_fixed_separation(initial_params_scaled, matches, c, hydrophone_separation)
    best_params = [0, 0, 0, 0]
    param_range = [ [-70, -30, 0, 2], [-10, 0, 2*np.pi, 15] ]
    for p0 in np.arange(param_range[0][0], param_range[1][0], 5):
        for p1 in np.arange(param_range[0][1], param_range[1][1], 5):
            for p2 in np.arange(param_range[0][2], param_range[1][2], 0.25):
                for p3 in np.arange(param_range[0][3], param_range[1][3], 1):
                    this_loss = loss_function_fixed_separation([p0, p1, p2, p3], matches, c, hydrophone_separation)
                    if this_loss < min_loss:
                        min_loss = this_loss
                        best_params = [p0, p1, p2, p3]
    return min_loss, best_params


def localize_hydrophones_fixed_separation(initial_positions: np.ndarray, matches: list[dict], 
                                        c: float = 1485, hydrophone_separation: float = 1.7,
                                        method: str = 'CG', max_iter: int = 2000) -> tuple[np.ndarray, float, list[float]]:
    """
    Optimize hydrophone positions with fixed separation and same depth constraints.
            many methods: Nelder-Mead  Powell  CG  BFGS  Newton-CG  L-BFGS-B TNC ....
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
    Args:
        initial_positions: Array of shape (2, 3) with initial [x, y, z] for each hydrophone
        matches: List of matched source-annotation pairs
        c: Speed of sound
        hydrophone_separation: Fixed distance between hydrophones (1.7m)
        method: Optimization method
        max_iter: Maximum iterations
        
    Returns:
        tuple: (optimized_positions, final_loss, loss_history)
    """
    # Calculate initial parameters from initial positions
    x1, y1, z1 = initial_positions[0]
    x2, y2, z2 = initial_positions[1]
    
    # Calculate initial angle from H1 to H2
    dx = x2 - x1
    dy = y2 - y1
    theta_initial = math.atan2(dy, dx)
    
    # Use average depth as initial z
    z_initial = (z1 + z2) / 2
    
    initial_params = np.array([x1, y1, theta_initial, z_initial])
    
    loss_history = []
    
    def wrapped_loss(params):
        loss_val = loss_function_fixed_separation(params, matches, c, hydrophone_separation)
        loss_history.append(loss_val)
        if len(loss_history) % 25 == 0 or len(loss_history) == 1:
            print(f"Iteration {len(loss_history)}: loss = {loss_val:.6e}")
        return loss_val
    
    # Scale parameters for better convergence
    scale_factor = 1.0
    initial_params_scaled = [initial_params[0] * scale_factor, initial_params[1] * scale_factor, initial_params[2],  initial_params[3] * scale_factor]
    
    def scaled_loss(params):
        return wrapped_loss(params / scale_factor)
    
    # Optimization options
    options = {
        'gtol': 1e-11,
        'maxiter': max_iter,
        'disp': True
    }
    #  Do grid search for good starting location
    lowest_loss, starting_position = grid_search(initial_params_scaled, matches, c, hydrophone_separation)
    # Calculate initial parameters from initial positions
    x1 = starting_position[0]
    y1 = starting_position[1]
    theta_initial = starting_position[2]
    z_initial = starting_position[3]
    new_initial_hydrophone_positions = [ [x1, y1, z_initial], [x1+hydrophone_separation*math.cos(theta_initial), y1+hydrophone_separation*math.sin(theta_initial), z_initial] ]

    initial_params = np.array([x1, y1, theta_initial, z_initial])
    print(f"\nStarting optimization with {len(matches)} matches (fixed {hydrophone_separation}m separation)...")
    print(f"Initial parameters: H1=({x1:.2f}, {y1:.2f}), theta={math.degrees(theta_initial):.1f}°, z={z_initial:.2f}m")
    print(f"Initial loss function value: {lowest_loss:.6e}")
    start_time = time.time()
    
    result = minimize(
        scaled_loss,
        initial_params,  # these are from grid search of scaled_initial_params
        method=method,
        options=options
    )
    
    end_time = time.time()
    
    # Convert back to original scale and reconstruct positions
    optimized_params = [result.x[0] / scale_factor, result.x[1] / scale_factor, result.x[2], result.x[3] / scale_factor]
    x1_opt, y1_opt, theta_opt, z_opt = optimized_params
    
    # Calculate second hydrophone position
    x2_opt = x1_opt + hydrophone_separation * math.cos(theta_opt)
    y2_opt = y1_opt + hydrophone_separation * math.sin(theta_opt)
    
    optimized_positions = np.array([
        [x1_opt, y1_opt, z_opt],
        [x2_opt, y2_opt, z_opt]
    ])
    
    print(f"\nOptimization completed in {end_time - start_time:.2f} seconds")
    print(f"Success: {result.success}")
    print(f"Message: {result.message}")
    print(f"Final loss: {result.fun:.6e}")
    print(f"Number of iterations: {result.nit}")
    print(f"Number of function evaluations: {result.nfev}")


    return new_initial_hydrophone_positions, optimized_positions, result.fun, loss_history

def plot_results(source_locations: list[list], hydrophone_positions: np.ndarray, 
                matches: list[dict], coastline: list[float, float], loss_history: list[float] = None):
    """Plot source locations, estimated hydrophone positions, and loss history"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Spatial configuration
    source_x = [src[1][0] for src in source_locations]
    source_y = [src[1][1] for src in source_locations]
    ax1.scatter(source_x, source_y, c='blue', marker='o', s=80, label='Source Locations', alpha=0.7)
    
    # Plot matched sources
    matched_source_indices = [match['index'] for match in matches]
    matched_x = [source_x[i] for i in matched_source_indices if i < len(source_x)]
    matched_y = [source_y[i] for i in matched_source_indices if i < len(source_y)]
    ax1.scatter(matched_x, matched_y, c='green', marker='*', s=5, label='Matched Sources')

    # Plot hydrophone positions
    hydro_x = hydrophone_positions[:, 0]
    hydro_y = hydrophone_positions[:, 1]
    ax1.scatter(hydro_x, hydro_y, c='red', marker='s', s=100, label='Hydrophones')
    
    # Add labels and connections
    for i, (x, y) in enumerate(zip(hydro_x, hydro_y)):
        ax1.annotate(f'H{i+1}', (x, y), xytext=(5, 5), textcoords='offset points', 
                    fontweight='bold', fontsize=12)
    
    # Draw baseline and show azimuth
    ax1.plot(hydro_x, hydro_y, 'r--', alpha=0.5, label='Baseline')
    
    # Calculate and display azimuth
    azimuth_deg, baseline_angle_deg = calculate_azimuth(hydrophone_positions)
    mid_x = (hydro_x[0] + hydro_x[1]) / 2
    mid_y = (hydro_y[0] + hydro_y[1]) / 2
    ax1.text(mid_x, mid_y, f'Azimuth: {azimuth_deg:.1f}°', 
             fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.set_title('Hydrophone Localization Results (Local Coordinates)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Plot 2: Loss history
    if loss_history:
        ax2.semilogy(loss_history)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Loss (log scale)')
        ax2.set_title('Optimization Convergence')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_lat_lon_results(source_locations: list[list], hydrophone_positions: np.ndarray,
                         matches: list[dict], localizer: HydrophoneLocalizer, coastline: list[float, float],
                         loss_history: list[float] = None):
    """
    Plot source locations and hydrophone positions in latitude/longitude coordinates.
    Displays only decimal parts on axes with integer values in corners.

    Args:
        source_locations: List of source locations in local coordinates
        hydrophone_positions: Array of hydrophone positions in local coordinates
        matches: List of matched source-annotation pairs
        localizer: HydrophoneLocalizer instance for coordinate conversion
        loss_history: Optional list of loss values for convergence plot
    """
    # Convert all positions to latitude/longitude
    source_lat_lon = []
    for source in source_locations:
        x, y, z = source[1]
        lat, lon = convert_local_to_lat_lon(localizer, x, y)
        source_lat_lon.append((lat, lon))

    hydrophone_lat_lon = convert_hydrophone_positions_to_lat_lon(localizer, hydrophone_positions)

    # Extract coordinates for plotting
    source_lats = [lat for lat, lon in source_lat_lon]
    source_lons = [lon for lat, lon in source_lat_lon]

    hydro_lats = [lat for lat, lon in hydrophone_lat_lon]
    hydro_lons = [lon for lat, lon in hydrophone_lat_lon]

    # Calculate geographic azimuth (using small-angle approximation)
    local_azimuth_deg, _ = calculate_azimuth(hydrophone_positions)

    # Create figure with subplots
    if loss_history:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 8))
        ax2 = None

    # Calculate integer and fractional parts for display
    # Use the reference point as the base integer values
    ref_lat_int = int(localizer.local_ref_lat_lon[0])
    ref_lon_int = int(localizer.local_ref_lat_lon[1])

    # Convert all coordinates to fractional parts relative to reference integer
    source_lats_frac = [lat - ref_lat_int for lat in source_lats]
    source_lons_frac = [lon - ref_lon_int for lon in source_lons]
    hydro_lats_frac = [lat - ref_lat_int for lat in hydro_lats]
    hydro_lons_frac = [lon - ref_lon_int for lon in hydro_lons]

    shore_lats_frac = [lat - ref_lat_int for lat in coastline[:,0]]
    shore_lons_frac = [lon - ref_lon_int for lon in coastline[:,1]]
    shore_ref_lats_frac = localizer.local_ref_lat_lon[0] - ref_lat_int
    shore_ref_lons_frac = localizer.local_ref_lat_lon[1] - ref_lon_int

    # Plot 1: Latitude/Longitude configuration (fractional parts only)
    ax1.scatter(source_lons_frac, source_lats_frac, c='blue', marker='o', s=80,
                label='Source Locations', alpha=0.7)

    ax1.plot(shore_lons_frac, shore_lats_frac, label='Shoreline')
    ax1.scatter(shore_ref_lons_frac, shore_ref_lats_frac, c='red', marker='s', s=50)
    xtext = random.uniform(1, 5)
    ytext = random.uniform(1, 5)
    ax1.annotate(f'OS', (shore_ref_lons_frac, shore_ref_lats_frac), xytext=(xtext, ytext),
                 textcoords='offset points', fontweight='bold', fontsize=12)

    # Plot matched sources
    matched_source_indices = [match['index'] for match in matches]
    matched_lons_frac = [source_lons_frac[i] for i in matched_source_indices if i < len(source_lons_frac)]
    matched_lats_frac = [source_lats_frac[i] for i in matched_source_indices if i < len(source_lats_frac)]
    ax1.scatter(matched_lons_frac, matched_lats_frac, c='green', marker='*', s=50,
                label='Matched Sources')
    xtext = random.uniform(1, 5)
    ytext = random.uniform(1, 5)
    # Add labels for sources
    for i, (lat_frac, lon_frac) in enumerate(zip(matched_lons_frac, matched_lats_frac)):
        ax1.annotate(f'{i + 1}', (lat_frac, lon_frac), xytext=(xtext, ytext),
                     textcoords='offset points', fontweight='bold', fontsize=7)
    # Plot hydrophone positions
    ax1.scatter(hydro_lons_frac, hydro_lats_frac, c='red', marker='s', s=50,
                label='Hydrophones')
    # Add labels for hydrophones
    for i, (lat_frac, lon_frac) in enumerate(zip(hydro_lats_frac, hydro_lons_frac)):
        ax1.annotate(f'H{i + 1}', (lon_frac, lat_frac), xytext=(5, 5),
                     textcoords='offset points', fontweight='bold', fontsize=7)

    # Draw baseline between hydrophones
    ax1.plot(hydro_lons_frac, hydro_lats_frac, 'r--', alpha=0.5, label='Baseline')

    # Calculate data ranges for positioning
    lat_min_frac = min(source_lats_frac + hydro_lats_frac)
    lat_max_frac = max(source_lats_frac + hydro_lats_frac)
    lon_min_frac = min(source_lons_frac + hydro_lons_frac)
    lon_max_frac = max(source_lons_frac + hydro_lons_frac)
    lat_range = lat_max_frac - lat_min_frac
    lon_range = lon_max_frac - lon_min_frac

    # OPTION 1: Place azimuth text above the plot area using axis coordinates
    # This places it consistently in the upper part of the plot regardless of data
    ax1.text(0.5, 0.65, f'Azimuth: {local_azimuth_deg:.1f}°',
             transform=ax1.transAxes, fontsize=12,
             horizontalalignment='center', verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))

    # OPTION 2: Alternative placement - above the highest point in the data
    # Uncomment below if you prefer this approach instead
    # azimuth_y_pos = lat_max_frac + 0.1 * lat_range  # 10% above highest point
    # azimuth_x_pos = (lon_min_frac + lon_max_frac) / 2  # center horizontally
    # ax1.text(azimuth_x_pos, azimuth_y_pos, f'Azimuth: {local_azimuth_deg:.1f}°',
    #          fontsize=12, horizontalalignment='center', verticalalignment='bottom',
    #          bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))

    # OPTION 3: Place in a corner (e.g., upper right)
    # Uncomment below if you prefer corner placement
    # ax1.text(0.98, 0.95, f'Azimuth: {local_azimuth_deg:.1f}°',
    #          transform=ax1.transAxes, fontsize=12,
    #          horizontalalignment='right', verticalalignment='top',
    #          bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))

    # Set axis labels to show these are fractional parts
    ax1.set_xlabel(f'Longitude - {ref_lon_int}° (decimal part)')
    ax1.set_ylabel(f'Latitude - {ref_lat_int}° (decimal part)')
    ax1.set_title('Hydrophone Localization Results (Latitude/Longitude - Decimal Parts)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Set equal aspect ratio
    ax1.set_aspect('equal')

    # Add integer value annotations in corners
    # Upper right corner
    ax1.text(0.98, 0.98, f'Latitude: {ref_lat_int}° + decimal\nLongitude: {ref_lon_int}° + decimal',
             transform=ax1.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))

    # Lower right corner - show actual range of decimal values
    ax1.text(0.98, 0.02,
             f'Lat range: {lat_min_frac:.5f} to {lat_max_frac:.5f}\nLon range: {lon_min_frac:.5f} to {lon_max_frac:.5f}',
             transform=ax1.transAxes, fontsize=9, verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))

    # Plot 2: Loss history (if provided)
    if ax2 is not None and loss_history:
        ax2.semilogy(loss_history)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Loss (log scale)')
        ax2.set_title('Optimization Convergence')
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # [Rest of the function remains the same...]
    # Print coordinate information
    print("\n=== Geographic Coordinates ===")
    print("Reference point (Madrona tree):")
    print(f"  Full: Latitude = {localizer.local_ref_lat_lon[0]:.8f}, Longitude = {localizer.local_ref_lat_lon[1]:.8f}")
    print(f"  Integer part: Latitude = {ref_lat_int}°, Longitude = {ref_lon_int}°")
    print(
        f"  Decimal part: Latitude = {localizer.local_ref_lat_lon[0] - ref_lat_int:.8f}, Longitude = {localizer.local_ref_lat_lon[1] - ref_lon_int:.8f}")

    print("\nHydrophone positions (full coordinates):")
    for i, (lat, lon) in enumerate(hydrophone_lat_lon):
        lat_int = int(lat)
        lon_int = int(lon)
        lat_frac = lat - lat_int
        lon_frac = lon - lon_int
        print(f"  H{i + 1}: Latitude = {lat_int}° + {lat_frac:.8f}, Longitude = {lon_int}° + {lon_frac:.8f}")
        print(f"        Full: {lat:.8f}, {lon:.8f}")

    print("\nSource positions (matched, full coordinates):")
    matched_lats = [source_lats[i] for i in matched_source_indices if i < len(source_lats)]
    matched_lons = [source_lons[i] for i in matched_source_indices if i < len(source_lons)]
    for i, (lat, lon) in enumerate(zip(matched_lats, matched_lons)):
        lat_int = int(lat)
        lon_int = int(lon)
        lat_frac = lat - lat_int
        lon_frac = lon - lon_int
        print(f"  Source {i}: Latitude = {lat_int}° + {lat_frac:.8f}, Longitude = {lon_int}° + {lon_frac:.8f}")
        print(f"             Full: {lat:.8f}, {lon:.8f}")

    return local_azimuth_deg


def analyze_tdoas(matches: list[dict], optimized_positions: np.ndarray, c: float = 1485):
    """Analyze TDOA predictions vs measurements"""
    print("\n=== TDOA Analysis ===")
    
    tdoa_errors = []
    print("Source | Measured TDOA | Predicted TDOA | Error")
    print("-" * 50)
    
    for i, match in enumerate(matches):
        source_pos = np.array(match['source'][1])
        measured_tdoa = match['annotation'][3]
        
        predicted_tdoa = (np.linalg.norm(optimized_positions[0] - source_pos) - 
                         np.linalg.norm(optimized_positions[1] - source_pos)) / c
        
        error = measured_tdoa - predicted_tdoa
        tdoa_errors.append(error)
        
        print(f"{i:6} | {measured_tdoa:13.6f} | {predicted_tdoa:13.6f} | {error:8.6f}")
    
    print(f"\nTDOA Error Statistics:")
    print(f"  Mean error: {np.mean(tdoa_errors):.6f} s")
    print(f"  Std error: {np.std(tdoa_errors):.6f} s")
    print(f"  Max error: {np.max(np.abs(tdoa_errors)):.6f} s")
    print(f"  RMS error: {np.sqrt(np.mean(np.array(tdoa_errors)**2)):.6f} s")

def get_coastline(coast_file):
    coast_line = []
    with open(coast_file) as f:
        for line in f:
            items = line.strip().split(',')
            coast_line.append([float(items[0]), float(items[1])])
    return np.array(coast_line)

def main():
    """Main function to run hydrophone localization"""
    # Configuration
    data_dir = "localizeHydrophoneData"
    label_filename = []
    wav_filename = []
    loc_data_filename = []

    label_filename.append("10_03_2025_11_28_45_fromTSsegments_LABELS.txt")
    wav_filename.append("10_03_2025_11_28_45_OS_SirensFromTSsegments.wav")
    loc_data_filename.append("1759516064_locData_1.txt")

    label_filename.append("10_15_2025_08_59_47_sirens_LABELS.txt")
    wav_filename.append("10_15_2025_08_59_47_sirens.wav")
    loc_data_filename.append("1760543949_locData.txt")


    coastline_filename = "OS_coastline_lat_lon.txt"
    
    local_ref_lat_lon = [48.55841, -123.17327]
    speed_of_sound = 1485  # m/s
    hydrophone_separation = 1.6  # Fixed separation in meters
    
    try:
        # Initialize coordinate localizer
        localizer = HydrophoneLocalizer(local_ref_lat_lon, speed_of_sound)
        # Read coastline
        coastline = get_coastline(os.path.join(data_dir, coastline_filename))
        matches = []
        annotations = []
        source_locations = []
        annotation_index_start = 0
        for label, wav, loc in zip(label_filename, wav_filename, loc_data_filename):
            # Read and process annotations
            print("Reading annotations...")
            wav_start_time = get_start_time_from_filename(label)
            annot_0 = (read_annotations(
                os.path.join(data_dir, label),
                wav_start_time,
                annotation_index_start
            ) )
            print(f"Found {len(annot_0)} annotations")

            annotation_index_start += len(annot_0)
            # Calculate TDOAs
            print("\nCalculating TDOAs...")
            annot = (calculate_tdoas(
                annot_0,
                os.path.join(data_dir, wav),
                wav_start_time
            ) )
            annotations += annot_0
            # Read source locations
            print("\nReading source locations...")
            locs = (read_source_locations(
                os.path.join(data_dir, loc),
                localizer
            ) )
            print(f"Found {len(locs)} source locations")
            source_locations += locs
            # Match sources to annotations
            print("\nMatching sources to annotations...")
            match = (match_sources_to_annotations(locs, annot))
            matches += match
            print(f"\nFound {len(match)} matched events")


        if len(matches) < 3:
                print("Error: Need at least 3 matched events for reliable localization")
                return
        
        # Print TDOA statistics
        tdoas = [match['annotation'][3] for match in matches]
        print(f"\nTDOA Statistics:")
        print(f"  Min: {min(tdoas):.6f} s")
        print(f"  Max: {max(tdoas):.6f} s")
        print(f"  Mean: {np.mean(tdoas):.6f} s")
        print(f"  Std: {np.std(tdoas):.6f} s")
        
        # Set initial hydrophone positions (educated guess)
        initial_hydrophone_positions = np.array([
            [-40, -1, 7],   # Hydrophone 1
            [-42, 0, 7]    # Hydrophone 2
        ])

        # Run optimization with fixed separation constraint
        print("\n" + "="*50)
        initial_hydrophone_positions, optimized_positions, final_loss, loss_history = localize_hydrophones_fixed_separation(
            initial_hydrophone_positions, 
            matches, 
            speed_of_sound,
            hydrophone_separation
        )
        
        # Calculate azimuth
        azimuth_deg, baseline_angle_deg = calculate_azimuth(optimized_positions)
        
        # Verify separation
        actual_separation = np.linalg.norm(optimized_positions[0] - optimized_positions[1])
        
        # Print results
        print("\n" + "="*50)
        print("FINAL RESULTS:")
        print("="*50)
        print(f"Initial hydrophone positions:")
        for i, pos in enumerate(initial_hydrophone_positions):
            print(f"  H{i+1}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
        
        print(f"\nOptimized hydrophone positions (fixed {hydrophone_separation}m separation):")
        for i, pos in enumerate(optimized_positions):
            print(f"  H{i+1}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.1f}]")

        # Convert hydrophone positions back to latitude/longitude
        print(f"\nHydrophone Positions in Latitude/Longitude:")
        for i, pos in enumerate(optimized_positions):
            lat, lon = convert_local_to_lat_lon(localizer, pos[0], pos[1])
            print(f"  H{i + 1}: Latitude = {lat:.8f}, Longitude = {lon:.8f}")
        for i, pos in enumerate(optimized_positions):
            lat, lon = convert_local_to_lat_lon(localizer, pos[0], pos[1])
            latdeg = (lat -int(lat))*60
            londeg = (lon -int(lon))*60
            lat = int(lat)
            lon = int(lon)
            print(f"  H{i + 1}: Latitude = {lat} {latdeg:.4f}', Longitude = {lon} {londeg:.4f}'")

        print(f"\nActual separation: {actual_separation:.6f} m (target: {hydrophone_separation} m)")
        print(f"Separation error: {abs(actual_separation - hydrophone_separation):.6f} m")
        print(f"Common depth: {optimized_positions[0, 2]:.3f} m")
        print(f"Azimuth (H1→H2): {azimuth_deg:.1f}°")
        print(f"  - {azimuth_deg:.1f}° from East (0° = East, 90° = North)")
        print(f"Baseline orientation: {baseline_angle_deg:.1f}°")
        print(f"Final loss: {final_loss:.6e}")
        
        # Analyze results
        analyze_tdoas(matches, optimized_positions, speed_of_sound)
        
        # Plot results in both coordinate systems
        print("\n" + "="*50)
        print("PLOTTING RESULTS")
        print("="*50)
        
        # Plot in local coordinates (meters)
        plot_results(source_locations, optimized_positions, matches, coastline, loss_history)
        
        # Plot in geographic coordinates (latitude/longitude)
        plot_lat_lon_results(source_locations, optimized_positions, matches, 
                           localizer, coastline, None)   #loss_history)
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()