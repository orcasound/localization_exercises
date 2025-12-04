import numpy as np
import soundfile as sf
import os
import glob
import random
from scipy import signal
from matplotlib import pyplot as plt
import pyproj
import pickle

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

class signal_obj:
    def __init__(self, call, x, y, z, localizer, time, speed_of_sound):
        self.call = call
        self.xyz = [x, y, z]
        self.localizer = localizer
        self.time = time
        self.speed_of_sound = speed_of_sound
        self.lat, self.lon = convert_local_to_lat_lon(localizer, x, y)


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



def plot_rotated_hyperbola(xy, D, min_fraction, max_fraction, theta_deg, lag, signal_xy):
    """
    Plots a hyperbola based on foci separation and angular orientation.

    Parameters:
    D (float): Separation distance between foci L and R.
    fraction (float): The vertex is at distance (fraction * D) from Focus R.
    theta_deg (float): Angle of the axis clockwise from +Y toward +X.
    """
    fraction = (min_fraction + max_fraction)/2.0

    # Rotation Matrix components
    # Standard basis: X is horizontal. Target basis: +Y is 0 deg, +X is 90 deg.
    # We map canonical x-axis to vector (sin(theta), cos(theta))
    # We map canonical y-axis to vector (cos(theta), -sin(theta))

    def rotate(x, y, th):
        X_new = x * np.sin(th) + y * np.cos(th)
        Y_new = x * np.cos(th) - y * np.sin(th)
        return X_new, Y_new
    # 5. Plotting
    plt.figure(figsize=(8, 8))
    i = 0
    for fraction in [min_fraction, max_fraction]:
        # 1. Define Standard Hyperbola Parameters
        c = D / 2.0  # Distance from center to focus
        dist_from_focus = fraction * D

        # The vertex position 'a' relative to center.
        # Distance from center to vertex = (Distance center to focus) - (Distance focus to vertex)
        a = abs(c - dist_from_focus)

        # Calculate semi-minor axis 'b' using c^2 = a^2 + b^2
        if a >= c:
            print("Error: The curve cannot pass further than the midpoint (fraction must be != 0.5)")
            return
        b = np.sqrt(c ** 2 - a ** 2)

        # 2. Generate Points in Canonical Frame (Standard Horizontal Hyperbola)
        # Range for parameter t (controls length of the curves drawn)
        t = np.linspace(-40.0, 40.0, 500)

        # Right Branch (canonical)
        x_can_R = a * np.cosh(t)
        y_can_R = b * np.sinh(t)

        # Left Branch (canonical)
        x_can_L = -a * np.cosh(t)
        y_can_L = b * np.sinh(t)

        # 3. Define Rotation Logic (Clockwise from +Y)
        theta_rad = np.radians(theta_deg)



        # Rotate both branches
        xr, yr = rotate(x_can_R, y_can_R, theta_rad)
        xl, yl = rotate(x_can_L, y_can_L, theta_rad)

        # 4. Calculate Foci Positions for Plotting
        # Focus R (Positive direction along axis)
        fx_R, fy_R = rotate(c, 0, theta_rad)
        # Focus L (Negative direction along axis)
        fx_L, fy_L = rotate(-c, 0, theta_rad)

        # shift origin to xy of binaural array
        xr += xy[0]
        xl += xy[0]
        fx_R += xy[0]
        fx_L += xy[0]
        yr += xy[1]
        yl += xy[1]
        fy_R += xy[1]
        fy_L += xy[1]


        # Plot Axis Line
        if i==0: plt.plot([fx_L, fx_R], [fy_L, fy_R], 'k--', alpha=0.3, label="Axis")

        # Plot Foci
        if i==0:
            plt.scatter([fx_R], [fy_R], color='red', zorder=5, label='Focus R')
            plt.scatter([fx_L], [fy_L], color='blue', zorder=5, label='Focus L')
            plt.scatter([signal_xy[0]], [signal_xy[1]], color='green', zorder=5, label='Signal')
        else:
            plt.scatter([fx_R], [fy_R], color='red', zorder=5)
            plt.scatter([fx_L], [fy_L], color='blue', zorder=5)
            plt.scatter([signal_xy[0]], [signal_xy[1]], color='green', zorder=5)
        # Determine which branch is the "Active" one based on user fraction
        # If fraction < 0.5, the vertex is closer to R, so R-branch is the one.
        if fraction < 0.5:
            if i==0:
                plt.plot(xr, yr, color='crimson', linewidth=2, label='Selected Branch')
            else:
                plt.plot(xr, yr, color='crimson', linewidth=2)
            plt.plot(xl, yl, color='gray', linestyle=':', alpha=0.5)  # Other branch
        else:
            if i ==0:
                plt.plot(xl, yl, color='crimson', linewidth=2, label='Selected Branch')
            else:
                plt.plot(xl, yl, color='crimson', linewidth=2)
            plt.plot(xr, yr, color='gray', linestyle=':', alpha=0.5)  # Other branch
        i += 1

    # Formatting
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.xlim([-50, 0])
    plt.ylim([-25, 25])
    #plt.axis('equal')

    plt.legend()
    plt.title(f"Hyperbola: D={D}, corr lags: {lag-1}->{lag+1}, theta={theta_deg}°")

    plt.show()







#################################################################################################
#################################################################################################

local_ref_lat_lon = [48.55841, -123.17327]
speed_of_sound = 1485  # m/s
hydrophone_separation = 1.6  # Fixed separation in meters
binaural_array = np.array([-20, 0, -10, 340], dtype=np.float64) #  x, y, z in meters and bearing in degrees of array with respect to local_ref_lat_lon = [48.55841, -123.17327]

signal_data_file = 'generated_sound.wav'
signal_label_file = 'generated_labels.txt'
pickle_file = "generatedd_signals.pkl"

# load signals for comparison with hyperbolas
# Use pickle.load(file) to read the object from a file
try:
    with open(pickle_file, 'rb') as file:
        signal_list = pickle.load(file)
    print("signal_list successfully loaded from file:")

except FileNotFoundError:
    print(f"Error: The file {pickle_file} was not found.")
except Exception as e:
    print(f"Error loading object: {e}")


# Initialize coordinate localizer
localizer = HydrophoneLocalizer(local_ref_lat_lon, speed_of_sound)

RL_data , sample_rate = sf.read(signal_data_file)

RL_calls = []
RL_TOADS = []
with open(signal_label_file) as f:
    for line in f:
        items = line.split('\t')
        RL_calls.append((float(items[0]), items[1].split('\n')[0]))


print(RL_calls)
binaural_loc = binaural_array[:3]
theta_deg = binaural_array[3]

window = 2  #  size of window to cover call
for call, signal in zip(RL_calls, signal_list):
    secs = call[0]
    ID = call[1]
    ID_signal = signal.ID
    t1 = int((secs - window/2) * sample_rate)
    t2 = int((secs + window/2) * sample_rate)
    L_signal = RL_data[t1:t2, 0]
    R_signal = RL_data[t1:t2, 1]
    # Cross-correlation to find the time delay in samples
    correlation = np.correlate(L_signal, R_signal, mode='full')  # with R, L order here, a positive delay implies R is the focus of a hyperbola, otherwise L
    delay_in_samples = np.argmax(correlation) - (len(L_signal) - 1)

    time_delay_sec = (delay_in_samples / sample_rate)
    distance_delay = time_delay_sec * speed_of_sound
    min_distance_delay_d_samples__1 = speed_of_sound / sample_rate  # delay_distance for a 1 sample change in the maximum of the correlation function

    print(secs, ID, delay_in_samples) #, time_delay_ms, correlation)
    fraction = (hydrophone_separation + distance_delay) / (2 * hydrophone_separation)
    max_fraction = (hydrophone_separation + distance_delay + min_distance_delay_d_samples__1) / (2 * hydrophone_separation)
    min_fraction = (hydrophone_separation + distance_delay - min_distance_delay_d_samples__1) / (
                2 * hydrophone_separation)
    print(f"{secs}, {ID}, {ID_signal}, {delay_in_samples}, {fraction:0.2f}, {distance_delay:0.2f}")  # , time_delay_ms, correlation)
    signal_xy = signal.xyz[:2]
    plot_rotated_hyperbola(binaural_loc, hydrophone_separation, min_fraction, max_fraction, theta_deg, delay_in_samples, signal_xy)
    print("")



