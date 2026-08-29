import numpy as np
import soundfile as sf
import glob
import math
import matplotlib.pyplot as plt
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import utilities as ut

"""
Read in wav files and source meta data
Calculate the 'extra distances' for all signals
Plot the hydrophone location and bearingnn
Plot the source locations
Consider 2-D geometry only for localization
Input a source location and bearing for the array and calculate the parabola determined
    by the location and bearing of the binaural array
Plot the hyperbola and show the closest point and the distance to the hyperbola.
Update the screen when the selected source or the binaural bearing changes
"""



"""Main function to run hydrophone and hyperbola display"""
print("Hydrophone and hyperbola display")

"""Main function to run hydrophone and hyperbola display"""
coastline_filename = "OS_coastline_lat_lon.txt"
local_ref_lat_lon = [48.55841, -123.17327]  # this is the madrona tree
speed_of_sound = 1485  # m/s
# Initialize coordinate localizer
localizer = ut.HydrophoneLocalizer(local_ref_lat_lon, speed_of_sound)
# Read coastline
# coastline = get_coastline(coastline_filename)
NODE_DEFAULTS = {
    'andrews_bay': {'lat': 48.549562, 'lon': -123.167215, 'bearing': 324,
                     'hydrophone_separation': 1.25, 'speed_of_sound': 1495},
    'orcasound_lab': {'lat': 48.55841, 'lon': -123.17327, 'bearing': 152,
                        'hydrophone_separation': 1.2, 'speed_of_sound': 1442},
}


def load_node_data(hydrophone_node, localizer):
    """Load hydrophone geometry defaults and TDOA annotations for one node.

    Returns (annotations, hydrophones, bearing, hydrophone_separation, speed_of_sound).
    """
    defaults = NODE_DEFAULTS[hydrophone_node]
    bearing = defaults['bearing']
    hydrophone_separation = defaults['hydrophone_separation']
    speed_of_sound = defaults['speed_of_sound']

    # get x, y wrt Madrona Tree
    x, y = localizer.convert_to_local_xy(defaults['lat'], defaults['lon'])
    initial_hydrophone_positions = np.array([
        [x, y, 7],  # Hydrophone 1
        [x + hydrophone_separation * np.sin(np.radians(bearing)),
         y + hydrophone_separation * np.sin(np.radians(bearing)), 7]  # Hydrophone 2
    ])

    data_dir = f"wavs_{hydrophone_node}"
    annot_type = 'LB'  # annot_type = 'LO'  # for low freq Janus seq

    annotations = []
    #  return list of filenames with given file type
    annot_files = glob.glob(f"{data_dir}/*.txt")

    loc_file = "lightbulb_annotations.csv"
    i = 0
    for annot_file in annot_files:
        datetime = annot_file.split('/')[-1].split('.')[0]
        datetime = datetime.replace('_', '-')
        datetime = ut.move_to_start_of_ith_in_string(datetime.split('-'), '-', 2)

        ## get location data from csv file
        with open(loc_file) as f:
            for line in f:
                items = line.strip().split(',')
                the_date = items[2]
                the_time = items[3]
                this_datetime = the_date + '.' + the_time
                this_datetime = this_datetime.replace('.', '-')
                # print(this_datetime, datetime)
                if this_datetime == datetime and hydrophone_node == items[0]:
                    # print(datetime)  #  node,     date,     time,   latitude, longitude, depth, wav
                    annot = ut.annotation(localizer, items[0], items[1], items[2], items[3], items[4], items[5], items[6],
                                       items[9])

        with open(annot_file, 'r') as f:
            for line in f:
                items = line.strip().split()
                if items[2] == annot_type:
                    annot.wav_t1 = float(items[0])
                    annot.wav_t2 = float(items[1])
                    annot.annot_type = annot_type
                    # calculate TOAD
                    """Extract audio segments from WAV file based on start/stop times"""
                    with sf.SoundFile(annot.wav_file, 'r') as wav_f:
                        annot.sample_rate = wav_f.samplerate
                    start_frame = int(annot.wav_t1 * annot.sample_rate)
                    stop_frame = int(annot.wav_t2 * annot.sample_rate)
                    audio_segment, _ = sf.read(annot.wav_file, start=start_frame, stop=stop_frame, always_2d=True)
                    try:
                        annot.tdoa = ut.calculate_time_delay(audio_segment, annot.sample_rate)
                        # TDOA in seconds
                        print(
                            f"{annot.wav_file} Annotation {i}: type {annot.annot_type} TDOA = {annot.tdoa:.6f} seconds {annot.tdoa * annot.sample_rate:.3f} samples")
                    except Exception as e:
                        print(f"Error calculating TDOA for annotation {i}: {e}")
                        annot.tdoa = -99  # Default value
                    annotations.append(annot)
                    i += 1

    # class binaural_array:
    # def __init__(self, localizer, latitude, longitude, depth, hydrophone_separation, true_bearing_deg)
    hydrophones = ut.binaural_array(localizer, initial_hydrophone_positions, hydrophone_separation)
    print(f"ready for tkinter display ({hydrophone_node}, {len(annotations)} signals)")

    return annotations, hydrophones, bearing, hydrophone_separation, speed_of_sound


# Configuration
hydrophone_node = "orcasound_lab"
annotations, hydrophones, bearing, hydrophone_separation, speed_of_sound = load_node_data(hydrophone_node, localizer)

def update_plot(event=None):
    global annotations, hydrophones, localizer
    try:
        # Get input from every control and attempt to convert to float.
        # Don't throw an error if a box is cleared or has a lone minus sign.
        raw_values = {
            "signal": signal_entry.get(),
            "bearing": bearing_entry.get(),
            "separation": entry_separation.get(),
            "speed": entry_speed.get(),
        }
        if any(v in ("", "-") for v in raw_values.values()):
            return

        bearing_deg = float(raw_values["bearing"])
        separation = float(raw_values["separation"])
        speed_of_sound = float(raw_values["speed"])
        signal_index = int(float(raw_values["signal"]))

        # Recompute the hydrophone positions from the live bearing/separation
        hydrophones.set_fitted_positions(bearing_deg, separation)
        localizer.c = speed_of_sound

        hydros_x = hydrophones.fitted_hydrophone_positions[:, 0]
        hydros_y = hydrophones.fitted_hydrophone_positions[:, 1]
        h1_xy = hydrophones.fitted_hydrophone_positions[0][:2]
        h2_xy = hydrophones.fitted_hydrophone_positions[1][:2]

        # RMS distance between each signal and the closest point on its own
        # hyperbola -- how well the current bearing/separation/speed fit
        # explains all the observed TDOAs at once.
        rms_error = ut.rms_hyperbola_error(annotations, h1_xy, h2_xy, speed_of_sound)
        hydrophones.fitted_error = rms_error if rms_error is not None else float('nan')
        rms_value_label.config(text=f"{rms_error:.3f}" if rms_error is not None else "N/A")

        sources_x = [src.x for src in annotations]
        sources_y = [src.y for src in annotations]

        # Clear the old plot
        axes.clear()
        axes.scatter(hydros_x, hydros_y, c='blue', marker='o', s=80, label='Hydrophone Locations', alpha=0.7)
        axes.scatter(sources_x, sources_y, c='red', marker='o', s=80, label='Source Locations', alpha=0.7)

        # Label each source with its index (left of the dot) and its TDOA-derived
        # extra distance (tdoa * speed_of_sound)
        for index, src in enumerate(annotations):
            axes.annotate(str(index), (src.x, src.y),
                           textcoords='offset points', xytext=(-8, 0), fontsize=8,
                           color='black', ha='right', va='center')
            if src.tdoa == -99:  # error sentinel set when TDOA calculation failed
                continue
            extra_distance = src.tdoa * speed_of_sound
            axes.annotate(f'{extra_distance:.2f} m', (src.x, src.y),
                           textcoords='offset points', xytext=(6, 6), fontsize=8, color='darkred')

        selected = annotations[signal_index] if 0 <= signal_index < len(annotations) else None
        if selected is not None:
            axes.scatter([selected.x], [selected.y], c='gold', marker='*', s=250,
                         label=f'Selected Signal ({signal_index})', zorder=5)

        # TODO: this straight line through the array's bearing will be
        # replaced by the parabola/hyperbola for the selected signal.
        bearing_rad = math.radians(bearing_deg)
        direction = np.array([math.sin(bearing_rad), math.cos(bearing_rad)])
        line_pts = np.array([-15, 15])[:, None] * direction[None, :]
        axes.plot(line_pts[:, 0], line_pts[:, 1], label=f'Bearing = {bearing_deg}\N{DEGREE SIGN}',
                  color='crimson', lw=2)

        axes.axhline(0, color='black', lw=0.5)
        axes.axvline(0, color='black', lw=0.5)
        axes.grid(True)
        axes.set_aspect('equal')

        # Let matplotlib autoscale to the fixed elements above first, then use
        # that view box to size and clip the hyperbola branch -- cosh(t)/sinh(t)
        # don't map to plot distance in any simple way, so it's extended/clipped
        # against the actual view box instead of a fixed parameter range.
        fig.canvas.draw()
        xlim, ylim = axes.get_xlim(), axes.get_ylim()

        if selected is not None and selected.tdoa != -99:
            delta_r = selected.tdoa * speed_of_sound
            branch = ut.hyperbola_branch_points(h1_xy, h2_xy, delta_r, xlim=xlim, ylim=ylim)

            if branch is None:
                print(f"Signal {signal_index}: extra distance {delta_r:.2f} m exceeds "
                      f"the {separation:.2f} m hydrophone separation - no valid hyperbola")
            else:
                branch_pts, center, R, a, b = branch
                axes.plot(branch_pts[:, 0], branch_pts[:, 1], 'b-', linewidth=2,
                          label='TDOA Hyperbola')

                closest = ut.closest_point_on_branch((selected.x, selected.y), center, R, a, b)
                axes.plot([selected.x, closest[0]], [selected.y, closest[1]], 'k--', lw=1.5,
                          label='Closest Approach')
                axes.scatter([closest[0]], [closest[1]], c='blue', marker='x', s=80,
                             label='Closest Point', zorder=5)

        # Lock the view back to the fixed-element autoscale so the hyperbola
        # (and closest-point marker, if it lies outside the box) can't stretch it
        axes.set_xlim(xlim)
        axes.set_ylim(ylim)

        # Standard graph formatting
        axes.legend(loc='upper left')
        axes.set_title(f"{hydrophone_node}:Live Hyperbola Plotter")
        # Refresh the screen canvas
        canvas.draw()

    except ValueError:
        # Ignore invalid inputs while typing (like "2.")
        pass


def on_node_change(*_args):
    """Reload annotations/hydrophones for the newly selected node, reset the
    parameter fields to that node's defaults, and redraw."""
    global annotations, hydrophones, hydrophone_node

    hydrophone_node = node_var.get()
    annotations, hydrophones, bearing, hydrophone_separation, speed_of_sound = load_node_data(
        hydrophone_node, localizer)

    signal_entry.delete(0, tk.END)
    signal_entry.insert(0, "0")
    bearing_entry.delete(0, tk.END)
    bearing_entry.insert(0, f"{bearing}")
    entry_separation.delete(0, tk.END)
    entry_separation.insert(0, f"{hydrophone_separation}")
    entry_speed.delete(0, tk.END)
    entry_speed.insert(0, f"{speed_of_sound}")

    update_plot()


# 1. Setup the Tkinter Window
root = tk.Tk()
root.title("Real-Time Source/Hydrophone/Hyperbola Plotter")
root.geometry("1200x1200")

# 2. Embed the Matplotlib Figure
fig, axes = plt.subplots(figsize=(5, 5))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# 3. Create the Input Panel
controls = tk.Frame(root)
controls.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

tk.Label(controls, text="Node:", font=("Arial", 11)).pack(side=tk.LEFT, padx=5)
node_var = tk.StringVar(value=hydrophone_node)
node_menu = tk.OptionMenu(controls, node_var, *NODE_DEFAULTS.keys(), command=on_node_change)
node_menu.config(font=("Arial", 11))
node_menu.pack(side=tk.LEFT, padx=5)

tk.Label(controls, text="Signal:", font=("Arial", 11)).pack(side=tk.LEFT, padx=5)
signal_entry = tk.Entry(controls, font=("Arial", 11), width=10)
signal_entry.pack(side=tk.LEFT, padx=5)
signal_entry.insert(0, "0")  # Index into the annotations list

tk.Label(controls, text="Bearing (deg):", font=("Arial", 11)).pack(side=tk.LEFT, padx=5)
bearing_entry = tk.Entry(controls, font=("Arial", 11), width=10)
bearing_entry.pack(side=tk.LEFT, padx=5)
bearing_entry.insert(0, f"{bearing}")  # Initial bearing

tk.Label(controls, text="Hydrophone Separation (m):", font=("Arial", 11)).pack(side=tk.LEFT, padx=5)
entry_separation = tk.Entry(controls, font=("Arial", 11), width=10)
entry_separation.pack(side=tk.LEFT, padx=5)
entry_separation.insert(0, f"{hydrophone_separation}")

tk.Label(controls, text="Speed of Sound (m/s):", font=("Arial", 11)).pack(side=tk.LEFT, padx=5)
entry_speed = tk.Entry(controls, font=("Arial", 11), width=10)
entry_speed.pack(side=tk.LEFT, padx=5)
entry_speed.insert(0, f"{speed_of_sound}")

# Output-only control: RMS distance between each signal and the closest point
# on its own hyperbola, recomputed whenever a parameter above changes.
tk.Label(controls, text="RMS Error (m):", font=("Arial", 11)).pack(side=tk.LEFT, padx=5)
rms_value_label = tk.Label(controls, text="--", font=("Arial", 11), width=10, relief=tk.SUNKEN)
rms_value_label.pack(side=tk.LEFT, padx=5)

# 4. Bind the Return Key and Key Releases for immediate changes
for control in (signal_entry, bearing_entry, entry_separation, entry_speed):
    control.bind("<Return>", update_plot)  # Updates when pressing Enter
    control.bind("<KeyRelease>", update_plot)  # Updates on-the-fly while typing

# Draw initial plot
update_plot()

# 5. Start GUI loop
root.mainloop()
