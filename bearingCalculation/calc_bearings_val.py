import numpy as np
import soundfile as sf
import os
from scipy import signal
from matplotlib import pyplot as plt
from scipy.constants import speed_of_sound


##########################################################
##########################################################
def get_audio_segments(wav_filename: str, start_stop_times: list[tuple[float, float]]) -> tuple[list[np.ndarray], int]:
    """Extract audio segments from WAV file based on start/stop times"""
    with sf.SoundFile(wav_filename, 'r') as f:
        sample_rate = f.samplerate

    audio_segments = []
    for start_time, stop_time, _ in start_stop_times:
        start_frame = int(start_time * sample_rate)
        stop_frame = int(stop_time * sample_rate)

        segment, _ = sf.read(wav_filename, start=start_frame, stop=stop_frame, always_2d=True)
        audio_segments.append(segment)

    return audio_segments, sample_rate


def calculate_time_sample_delay(audio_data: np.ndarray, sample_rate: int) -> float:
    """Calculate time delay between stereo channels using cross-correlation"""
    if audio_data.shape[1] < 2:
        raise ValueError("Audio data must have at least 2 channels")

    left_channel = audio_data[:, 0]
    right_channel = audio_data[:, 1]
    plt.plot(left_channel, color='g')
    plt.plot(right_channel, color='r')
    plt.xlim([133200, 133400])
    plt.show()
    left_max = np.argmax(left_channel)
    right_max = np.argmax(right_channel)

    # Normalize signals to improve correlation
    left_channel = (left_channel - np.mean(left_channel)) / np.std(left_channel)
    right_channel = (right_channel - np.mean(right_channel)) / np.std(right_channel)

    correlation = signal.correlate(right_channel, left_channel, mode='full')
    delay_in_samples = np.argmax(correlation) - (len(left_channel) - 1)
    time_delay_seconds = delay_in_samples / sample_rate
    print(left_max, right_max, right_max - left_max, delay_in_samples, time_delay_seconds)
    return time_delay_seconds, delay_in_samples
##########################################################
##########################################################

data_dir = "../data"
label_filename = "HW_1_labels.txt"
wav_filename = "HW_1.wav"
speed_of_sound = 1485  # m/s
hydrophone_separation = 1.1  # Fixed separation in meters
array_theta_axis_re_N = 60   # deg

#  Read in annotations
the_labels = []
with open(os.path.join(data_dir, label_filename), "r") as f:
    for line in f:
        items = line.strip().split()
        if items and items[0] != '\\':
            secs_start = float(items[0])
            secs_end = float(items[1])
            lbl = items[2]
            the_labels.append([secs_start, secs_end, lbl])

print(the_labels)

audio_segments, sample_rate = get_audio_segments(os.path.join(data_dir,wav_filename), start_stop_times=the_labels)
print(f"sample_rate = {sample_rate}")

time_delays = []
sample_delays = []
for wav in audio_segments:
    delay_t, delay_samples = calculate_time_sample_delay(wav, sample_rate)
    time_delays.append(delay_t)
    sample_delays.append(delay_samples)
    print(f"time_delay {time_delays[-1]:0.3e}, sample_delay {sample_delays[-1]}")

# now calculate the needed angles
thetas = []
theta_sig_Haxis = []
theta_bearing_1 = []
theta_bearing_2 = []
for dt in time_delays:
    theta = np.degrees(np.acos(dt*speed_of_sound / hydrophone_separation))    # angle between L->R axis and sound path coming in
    thetas.append(theta)
    theta_sig_Haxis.append(theta)
    theta_bearing_1.append(array_theta_axis_re_N - theta)
    theta_bearing_2.append(array_theta_axis_re_N + theta)

for theta, b_1, b_2 in zip(thetas, theta_bearing_1, theta_bearing_2):
    print(f"{theta:0.1f} {b_1:0.1f}, {b_2:0.1f}")
