"""Step 1: normalise the clip to 1080x1920, extract audio, find racket sounds."""
import json
import subprocess
import wave
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt

W, H = 1080, 1920
SR = 44100


def run(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def probe_fps(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=avg_frame_rate", "-of", "json", str(path)],
        check=True, capture_output=True, text=True).stdout
    num, den = json.loads(out)["streams"][0]["avg_frame_rate"].split("/")
    return float(num) / float(den)


def to_prepared(clip: Path, pts):
    """Map pixel positions in the original clip to the 1080x1920 prepared frame."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height:stream_side_data=rotation", "-of", "json", str(clip)],
        check=True, capture_output=True, text=True).stdout
    st = json.loads(out)["streams"][0]
    w, h = st["width"], st["height"]
    if any(abs(d.get("rotation", 0)) == 90 for d in st.get("side_data_list", [])):
        w, h = h, w
    s = min(W / w, H / h)
    ox, oy = (W - round(w * s)) / 2, (H - round(h * s)) / 2
    return [(x * s + ox, y * s + oy) for x, y in pts]


def has_audio(path: Path) -> bool:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
         "-of", "csv=p=0", str(path)], capture_output=True, text=True).stdout
    return bool(out.strip())


def prepare(clip: Path, work: Path):
    """Returns (video path, wav path or None, fps)."""
    video = work / "prepared.mp4"
    wav = work / "audio.wav"
    if not video.exists():
        fps = probe_fps(clip)
        vf = (f"scale={W}:{H}:force_original_aspect_ratio=decrease,"
              f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps}")
        run(["ffmpeg", "-y", "-i", str(clip), "-vf", vf, "-an", "-c:v", "libx264",
             "-preset", "veryfast", "-crf", "14", "-pix_fmt", "yuv420p", str(video)])
    if not wav.exists() and has_audio(clip):
        run(["ffmpeg", "-y", "-i", str(clip), "-vn", "-ac", "1", "-ar", str(SR),
             "-c:a", "pcm_s16le", str(wav)])
    return video, (wav if wav.exists() else None), probe_fps(video)


def frame_count(video: Path) -> int:
    cap = cv2.VideoCapture(str(video))
    n = 0
    while cap.grab():
        n += 1
    return n


def read_wav(wav: Path) -> np.ndarray:
    with wave.open(str(wav)) as f:
        return np.frombuffer(f.readframes(f.getnframes()), dtype="<i2").astype(np.float32) / 32768


def racket_onsets(wav: Path | None):
    """Onsets in the 2-8 kHz band. Returns [(time s, strength 0..1)], loudest = 1.

    Onsets are found on log energy (level-independent); strength is the linear band energy
    just after the onset, so quiet clicks from neighbouring courts score low.
    """
    if wav is None:
        return []
    x = read_wav(wav)
    x = sosfiltfilt(butter(4, [2000, 8000], btype="band", fs=SR, output="sos"), x)
    hop = int(SR * 0.005)
    n = len(x) // hop
    energy = np.log10((x[: n * hop].reshape(n, hop) ** 2).mean(1) + 1e-10)
    flux = np.maximum(np.diff(energy, prepend=energy[0]), 0)
    flux = np.convolve(flux, np.ones(3) / 3, mode="same")
    med = np.median(flux)
    mad = np.median(np.abs(flux - med)) + 1e-9
    peaks, _ = find_peaks(flux, height=med + 8 * mad, distance=int(0.15 / 0.005))
    if not len(peaks):
        return []
    lin = 10 ** energy
    loud = np.array([lin[p: p + 4].max() for p in peaks])
    return [(p * hop / SR, float(v / loud.max())) for p, v in zip(peaks, loud)]
