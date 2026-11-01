# badminton-hud

Adds a game-style HUD to a one-rally badminton clip (portrait, filmed from behind the
baseline, hand-held is fine).

```bash
hud my_rally.mov
```

Produces a folder named after the clip, next to it (`my_rally/`), holding:

| File | What |
| --- | --- |
| `my_rally_HUD.mp4` | 1080x1920 video with the HUD and the original audio |
| `my_rally_HUD_report/index.html` | error-check report: court on 8 frames, every shot (at the hit and 0.2 s later), long stretches without a shot, warnings |
| `my_rally_hud.toml` | players and fixes; created on the first run |
| `my_rally_hud_work/` | cached steps (YOLO, camera, court, shuttle); delete it or pass `--fresh` to recompute |

## Fixing a run

Read the warnings in the report, then edit `my_rally_hud.toml` and run `hud my_rally.mov`
again. Only the quick steps re-run.

```toml
[fixes]
add_shots = [3.42, { time = 5.10, hitter = "Alif" }]   # seconds; hitter optional
remove_shots = [7.85]                                    # removes the shot within 0.2 s
rally_end = 12.8
```

Players (names, shirt colour, side of the net) are also in that file.

### Fixed camera not behind the baseline

Automatic court finding expects the behind-the-baseline view. For a camera on a stand
somewhere else (a corner, the side), give 4+ court points once and reuse them for every clip
from that spot: `[x m, y m, pixel x, pixel y]` in the original clip's pixels, with
x = 0 / 0.46 / 3.05 / 5.64 / 6.10 from the camera side's left and y = 0 (near baseline) /
0.76 / 4.72 / 6.70 (net).

```toml
[court]
points = [[0, 4.72, 256, 763], [3.05, 4.72, 564, 720], [3.05, 0.76, 934, 823], [6.1, 0, 1248, 755]]
```

Without racket sounds (silent audio), shots come from seen flights only: a bright flight that
starts or ends at one of our players' rackets. From a corner, a near player at the net and a
far player behind them overlap on screen, so check the hitters in the report.

## How it works

1. **Prepare** (`prep.py`): ffmpeg to 1080x1920, audio to WAV, racket sounds = onsets in the 2-8 kHz band.
2. **Camera motion** (`camera.py`): frame-to-frame homography from features above the far baseline (walls, ceiling), people masked out.
3. **People** (`people.py`): YOLO11m + ByteTrack on every frame, imgsz 1280, CUDA/MPS when available; a torso colour sample per box.
4. **Court, first frame** (`court.py`): white paint on the floor -> Hough segments -> vanishing point of the lines running away -> the two lowest long cross lines are the baseline (0 m) and doubles long service line (0.76 m) -> the running-away lines are matched to x = 0, 0.46, 3.05, 5.64, 6.10 m with a cross-ratio (1-D projective) fit -> homography, snapped onto the paint. If the baseline is cut off by the frame, the other near-side pairings (0 / 0.76 / 4.72 m) are tried and the fit covering the most white paint wins. Drawn over the first frame in the report.
5. **Court, every frame**: predict with the camera motion, snap onto the white lines using the near half, and every 2nd frame re-read the lines from the picture; keep whichever fit covers more white paint.
6. **Who is who** (`identity.py`): side of the net picks the team, shirt hue/saturation/brightness picks the player within the team (re-centred on the clip's own lighting), last position only breaks near-ties; then one name per ByteTrack id.
7. **Shuttle** (`shuttle.py`): small bright blobs brighter than frames t±3 and t±10, each aligned by the camera and by the floor plane -> tracklets -> one path through the rally that may only change direction inside one of our players' racket zones.
8. **Shots** (`shots.py`): direction changes at a racket, plus racket sounds where the shuttle is not seen; the best alternating-sides sequence wins; hitter = team member closer to the shuttle. Launch speed = (e^(kD) - 1) / (kT), k = 9.81 / 6.8^2.
9. **HUD** (`render.py`): Pillow, Lato and the PD dark-mode colour tokens; original audio added back.
10. **Report** (`report.py`): pictures and warnings (speed outside 25-250 km/h, shots less than 0.33 s or more than 2.4 s apart, guessed hitters, weak court fits, unidentified players).

Stamina is a game-style number (it drains with distance and sprinting), not a measurement.

## Tests

```bash
.venv/bin/python tests/test_court_synthetic.py            # court fit on drawn courts
.venv/bin/python tests/synth_rally.py <cutouts> <out>     # synthetic rally with ground truth
hud <out>/synth.mp4 && .venv/bin/python tests/eval_synth.py <out>
```

`synth_rally.py` needs four person cut-outs (`near_red.png`, `near_pink.png`, `far_red.png`,
`far_blue.png`) cropped from any real footage.

## Setup

```bash
brew install ffmpeg python@3.12
python3.12 -m venv .venv && .venv/bin/pip install -e .
ln -sf "$PWD/.venv/bin/hud" ~/.local/bin/hud
```

`models/yolo11m.pt` is downloaded by Ultralytics on first use if missing.

The full build prompt, matching the current behaviour, is in [PROMPT.md](PROMPT.md).
