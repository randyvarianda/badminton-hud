"""Per-clip config: players and manual fixes, in <name>_hud.toml next to the clip."""
import colorsys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

TEMPLATE = '''# HUD config for {clip}
# Edit, then run `hud {clip}` again. Heavy steps are cached, so re-runs are quick.

# side = "near" (camera side of the net) or "far".
# color = a name (blue, magenta, red, pink, green, yellow, orange, purple, cyan, white, black)
#         or a hex shirt colour like "#d02040".
[[players]]
name = "Randy"
color = "blue"
side = "near"

[[players]]
name = "Maria"
color = "magenta"
side = "near"

[[players]]
name = "Alif"
color = "red"
side = "far"

[[players]]
name = "Rizky"
color = "pink"
side = "far"

[fixes]
# Times are seconds in the clip, as printed in the error report.
# add_shots = [3.42, {{ time = 5.10, hitter = "Alif" }}]
# remove_shots = [7.85]
# rally_end = 12.8
add_shots = []
remove_shots = []

# Fixed camera that is not behind the baseline? Give 4+ court points instead of letting the
# tool find the court: [x metres across, y metres from the near baseline, pixel x, pixel y]
# in the original clip's pixels. x: 0 / 0.46 / 3.05 / 5.64 / 6.10 (left to right from the
# camera side); y: 0 baseline / 0.76 / 4.72 / 6.70 net. Reuse it for clips from the same spot.
# [court]
# points = [[0, 4.72, 257, 763], [3.05, 4.72, 564, 720], [3.05, 0.76, 934, 823], [6.1, 0, 1248, 755]]
'''

# Typical shirt colours as (hue degrees, saturation, value). Only a starting point:
# identity.py re-centres them on the clip's own torso colours.
NAMED_COLORS = {
    "blue": (215, 0.70, 0.60),
    "magenta": (315, 0.65, 0.70),
    "red": (355, 0.80, 0.60),
    "pink": (335, 0.45, 0.90),
    "green": (120, 0.60, 0.55),
    "yellow": (55, 0.75, 0.85),
    "orange": (25, 0.80, 0.85),
    "purple": (275, 0.55, 0.50),
    "cyan": (190, 0.60, 0.75),
    "white": (0, 0.05, 0.95),
    "black": (0, 0.10, 0.12),
}


@dataclass
class Player:
    name: str
    color: str
    side: str  # "near" | "far"

    @property
    def hsv(self):
        c = self.color.lower()
        if c in NAMED_COLORS:
            return NAMED_COLORS[c]
        r, g, b = (int(c.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4))
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        return h * 360, s, v

    @property
    def rgb(self):
        h, s, v = self.hsv
        # Brighten for drawing on video.
        r, g, b = colorsys.hsv_to_rgb(h / 360, min(s, 0.85), max(v, 0.85))
        return int(r * 255), int(g * 255), int(b * 255)


@dataclass
class Fixes:
    add_shots: list = field(default_factory=list)  # [(time, hitter or None)]
    remove_shots: list = field(default_factory=list)
    rally_end: float | None = None


@dataclass
class Config:
    path: Path
    players: list
    fixes: Fixes
    court_points: list | None = None  # [[x m, y m, px, py]] in original clip pixels


def load(clip: Path, folder: Path) -> Config:
    path = folder / (clip.stem + "_hud.toml")
    if not path.exists():
        path.write_text(TEMPLATE.format(clip=clip.name))
    raw = tomllib.loads(path.read_text())
    players = [Player(p["name"], p["color"], p["side"]) for p in raw.get("players", [])]
    sides = [p.side for p in players]
    if len(players) != 4 or sides.count("near") != 2 or sides.count("far") != 2:
        raise SystemExit(f"{path}: need 4 players, 2 with side='near' and 2 with side='far'")
    f = raw.get("fixes", {})
    adds = []
    for a in f.get("add_shots", []):
        if isinstance(a, dict):
            adds.append((float(a["time"]), a.get("hitter")))
        else:
            adds.append((float(a), None))
    names = {p.name for p in players}
    for _, h in adds:
        if h is not None and h not in names:
            raise SystemExit(f"{path}: add_shots hitter {h!r} is not one of {sorted(names)}")
    fixes = Fixes(adds, [float(t) for t in f.get("remove_shots", [])], f.get("rally_end"))
    points = raw.get("court", {}).get("points")
    if points is not None and (len(points) < 4 or any(len(p) != 4 for p in points)):
        raise SystemExit(f"{path}: [court] points needs 4+ entries of [x m, y m, px, py]")
    return Config(path, players, fixes, points)
