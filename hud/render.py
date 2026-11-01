"""Step 9: draw the HUD with Pillow and put the original audio back."""
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import court

FONTS = Path(__file__).resolve().parent / "fonts"
# PD design system, dark mode (tokens/ui-color.json).
TEXT_1 = (246, 246, 246, 255)        # UI/Text/1
TEXT_3 = (255, 255, 255, 153)        # UI/Text/3
PANEL = (0, 0, 0, 128)               # UI/Image Overlay/Black A50
HAIRLINE = (255, 255, 255, 36)       # UI/Shape/Neutral 3
LINE = (255, 255, 255, 102)          # UI/Image Overlay/White A40
POSITIVE = (29, 213, 141, 255)       # UI/Text/Positive
WARNING = (244, 194, 81, 255)        # UI/Text/Warning
NEGATIVE = (255, 122, 133, 255)      # UI/Text/Negative
RING_R = 0.45  # m


def font(weight, size):
    return ImageFont.truetype(str(FONTS / f"Lato-{weight}.ttf"), size)


F = {
    "label": font("Bold", 22), "value": font("Black", 54), "unit": font("Bold", 26),
    "tag": font("Bold", 30), "tag_far": font("Bold", 24), "row": font("Bold", 26), "small": font("Bold", 22),
}


def spaced(d, xy, text, f, fill, tracking=2.0, anchor="la"):
    """Caps label with letter-spacing (Pillow has no tracking)."""
    w = sum(d.textlength(c, font=f) for c in text) + tracking * (len(text) - 1)
    x, y = xy
    if anchor[0] == "m":
        x -= w / 2
    elif anchor[0] == "r":
        x -= w
    for c in text:
        d.text((x, y), c, font=f, fill=fill, anchor="l" + anchor[1])
        x += d.textlength(c, font=f) + tracking
    return w


def stats(tracks, fps, n):
    """Cumulative distance (m) and a game-style stamina (100 -> lower with running)."""
    out = {}
    for name, tr in tracks.items():
        pos = tr["pos"]
        dist, stam = np.zeros(n), np.zeros(n)
        total, s = 0.0, 100.0
        for t in range(n):
            if t and not np.isnan(pos[t, 0]) and not np.isnan(pos[t - 1, 0]):
                step = float(np.linalg.norm(pos[t] - pos[t - 1]))
                if step * fps < 8:  # faster than a sprint = tracking glitch
                    total += step
                    s -= (0.35 + 1.2 * max(0.0, step * fps - 1.0)) / fps
            dist[t], stam[t] = total, max(0.0, s)
        out[name] = (dist, stam)
    return out


class Hud:
    def __init__(self, players, tracks, shots, start, end, Hs, shuttle_pos, fps, n):
        self.players, self.tracks, self.shots = players, tracks, shots
        self.start, self.end, self.Hs, self.pos, self.fps = start, end, Hs, shuttle_pos, fps
        self.stats = stats(tracks, fps, n)
        self.color = {p.name: p.rgb for p in players}

    def draw(self, frame_bgr, t):
        img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        now = t / self.fps
        done = [s for s in self.shots if s.time <= now]
        self._rings(d, t)
        self._tags(d, t)
        self._top(d, now, done)
        self._roster(d, t)
        self._map(d, t, done)
        return np.asarray(Image.alpha_composite(img, layer).convert("RGB"))

    # -- on the court

    def _rings(self, d, t):
        ang = np.linspace(0, 2 * np.pi, 48)
        for name, tr in self.tracks.items():
            p = tr["pos"][t]
            if np.isnan(p[0]):
                continue
            c = self.color[name]
            for r, a, wdt in ((RING_R, 255, 5), (RING_R * 1.35, 90, 3)):
                q, z = court.project(self.Hs[t], np.c_[p[0] + r * np.cos(ang), p[1] + r * np.sin(ang)])
                if (z > 0).all():
                    pts = [tuple(v) for v in q]
                    if r == RING_R:
                        d.polygon(pts, fill=c + (60,))
                    d.line(pts + [pts[0]], fill=c + (a,), width=wdt, joint="curve")

    def _tags(self, d, t):
        tags = []
        for p in self.players:
            bx = self.tracks[p.name]["box"][t]
            if np.isnan(bx[0]):
                continue
            far = p.side == "far"
            f = F["tag_far" if far else "tag"]
            h = 40 if far else 50
            w = d.textlength(p.name, font=f) + h / 2 + 28
            cx, top = (bx[0] + bx[2]) / 2, bx[1] - (14 if far else 22)
            tags.append([cx - w / 2, top - h, w, h, cx, p, f])
        # Lift a tag above any lower tag it would overlap (partners standing in line).
        tags.sort(key=lambda g: -g[1])
        for i, g in enumerate(tags):
            for o in tags[:i]:
                if g[0] < o[0] + o[2] and o[0] < g[0] + g[2] and g[1] < o[1] + o[3] and o[1] < g[1] + g[3]:
                    g[1] = o[1] - g[3] - 8
        for x0, y0, w, h, cx, p, f in tags:
            top = y0 + h
            d.rounded_rectangle((x0, y0, x0 + w, top), radius=h / 2, fill=PANEL)
            r = h * 0.18
            dx = x0 + h / 2
            d.ellipse((dx - r, y0 + h / 2 - r, dx + r, y0 + h / 2 + r), fill=self.color[p.name] + (255,))
            d.text((dx + r + 8, y0 + h / 2), p.name, font=f, fill=TEXT_1, anchor="lm")
            d.polygon([(cx - 8, top), (cx + 8, top), (cx, top + 8)], fill=PANEL)

    # -- panels

    def _top(self, d, now, done):
        elapsed = min(max(0.0, now - self.start), self.end - self.start) if done else 0.0
        m, s = divmod(elapsed, 60)
        last = done[-1] if done else None
        flash = last is not None and now - last.time < 0.35
        speed = f"{last.speed:.0f}" if last and last.speed else "--"
        cells = [("RALLY TIME", f"{int(m):02d}:{s:04.1f}", None),
                 ("RALLY COUNT", str(len(done)), None),
                 ("SHOT SPEED", speed, "km/h")]
        x0, y0, x1, y1 = 32, 48, 1048, 196
        d.rounded_rectangle((x0, y0, x1, y1), radius=20, fill=PANEL)
        cw = (x1 - x0) / 3
        for i, (label, value, unit) in enumerate(cells):
            cx = x0 + cw * (i + 0.5)
            if i:
                d.line([(x0 + cw * i, y0 + 28), (x0 + cw * i, y1 - 28)], fill=HAIRLINE, width=2)
            spaced(d, (cx, y0 + 38), label, F["label"], TEXT_3, anchor="mm")
            color = WARNING if (flash and i == 2) else TEXT_1
            vw = d.textlength(value, font=F["value"])
            uw = d.textlength(" " + unit, font=F["unit"]) if unit else 0
            vx = cx - (vw + uw) / 2
            d.text((vx, y0 + 100), value, font=F["value"], fill=color, anchor="ls")
            if unit:
                d.text((vx + vw, y0 + 100), " " + unit, font=F["unit"], fill=TEXT_3, anchor="ls")

    def _roster(self, d, t):
        x0, y1 = 32, 1888
        row = 74
        y0 = y1 - 56 - row * len(self.players)
        x1 = 640
        d.rounded_rectangle((x0, y0, x1, y1), radius=20, fill=PANEL)
        spaced(d, (x0 + 24, y0 + 34), "STAMINA", F["label"], TEXT_3, anchor="lm")
        spaced(d, (x1 - 24, y0 + 34), "DISTANCE", F["label"], TEXT_3, anchor="rm")
        for i, p in enumerate(self.players):
            dist, stam = self.stats[p.name]
            y = y0 + 56 + row * i + row / 2
            c = self.color[p.name]
            d.ellipse((x0 + 24, y - 9, x0 + 42, y + 9), fill=c + (255,))
            d.text((x0 + 54, y - 12), p.name, font=F["row"], fill=TEXT_1, anchor="lm")
            bx0, bx1, by = x0 + 54, x1 - 150, y + 16
            d.rounded_rectangle((bx0, by - 5, bx1, by + 5), radius=5, fill=HAIRLINE)
            s = stam[t]
            col = POSITIVE if s > 60 else WARNING if s > 30 else NEGATIVE
            d.rounded_rectangle((bx0, by - 5, bx0 + max(10, (bx1 - bx0) * s / 100), by + 5), radius=5, fill=col)
            d.text((bx1 - 0, y - 12), f"{s:.0f}%", font=F["small"], fill=TEXT_3, anchor="rm")
            d.text((x1 - 24, y), f"{dist[t]:.1f} m", font=F["row"], fill=TEXT_1, anchor="rm")

    def _map(self, d, t, done):
        x1, y1 = 1048, 1888
        mh = 400
        scale = mh / court.LENGTH
        mw = court.WIDTH * scale
        pad = 24
        x0, y0 = x1 - mw - 2 * pad, y1 - mh - 2 * pad - 44
        d.rounded_rectangle((x0, y0, x1, y1), radius=20, fill=PANEL)
        spaced(d, ((x0 + x1) / 2, y0 + 30), "COURT MAP", F["label"], TEXT_3, anchor="mm")
        ox, oy = x0 + pad, y1 - pad  # world (0,0) = near-left corner at the bottom

        def m(p):
            return ox + p[0] * scale, oy - p[1] * scale

        for p, q in court.segments():
            d.line([m(p), m(q)], fill=LINE, width=2)
        d.line([m((-0.15, court.NET_Y)), m((court.WIDTH + 0.15, court.NET_Y))], fill=TEXT_1, width=3)
        if done and t / self.fps <= self.end:
            s = done[-1]
            a = self.tracks[s.hitter]["pos"][int(round(s.time * self.fps))]
            nxt = next((n_ for n_ in self.shots if n_.time > s.time), None)
            if nxt is not None and not np.isnan(a[0]):
                b = self.tracks[nxt.hitter]["pos"][min(len(self.pos) - 1, int(round(nxt.time * self.fps)))]
                if not np.isnan(b[0]):
                    d.line([m(a), m(b)], fill=WARNING, width=3)
                    bx, by = m(b)
                    d.ellipse((bx - 5, by - 5, bx + 5, by + 5), fill=WARNING)
        for p in self.players:
            pos = self.tracks[p.name]["pos"][t]
            if np.isnan(pos[0]):
                continue
            cx, cy = m(np.clip(pos, [-1, -2], [court.WIDTH + 1, court.LENGTH + 2]))
            d.ellipse((cx - 13, cy - 13, cx + 13, cy + 13), fill=self.color[p.name] + (255,),
                      outline=TEXT_1, width=2)
            d.text((cx, cy), p.name[0], font=F["small"], fill=(0, 0, 0, 255), anchor="mm")


def render(hud, read_frames, n, fps, clip: Path, out: Path, progress):
    cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1080x1920", "-r", f"{fps}",
           "-i", "-", "-i", str(clip), "-map", "0:v", "-map", "1:a?", "-c:v", "libx264",
           "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
           "-shortest", "-movflags", "+faststart", str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    for t, frame in enumerate(read_frames()):
        proc.stdin.write(hud.draw(frame, t).tobytes())
        progress(t + 1)
    proc.stdin.close()
    if proc.wait():
        raise SystemExit("ffmpeg failed while writing the HUD video")
