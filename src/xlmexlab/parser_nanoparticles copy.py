"""
Shared helpers for the graph-grounding code (moved out of parser_nanoparticles.py
so graph_grounder.py does not depend on that file).
 
Contents: prompt builders, JSON/number parsing, coordinate conversion,
axis calibration (NumericAxis / CategoricalAxis) and marker snapping.
"""
import json
import re
from typing import Any, List, Optional
 
import numpy as np
 
from xlmexlab.prompt import PromptFormatter  # noqa: F401  (re-exported for graph_grounder)

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

import difflib
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy import ndimage

"""
Grid-assisted direct extraction with a small VLM.

Idea: small VLMs read a value much better when there is a labelled line next to
the point than when they must interpolate between two far-apart tick labels.
So, after calibrating the axes (in code), we draw a dense grid IN DATA UNITS
and print the value of every major line at the image border, then ask the VLM
to read the series.

Rules that make the grid help instead of hurt
- the grid is painted only on background pixels, so markers, lines and error
  bars are never covered;
- gridlines are neutral gray (never a series colour), minor lines lighter;
- labels go in a white margin outside the figure, never on top of the data;
- the image is upscaled first (small figures are the main source of error);
- ask ONE series per call (or one x position per call), never "all series".

Typical use (after GraphGrounder has calibrated the axes):
    g = draw_value_grid(image_path, out_path, x_map, y_map, box)
    prompt_dict = build_grid_read_prompt(series, g)
    raw = vlm.run_image_single_prompt_rescale(prompt, out_path)
    pts = parse_grid_answer(raw)
"""
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_PATHS = ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "arial.ttf"]


def _font(size: int):
    for p in FONT_PATHS:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


# ───────────────────────── axis helpers ─────────────────────────
class LinearMap:
    """value = a * pixel + b  (or 10**(a*pixel+b) when log). Built from tick pairs,
    or wrap an existing NumericAxis with LinearMap.from_axis(axis)."""

    def __init__(self, a: float, b: float, log: bool = False):
        self.a, self.b, self.log = a, b, log

    @classmethod
    def from_ticks(cls, pix: Sequence[float], vals: Sequence[float], log: bool = False):
        v = np.log10(vals) if log else np.asarray(vals, float)
        a, b = np.polyfit(np.asarray(pix, float), v, 1)
        return cls(float(a), float(b), log)

    @classmethod
    def from_axis(cls, axis):                       # NumericAxis from grounding_utils
        return cls(float(axis.a), float(axis.b), bool(axis.log))

    def value(self, p: float) -> float:
        v = self.a * p + self.b
        return 10 ** v if self.log else v

    def pixel(self, v: float) -> float:
        if self.log:
            v = math.log10(v)
        return (v - self.b) / self.a


def nice_step(span: float, n_lines: int) -> float:
    raw = abs(span) / max(1, n_lines)
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if m * mag >= raw:
            return m * mag
    return 10 * mag


def _major_minor(span: float, n_labels: int) -> Tuple[float, float]:
    """Labelled step ~span/n_labels (denser than the original ticks), minor = half of it."""
    major = nice_step(span, n_labels)
    mant = round(major / 10 ** math.floor(math.log10(major)), 3)
    return major, major / (5 if mant == 2.5 else 2)


def _grid_values(lo: float, hi: float, step: float) -> List[float]:
    lo, hi = min(lo, hi), max(lo, hi)
    start = math.ceil(lo / step - 1e-9) * step
    out, v = [], start
    while v <= hi + 1e-9:
        out.append(round(v, 10))
        v += step
    return out


def _log_values(lo: float, hi: float) -> Tuple[List[float], List[float]]:
    """(minor 1..9 x 10^k, major 10^k) between lo and hi."""
    minor, major = [], []
    for k in range(math.floor(math.log10(lo)), math.ceil(math.log10(hi)) + 1):
        for m in range(1, 10):
            v = m * 10 ** k
            if lo <= v <= hi:
                (major if m in (1, 2, 5) else minor).append(v)
    return minor, major


def _fmt(v: float) -> str:
    if v == 0:
        return "0"
    if abs(v) >= 1e4 or abs(v) < 1e-3:
        return f"{v:.0e}".replace("e+0", "e").replace("e-0", "e-")
    s = f"{v:.4f}".rstrip("0").rstrip(".")
    return s


# ───────────────────────── drawing ─────────────────────────
def draw_value_grid(image_path: str, out_path: str,
                    x_map: Optional[Any], y_map: Any, box: Tuple[float, float, float, float],
                    upscale: Optional[float] = None, target_width: int = 1100,
                    y_labels: int = 25, x_labels: int = 20,
                    x_positions: Optional[Sequence[float]] = None,
                    exclude: Sequence[Tuple[float, float, float, float]] = (),
                    bg_threshold: int = 225) -> Dict[str, Any]:
    """Draw a data-unit grid inside the plot box and save it.

    x_map / y_map : LinearMap or NumericAxis (x_map=None for categorical x / bars)
    box           : plot area in ORIGINAL pixels (x1, y1, x2, y2)
    x_positions   : optional data-x values to mark with a dashed vertical guide
                    (e.g. the shared x of the markers found by the colour digitizer)
    Returns the grid description used to build the prompt."""
    if y_map is not None and not isinstance(y_map, LinearMap):
        y_map = LinearMap.from_axis(y_map)
    if x_map is not None and not isinstance(x_map, LinearMap):
        x_map = LinearMap.from_axis(x_map)

    img = Image.open(image_path).convert("RGB")
    k = upscale or max(1.0, target_width / img.width)
    W, H = int(img.width * k), int(img.height * k)
    big = np.array(img.resize((W, H), Image.LANCZOS))
    x1, y1, x2, y2 = [int(round(v * k)) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W - 1, x2), min(H - 1, y2)

    # where the grid may be painted: background pixels inside the plot box, outside legend/inset
    can_paint = np.zeros((H, W), bool)
    can_paint[y1:y2 + 1, x1:x2 + 1] = True
    for ex in exclude:
        a, b, c, d = [int(round(v * k)) for v in ex]
        can_paint[max(0, b):d, max(0, a):c] = False
    can_paint &= (big.min(axis=2) >= bg_threshold)

    layer = np.zeros((H, W), np.uint8)              # 0 none, 1 minor, 2 major, 3 guide
    desc: Dict[str, Any] = {"y_minor": None, "y_major": None, "x_minor": None, "x_major": None,
                            "y_log": bool(y_map.log), "x_log": bool(x_map.log) if x_map else False}

    # horizontal lines (y values)
    vt, vb = y_map.value(y1 / k), y_map.value(y2 / k)
    if y_map.log:
        minor, major = _log_values(min(vt, vb), max(vt, vb))
    else:
        major_step, step = _major_minor(vt - vb, y_labels)
        allv = _grid_values(vb, vt, step)
        major = [v for v in allv if abs(v / major_step - round(v / major_step)) < 1e-6]
        minor = [v for v in allv if v not in major]
        desc["y_minor"], desc["y_major"] = step, major_step
    y_rows = []
    for lvl, vals in ((1, minor), (2, major)):
        for v in vals:
            r = int(round(y_map.pixel(v) * k))
            if y1 <= r <= y2:
                layer[r, x1:x2 + 1] = np.maximum(layer[r, x1:x2 + 1], lvl)
                if lvl == 2:
                    y_rows.append((r, v))

    # vertical lines (x values)
    x_cols = []
    if x_map is not None:
        vl, vr = x_map.value(x1 / k), x_map.value(x2 / k)
        if x_map.log:
            minor, major = _log_values(min(vl, vr), max(vl, vr))
        else:
            major_step, step = _major_minor(vr - vl, x_labels)
            allv = _grid_values(vl, vr, step)
            major = [v for v in allv if abs(v / major_step - round(v / major_step)) < 1e-6]
            minor = [v for v in allv if v not in major]
            desc["x_minor"], desc["x_major"] = step, major_step
        for lvl, vals in ((1, minor), (2, major)):
            for v in vals:
                c = int(round(x_map.pixel(v) * k))
                if x1 <= c <= x2:
                    layer[y1:y2 + 1, c] = np.maximum(layer[y1:y2 + 1, c], lvl)
                    if lvl == 2:
                        x_cols.append((c, v))
        for v in x_positions or []:                   # dashed guides at the data x positions
            c = int(round(x_map.pixel(v) * k))
            if x1 <= c <= x2:
                for r in range(y1, y2 + 1):
                    if (r // 6) % 2 == 0:
                        layer[r, c] = 3

    colours = {1: (222, 222, 222), 2: (165, 165, 165), 3: (120, 120, 120)}
    for lvl, col in colours.items():
        m = (layer == lvl) & can_paint
        big[m] = col

    # white margins with the value of every major line
    font = _font(max(13, int(H / 55)))
    ml, mt = 0, 34
    mr = 70
    canvas = Image.new("RGB", (W + ml + mr, H + mt), "white")
    canvas.paste(Image.fromarray(big), (ml, mt))
    d = ImageDraw.Draw(canvas)
    for r, v in y_rows:
        d.line([(W + ml, r + mt), (W + ml + 6, r + mt)], fill=(90, 90, 90))
        d.text((W + ml + 9, r + mt), _fmt(v), fill=(40, 40, 40), font=font, anchor="lm")
    for c, v in x_cols:
        d.line([(c + ml, mt - 6), (c + ml, mt)], fill=(90, 90, 90))
        d.text((c + ml, mt - 8), _fmt(v), fill=(40, 40, 40), font=font, anchor="mb")
    canvas.save(out_path)
    desc.update({"path": out_path, "scale": k, "offset": (ml, mt)})
    return desc


# ───────────────────────── prompting ─────────────────────────
def build_grid_read_prompt(series: Dict[str, str], grid: Dict[str, Any],
                           x_values: Optional[Sequence[Any]] = None) -> dict:
    """Prompt dict (same keys as your PromptFormatter) to read ONE series."""
    name, color = series.get("name", "UNKNOWN"), series.get("color", "UNKNOWN")
    marker = series.get("marker", "UNKNOWN")
    if grid["y_log"]:
        ygrid = ("The y axis is logarithmic. Gray horizontal lines are at 1, 2, ..., 9 x 10^k; "
                 "the darker ones are labelled on the RIGHT side of the image.")
        precision = "Give 2 significant digits."
    else:
        ygrid = (f"Light gray horizontal lines are every {_fmt(grid['y_minor'])} units; darker lines every "
                 f"{_fmt(grid['y_major'])} units, labelled on the RIGHT side of the image.")
        precision = f"Read each value to the nearest {_fmt(grid['y_minor'] / 2)}."
    if grid.get("x_minor"):
        xgrid = (f" Vertical gray lines are every {_fmt(grid['x_minor'])} units; darker ones every "
                 f"{_fmt(grid['x_major'])}, labelled at the TOP of the image.")
    else:
        xgrid = " The x axis is categorical: use the category names under the axis."
    if x_values:
        task = (f'For the series "{name}" (color={color}, marker={marker}) give its y value at each of these x '
                f"values: {list(x_values)}. Dashed vertical lines mark these x positions.")
    else:
        task = f'List every data point of the series "{name}" (color={color}, marker={marker}), left to right.'
    return {
        "expertise": "You are an expert at reading exact values from scientific plots using a measurement grid.",
        "initialization": "",
        "definitions": {},
        "objective": f"{task} {ygrid}{xgrid} {precision}",
        "answer_schema": {
            "Format": "x | y   (one point per line)",
            "Rules": "\n".join([
                "- Use the grid lines and their labels, not the original tick labels.",
                "- Use the CENTER of the marker (or the TOP of a bar; the BOTTOM for negative bars).",
                "- Ignore error bars, the legend and any inset plot.",
                "- If the series is not visible at an x value, write x | NA.",
                "- Never repeat a point. Stop after the last point.",
            ]),
        },
        "conclusion": "Return ONLY the lines in the format x | y. No explanations.",
    }


_NUM = r"-?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?"


def parse_grid_answer(text: str, max_points: int = 60) -> List[List[Any]]:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    pts, seen = [], set()
    for line in text.splitlines():
        if "|" not in line:
            continue
        xs, ys = [p.strip() for p in line.split("|", 1)]
        my = re.search(_NUM, ys.replace("−", "-"))
        if not my:
            continue
        mx = re.fullmatch(_NUM, xs.replace("−", "-").replace(",", ""))
        x = float(mx.group()) if mx else xs
        y = float(my.group().replace(",", "."))
        key = (str(x), round(y, 6))
        if key in seen:
            continue
        seen.add(key)
        pts.append([x, y])
        if len(pts) >= max_points:                      # runaway generation
            break
    return pts


# ───────────────────────── pipeline ─────────────────────────
def grid_extract(grounder, image_path: str, stage1: dict, out_dir: str, scale: float = 1.0,
                 ask_at_data_x: bool = True) -> Dict[str, Any]:
    """Calibrate with GraphGrounder, draw the value grid, ask the VLM ONE series per
    call, and compare with the pixel result.

    grounder : graph_grounder.GraphGrounder (its VLM and prompt template are reused)
    Returns {"grid": {series: [[x, y], ...]}, "pixel": {...}, "agreement": {...}, "grid_image": path}
    """
    import os
    from xlmexlab.prompt import PromptFormatter

    pixel = grounder.extract(image_path, stage1, scale=scale, debug_dir=out_dir)
    cal = getattr(grounder, "last_calibration", {}) or {}
    if not cal.get("y_map"):
        return {"grid": {}, "pixel": pixel, "agreement": {}, "error": "axis calibration failed"}

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]
    grid_path = os.path.join(out_dir, f"{base}_grid.png")
    data_x = cal.get("data_x") if ask_at_data_x else None
    g = draw_value_grid(image_path, grid_path, cal.get("x_map"), cal["y_map"], cal["box"],
                        exclude=cal.get("exclude", ()), x_positions=data_x or None)

    out: Dict[str, Any] = {}
    for s in stage1.get("series", []):
        fm = PromptFormatter(**build_grid_read_prompt(s, g, data_x or None))
        fm.model_post_init(grounder.template)
        raw = grounder.vlm.run_image_single_prompt_rescale(fm.format_prompt("<image>"), grid_path, scale=1.0)
        print(f"\n  [grid_extract] {s.get('name')} RAW:\n{raw}")
        out[s.get("name", "UNKNOWN")] = parse_grid_answer(raw)

    return {"grid": out, "pixel": pixel, "agreement": compare_series(out, pixel, cal["y_map"]),
            "grid_image": grid_path}


def compare_series(a: Dict[str, List], b: Dict[str, Any], y_map=None, x_tol: float = 0.02) -> Dict[str, Any]:
    """Mean |y_a - y_b| as a fraction of the y range, per series (points paired by x)."""
    rep = {}
    for name, pa in a.items():
        pb = b.get(name)
        if not isinstance(pb, list) or not pa or not pb:
            rep[name] = None
            continue
        ys = [p[1] for p in pa + pb if isinstance(p[1], (int, float))]
        yr = (max(ys) - min(ys)) or 1.0
        xs = [p[0] for p in pa + pb if isinstance(p[0], (int, float))]
        xr = ((max(xs) - min(xs)) or 1.0) if xs else 1.0
        errs = []
        for xa, ya in pa:
            for xb, yb in pb:
                same = (abs(xa - xb) / xr <= x_tol) if isinstance(xa, (int, float)) and isinstance(xb, (int, float)) \
                    else str(xa).strip().lower() == str(xb).strip().lower()
                if same:
                    errs.append(abs(ya - yb) / yr)
                    break
        rep[name] = {"paired": len(errs), "mean_err_frac": round(float(np.mean(errs)), 4) if errs else None}
    return rep

# ───────────────────────── prompts ─────────────────────────
def _prompt(expertise: str, objective: str, schema: str, rules: List[str]) -> dict:
    return {
        "expertise": expertise,
        "initialization": "",
        "definitions": {},
        "objective": objective,
        "answer_schema": {
            "Format": schema,
            "Rules": "\n".join(f"- {r}" for r in rules),
        },
        "conclusion": "Return ONLY the JSON list. No explanations, markdown, headers, or comments.",
    }
 
 
def build_tick_locator_prompt(axis: str, ticks: List[str]) -> dict:
    return _prompt(
        expertise="You are an expert at precisely localizing text in scientific figures.",
        objective=(
            f"Locate the tick labels of the {axis}-axis in the image. "
            f"The tick labels, in order, are: {ticks}."
        ),
        schema='[{"label": "<tick text exactly as in the list>", "bbox_2d": [x1, y1, x2, y2]}, ...]',
        rules=[
            "Return one entry per tick label that is visible in the image.",
            "bbox_2d must tightly enclose the tick label text only (not the axis title, not the legend).",
            "Use the label text exactly as given in the list.",
            "Do not invent labels that are not visible.",
        ],
    )
 
 
def build_points_prompt(name: str, color: str, marker: str, line: str) -> dict:
    return _prompt(
        expertise="You are an expert at precisely localizing data markers in scientific graphs.",
        objective=(
            f'Locate every data marker of the series "{name}" '
            f"(color={color}, marker={marker}, line={line})."
        ),
        schema='[{"point_2d": [x, y], "label": "<series name>"}, ...]',
        rules=[
            "Return the CENTER of each marker.",
            "One entry per marker, ordered left to right.",
            "Do NOT add points along the connecting line; only actual markers.",
            "Ignore markers of other series and legend symbols.",
        ],
    )
 
 
def build_bars_prompt(name: str, color: str) -> dict:
    return _prompt(
        expertise="You are an expert at precisely localizing bars in scientific bar charts.",
        objective=f'Locate every bar of the series "{name}" (color={color}).',
        schema='[{"bbox_2d": [x1, y1, x2, y2], "label": "<series name>"}, ...]',
        rules=[
            "bbox_2d must tightly enclose the bar rectangle only (exclude error bars and labels).",
            "One entry per bar, ordered left to right.",
            "Ignore bars of other series and legend swatches.",
        ],
    )
 
 
# ───────────────────────── parsing ─────────────────────────
_LIST_RX = re.compile(r"\[.*\]", re.S)
 
 
def parse_json_list(text: str) -> List[dict]:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"```(?:json)?", "", text)
    m = _LIST_RX.search(text)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        data = []
        for obj in re.findall(r"\{[^{}]*\}", m.group(0)):
            try:
                data.append(json.loads(obj))
            except json.JSONDecodeError:
                pass
    return [d for d in data if isinstance(d, dict)] if isinstance(data, list) else []
 
 
def to_float(s: Any) -> Optional[float]:
    s = str(s).replace("−", "-").replace("–", "-").replace(",", "").replace("%", "").strip()
    m = re.search(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", s)
    return float(m.group()) if m else None
 
 
def _norm_label(s: str) -> str:
    return re.sub(r"\s+", "", str(s).replace("−", "-")).lower()
 
 
# ───────────────────────── coordinates ─────────────────────────
class CoordSpace:
    """Converts model coordinates to ORIGINAL image pixels."""
 
    def __init__(self, width: int, height: int, mode: str = "norm1000", scale: float = 1.0):
        self.w, self.h, self.mode, self.scale = width, height, mode, scale
 
    def px(self, v: float, axis: str) -> float:
        if self.mode == "norm1000":
            return v / 1000.0 * (self.w if axis == "x" else self.h)
        return v / self.scale  # "absolute": pixels of the (rescaled) image fed to the model
 
    def bbox(self, b: List[float]):
        x1, y1, x2, y2 = b
        x1, x2 = sorted((self.px(x1, "x"), self.px(x2, "x")))
        y1, y2 = sorted((self.px(y1, "y"), self.px(y2, "y")))
        return x1, y1, x2, y2
 
 
# ───────────────────────── axis calibration ─────────────────────────
class NumericAxis:
    def __init__(self, pix: List[float], vals: List[float]):
        lin = self._fit(pix, vals, log=False)
        best, self.log = lin, False
        if len(vals) >= 3 and min(vals) > 0:
            lg = self._fit(pix, vals, log=True)
            if lin["err"] > 0.01 and lg["err"] < 0.5 * lin["err"]:
                best, self.log = lg, True
        self.a, self.b, self.err = best["a"], best["b"], best["err"]
        self.n_used = int(best["keep"].sum())
        self.n_total = len(vals)
 
    @staticmethod
    def _fit(pix, vals, log):
        p = np.asarray(pix, float)
        v = np.log10(np.asarray(vals, float)) if log else np.asarray(vals, float)
        keep = np.ones(len(p), bool)
        for _ in range(2):                       # drop up to 2 outlier ticks
            if keep.sum() < 3:
                break
            a, b = np.polyfit(p[keep], v[keep], 1)
            res = np.abs(a * p + b - v)
            worst = int(np.argmax(np.where(keep, res, -1)))
            if res[worst] / (np.ptp(v[keep]) or 1.0) > 0.05:
                keep[worst] = False
            else:
                break
        a, b = np.polyfit(p[keep], v[keep], 1)
        err = float(np.sqrt(np.mean((a * p[keep] + b - v[keep]) ** 2)) / (np.ptp(v[keep]) or 1.0))
        return {"a": a, "b": b, "err": err, "keep": keep}
 
    def value(self, p: float) -> float:
        v = self.a * p + self.b
        return float(10 ** v) if self.log else float(v)
 
    def pixel(self, v: float) -> float:
        if self.log:
            v = np.log10(v)
        return float((v - self.b) / self.a)
 
    def info(self):
        return {"type": "log" if self.log else "linear", "rel_rmse": round(self.err, 5),
                "ticks_used": f"{self.n_used}/{self.n_total}"}
 
 
class CategoricalAxis:
    def __init__(self, pix: List[float], labels: List[str]):
        self.pix, self.labels = np.asarray(pix, float), labels
 
    def value(self, p: float) -> str:
        return self.labels[int(np.argmin(np.abs(self.pix - p)))]
 
    def info(self):
        return {"type": "categorical", "n": len(self.labels)}
 
 
# ───────────────────────── refinement ─────────────────────────
def snap_to_marker(arr: np.ndarray, x: float, y: float, radius: int = 12, tol: float = 40.0):
    """Move (x, y) to the centroid of pixels with the same color as the VLM's point."""
    h, w, _ = arr.shape
    xi, yi = int(round(x)), int(round(y))
    if not (0 <= xi < w and 0 <= yi < h):
        return x, y
    ref = np.median(arr[max(0, yi - 2):yi + 3, max(0, xi - 2):xi + 3].reshape(-1, 3), axis=0)
    if ref.min() > 235:                          # landed on background -> don't trust a snap
        return x, y
    X0, X1 = max(0, xi - radius), min(w, xi + radius + 1)
    Y0, Y1 = max(0, yi - radius), min(h, yi + radius + 1)
    patch = arr[Y0:Y1, X0:X1].astype(float)
    mask = np.linalg.norm(patch - ref, axis=2) < tol
    if mask.sum() < 4:
        return x, y
    ys, xs = np.nonzero(mask)
    return X0 + xs.mean(), Y0 + ys.mean()


"""
Tick-label localisation with Tesseract OCR (exact text boxes) instead of VLM boxes.

    tokens = ocr_tokens(image_path)                         # every word / number + exact box
    axes   = locate_ticks(tokens, stage1, image_size)       # pick the x / y tick labels
    axes["x"] -> [(cx, cy, "text"), ...]   axes["y"] -> [...]   (original-image pixels)

How the tick labels are chosen (no VLM needed in most figures):
  * x ticks = tokens on one horizontal row, y ticks = tokens sharing a right edge
    (labels are right-aligned against the y axis);
  * numeric groups must fit value = a*pixel + b (or log10) with the right direction
    (x grows to the right, y grows upwards); misread tokens are dropped as outliers;
  * groups are scored by: consistent numbers + agreement with the stage-1 tick list
    + pixel span (so the main axis beats an inset's axis);
  * categorical x axes (bar charts) are matched to the stage-1 category names.
Optional: classify_tokens_with_vlm() asks a small VLM to label numbered boxes
(x_tick / y_tick / title / legend / inset / ...) when the geometry is ambiguous.

Requires: apt install tesseract-ocr ; pip install pytesseract
"""

try:
    import pytesseract
except ImportError:                                       # pragma: no cover
    pytesseract = None

NUM_WHITELIST = "0123456789.,-%eE^x×"


@dataclass
class Token:
    text: str
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    num: Optional[float] = None

    @property
    def cx(self): return (self.x1 + self.x2) / 2

    @property
    def cy(self): return (self.y1 + self.y2) / 2

    @property
    def h(self): return self.y2 - self.y1


# ───────────────────────── number parsing ─────────────────────────
_SUP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")
_FIX = str.maketrans({"O": "0", "o": "0", "l": "1", "I": "1", "|": "1", "S": "5", "−": "-", "–": "-", "—": "-"})


def parse_number(text: str) -> Optional[float]:
    """'1,000' '0.5' '0,5' '-20' '−20' '75%' '10³' '10^3' '1e3' '1×10^3' -> float."""
    t = text.strip().replace(" ", "")
    if not t:
        return None
    t = t.translate(_FIX).replace("%", "")
    m = re.fullmatch(r"(?:(-?\d+(?:\.\d+)?)[x×*])?10\^?\{?(-?\d+|[⁰¹²³⁴⁵⁶⁷⁸⁹⁻]+)\}?", t)
    if m:                                                 # 10^3, 10³, 2x10^3
        mant = float(m.group(1)) if m.group(1) else 1.0
        return mant * 10 ** float(m.group(2).translate(_SUP))
    t = t.translate(_SUP)
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", t):   # thousands separators
        t = t.replace(",", "")
    elif re.fullmatch(r"-?\d+,\d+", t):                   # decimal comma
        t = t.replace(",", ".")
    try:
        return float(t)
    except ValueError:
        return None


# ───────────────────────── OCR ─────────────────────────
def _remove_lines(gray: np.ndarray, dark: int = 140) -> np.ndarray:
    """Erase axis lines, frames and gridlines (they glue to labels and confuse OCR)."""
    d = gray < dark
    lines = ndimage.binary_opening(d, structure=np.ones((1, 30))) | \
            ndimage.binary_opening(d, structure=np.ones((30, 1)))
    rest = d & ~lines
    # tick marks: small thin pieces that touch an axis line (minus signs never touch it)
    lab, n = ndimage.label(rest)
    touch = ndimage.binary_dilation(lines, iterations=1)
    ticks = np.zeros_like(d)
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        if sl is None:
            continue
        comp = lab[sl] == i
        h, w = comp.shape
        if min(h, w) <= 3 and max(h, w) <= 12 and (touch[sl] & comp).any():
            ticks[sl] |= comp
    out = gray.copy()
    out[lines | ticks] = 255
    return out


def _to_ocr_image(img: Image.Image, scale: int) -> Image.Image:
    g = np.array(img.convert("L"))
    g = _remove_lines(g)
    im = Image.fromarray(g).resize((img.width * scale, img.height * scale), Image.LANCZOS)
    return ImageOps.autocontrast(im)


def _read_number_crop(ocr_img: Image.Image, t: Token, scale: int) -> Tuple[Optional[str], float]:
    """Re-read one token as a single number line with a digit whitelist."""
    pad = max(4, int(0.5 * t.h * scale))
    box = (max(0, int(t.x1 * scale) - pad), max(0, int(t.y1 * scale) - pad),
           min(ocr_img.width, int(t.x2 * scale) + pad), min(ocr_img.height, int(t.y2 * scale) + pad))
    crop = ImageOps.expand(ocr_img.crop(box), border=10, fill=255)
    d = pytesseract.image_to_data(crop, config=f"--psm 7 -c tessedit_char_whitelist={NUM_WHITELIST}",
                                  output_type=pytesseract.Output.DICT)
    words = [(d["text"][i], float(d["conf"][i])) for i in range(len(d["text"])) if d["text"][i].strip()]
    if not words:
        return None, -1
    return "".join(w for w, _ in words), float(np.mean([c for _, c in words]))


def ocr_tokens(image_path: str, scale: int = 3, min_conf: float = 0) -> List[Token]:
    """All text tokens with exact boxes in ORIGINAL image pixels.
    Adjacent words on one line are merged ('Free DOX', '2,000'); number-like tokens
    are re-read with a digit whitelist, which fixes '10¢'->'100', 'O'->'0', etc."""
    if pytesseract is None:
        raise ImportError("pip install pytesseract (and install the tesseract binary)")
    img = Image.open(image_path).convert("RGB")
    ocr_img = _to_ocr_image(img, scale)
    d = pytesseract.image_to_data(ocr_img, config="--psm 11", output_type=pytesseract.Output.DICT)

    words: List[Token] = []
    for i, txt in enumerate(d["text"]):
        txt = txt.strip()
        if not txt or float(d["conf"][i]) < min_conf:
            continue
        x, y, w, h = d["left"][i] / scale, d["top"][i] / scale, d["width"][i] / scale, d["height"][i] / scale
        if h < 4 or w < 2:
            continue
        words.append(Token(txt, x, y, x + w, y + h, float(d["conf"][i])))

    # merge words on the same line that are close together
    words.sort(key=lambda t: (round(t.cy / 4), t.x1))
    merged: List[Token] = []
    for t in words:
        if merged:
            p = merged[-1]
            same_line = abs(p.cy - t.cy) < 0.5 * max(p.h, t.h)
            gap = t.x1 - p.x2
            numeric = parse_number(p.text) is not None or parse_number(t.text) is not None \
                or re.fullmatch(r"[-–—_.]+", t.text) is not None
            if same_line and -2 < gap < 0.6 * max(p.h, t.h) and not numeric:
                merged[-1] = Token(p.text + ("" if gap < 0.25 * p.h else " ") + t.text,
                                   min(p.x1, t.x1), min(p.y1, t.y1), max(p.x2, t.x2), max(p.y2, t.y2),
                                   min(p.conf, t.conf))
                continue
        merged.append(t)

    for t in merged:
        t.num = parse_number(t.text)
        looks_numeric = sum(c.isdigit() for c in t.text) >= max(1, len(t.text) // 2) or \
            re.fullmatch(r"[\dOolIS.,%\-−¢()]+", t.text) is not None
        if t.num is None or (looks_numeric and t.conf < 85):
            txt2, conf2 = _read_number_crop(ocr_img, t, scale)
            n2 = parse_number(txt2) if txt2 else None
            if n2 is not None and (t.num is None or conf2 > t.conf):
                t.text, t.num, t.conf = txt2, n2, conf2
    return merged


# ───────────────────────── tick selection ─────────────────────────
def _cluster(values: Sequence[float], tol: float) -> List[List[int]]:
    order = np.argsort(values)
    groups: List[List[int]] = []
    for i in order:
        if groups and abs(values[i] - values[groups[-1][-1]]) <= tol:
            groups[-1].append(int(i))
        else:
            groups.append([int(i)])
    return [g for g in groups if len(g) >= 2]


def _consistent_numeric(toks: List[Token], axis: str) -> Tuple[List[Token], float, bool]:
    """Largest subset whose numbers fit a line (or log line) against pixel position
    with the right direction. Returns (subset, relative rmse, is_log)."""
    nums = [t for t in toks if t.num is not None]
    # one label per position (keep the most confident)
    nums.sort(key=lambda t: -t.conf)
    uniq: List[Token] = []
    for t in nums:
        if all(abs((t.cx if axis == "x" else t.cy) - (u.cx if axis == "x" else u.cy)) > 2 for u in uniq):
            uniq.append(t)
    if len(uniq) < 2:
        return [], 1.0, False
    fits = {}
    for log in (False, True):
        if log and (any(t.num <= 0 for t in uniq) or len(uniq) < 3):
            continue
        keep = list(uniq)
        while len(keep) >= 2:
            p = np.array([t.cx if axis == "x" else t.cy for t in keep])
            v = np.array([math.log10(t.num) if log else t.num for t in keep])
            if np.ptp(v) == 0:
                keep = []
                break
            a, b = np.polyfit(p, v, 1)
            if (axis == "x" and a <= 0) or (axis == "y" and a >= 0):
                keep = []                                   # wrong direction: not this axis
                break
            res = np.abs(a * p + b - v) / np.ptp(v)
            if res.max() <= 0.03 or len(keep) == 2:
                break
            keep.pop(int(np.argmax(res)))
        if len(keep) >= 2:
            p = np.array([t.cx if axis == "x" else t.cy for t in keep])
            v = np.array([math.log10(t.num) if log else t.num for t in keep])
            a, b = np.polyfit(p, v, 1)
            fits[log] = (keep, float(np.sqrt(np.mean(((a * p + b - v) / np.ptp(v)) ** 2))))
    if not fits:
        return [], 1.0, False
    lin, lg = fits.get(False), fits.get(True)
    if lg and (not lin or len(lg[0]) > len(lin[0])
               or (len(lg[0]) == len(lin[0]) and lin[1] > 0.01 and lg[1] < 0.5 * lin[1])):
        return lg[0], lg[1], True
    return lin[0], lin[1], False


def _expected_match(toks: List[Token], expected: Sequence[str]) -> int:
    if not expected:
        return 0
    exp_nums = {parse_number(e) for e in expected} - {None}
    n = 0
    for t in toks:
        if t.num is not None and t.num in exp_nums:
            n += 1
        elif difflib.get_close_matches(t.text.lower(), [e.lower() for e in expected], n=1, cutoff=0.75):
            n += 1
    return n


def _pick_axis(tokens: List[Token], axis: str, expected: Sequence[str], size: Tuple[int, int],
               allowed: Optional[set] = None) -> Tuple[List[Token], Dict[str, Any]]:
    W, H = size
    cand = [t for i, t in enumerate(tokens) if allowed is None or i in allowed]
    if not cand:
        return [], {"reason": "no tokens"}
    hmed = float(np.median([t.h for t in cand]))
    if axis == "x":
        groups = _cluster([t.cy for t in cand], tol=0.5 * hmed)
    else:   # right-aligned labels; also try left edge (right-hand axes / left-aligned labels)
        groups = _cluster([t.x2 for t in cand], tol=max(4.0, 0.7 * hmed)) + \
                 _cluster([t.x1 for t in cand], tol=max(4.0, 0.7 * hmed))
    numeric_expected = expected and sum(parse_number(e) is not None for e in expected) >= len(expected) / 2

    best, best_score, best_info = [], -1e9, {}
    for g in groups:
        toks = [cand[i] for i in g]
        if numeric_expected or not expected:
            sub, err, log = _consistent_numeric(toks, axis)
            if len(sub) < 2:
                continue
            pos = [t.cx if axis == "x" else t.cy for t in sub]
            span = (max(pos) - min(pos)) / (W if axis == "x" else H)
            score = len(sub) + 1.5 * _expected_match(sub, expected) + 3 * span - 5 * err
            info = {"type": "log" if log else "linear", "rel_rmse": round(err, 4)}
        else:                                               # categorical (bar chart)
            exp_l = [e.lower() for e in expected]
            sub = [t for t in toks if difflib.get_close_matches(t.text.lower(), exp_l, n=1, cutoff=0.7)]
            if len(sub) < 2:
                continue
            pos = [t.cx for t in sub]
            span = (max(pos) - min(pos)) / W
            score = 2 * len(sub) + 3 * span
            info = {"type": "categorical"}
        # tick labels sit at the bottom (x) / left (y) of the main plot more often than not
        score += 0.5 * (np.mean([t.cy for t in sub]) / H if axis == "x" else 1 - np.mean([t.cx for t in sub]) / W)
        if score > best_score:
            best, best_score, best_info = sub, score, info
    best_info["n"] = len(best)
    return sorted(best, key=lambda t: t.cx if axis == "x" else t.cy), best_info


def _canonical_text(t: Token, expected: Sequence[str]) -> str:
    """Return the stage-1 spelling when it matches (keeps your tick strings stable)."""
    for e in expected or []:
        n = parse_number(e)
        if t.num is not None and n is not None and abs(n - t.num) <= 1e-9 * max(1, abs(n)):
            return e
    m = difflib.get_close_matches(t.text, list(expected or []), n=1, cutoff=0.7)
    return m[0] if m else (t.text if t.num is None else repr(t.num).rstrip("0").rstrip("."))


def locate_ticks(tokens: List[Token], stage1: Dict[str, Any], size: Tuple[int, int],
                 vlm_labels: Optional[Dict[int, str]] = None) -> Dict[str, Any]:
    """Choose x and y tick labels among OCR tokens.
    vlm_labels: optional {token_index: label} from classify_tokens_with_vlm()."""
    out: Dict[str, Any] = {"x": [], "y": [], "info": {}}
    for axis in ("x", "y"):
        expected = [str(t) for t in (stage1.get(f"{axis}_ticks") or []) if t not in (None, "", "UNKNOWN")]
        allowed = None
        if vlm_labels:
            allowed = {i for i, lab in vlm_labels.items() if lab == f"{axis}_tick"}
            if len(allowed) < 2:
                allowed = None                              # VLM unhelpful -> geometry only
        toks, info = _pick_axis(tokens, axis, expected, size, allowed)
        if axis == "x" and out["y"]:                        # x labels must lie below the y labels' top
            pass
        out[axis] = [(t.cx, t.cy, _canonical_text(t, expected)) for t in toks]
        out["info"][axis] = info
    return out


# ───────────────────────── optional: VLM classification ─────────────────────────
TOKEN_LABELS = ["x_tick", "y_tick", "x_title", "y_title", "legend", "panel_label", "inset", "annotation", "other"]


def draw_token_marks(image_path: str, tokens: List[Token], out_path: str, zoom: int = 2) -> str:
    """Numbered red boxes around every OCR token (set-of-marks image for the VLM)."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((img.width * zoom, img.height * zoom), Image.LANCZOS)
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 12)
    except OSError:
        font = ImageFont.load_default()
    for i, t in enumerate(tokens):
        b = [t.x1 * zoom - 2, t.y1 * zoom - 2, t.x2 * zoom + 2, t.y2 * zoom + 2]
        d.rectangle(b, outline=(230, 0, 0), width=2)
        d.text((b[0], b[1] - 13), str(i), fill=(230, 0, 0), font=font)
    img.save(out_path)
    return out_path


def build_token_classification_prompt(tokens: List[Token]) -> dict:
    listing = "\n".join(f"{i} | {t.text}" for i, t in enumerate(tokens))
    return {
        "expertise": "You are an expert at reading the layout of scientific charts.",
        "initialization": "",
        "definitions": {},
        "objective": ("Every text in the image has a red box with a number. For each number, say what the "
                      f"text is. Allowed labels: {', '.join(TOKEN_LABELS)}.\n"
                      "x_tick / y_tick = tick labels of the MAIN plot axes only. Tick labels of a small "
                      "inset plot are 'inset'.\nTexts:\n" + listing),
        "answer_schema": {"Format": "number | label   (one line per text)",
                          "Rules": "- One line for every number.\n- Use only the allowed labels."},
        "conclusion": "Return ONLY the lines. No explanations.",
    }


def parse_token_labels(text: str, n_tokens: int) -> Dict[int, str]:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    out = {}
    for line in text.splitlines():
        m = re.match(r"\s*(\d+)\s*[|:\-]\s*([a-z_]+)", line.strip().lower())
        if m and int(m.group(1)) < n_tokens and m.group(2) in TOKEN_LABELS:
            out[int(m.group(1))] = m.group(2)
    return out


def classify_tokens_with_vlm(vlm, prompt_template_path: Optional[str], image_path: str,
                             tokens: List[Token], work_path: str) -> Dict[int, str]:
    from xlmexlab.prompt import PromptFormatter
    marked = draw_token_marks(image_path, tokens, work_path)
    fm = PromptFormatter(**build_token_classification_prompt(tokens))
    fm.model_post_init(prompt_template_path)
    raw = vlm.run_image_single_prompt_rescale(fm.format_prompt("<image>"), marked, scale=1.0)
    print(f"\n  [ocr_ticks] token labels RAW:\n{raw}")
    return parse_token_labels(raw, len(tokens))


# ───────────────────────── gap filling ─────────────────────────
def _read_crop_variants(images: List[Image.Image], box_px: Tuple[float, float, float, float],
                        scale: int) -> List[str]:
    """Read one small region as a number: whitelisted single-line and single-char modes,
    on the cleaned image and on the raw one (line removal can cut a '0' at the origin)."""
    out = []
    x1, y1, x2, y2 = [int(round(v * scale)) for v in box_px]
    for im in images:
        crop = im.crop((max(0, x1), max(0, y1), min(im.width, x2), min(im.height, y2)))
        if crop.width < 4 or crop.height < 4:
            continue
        crop = ImageOps.expand(crop, border=12, fill=255)
        for psm in (7, 8, 10):
            txt = pytesseract.image_to_string(
                crop, config=f"--psm {psm} -c tessedit_char_whitelist={NUM_WHITELIST}").strip()
            if txt:
                out.append(txt)
    return out


def fill_missing_ticks(image_path: str, picked: List[Tuple[float, float, str]], axis: str,
                       expected: Sequence[str], scale: int = 3, max_extend: int = 3) -> List[Tuple[float, float, str]]:
    """Predict where the missing tick labels must be (from the labels already found)
    and re-read only those spots. A reading is accepted only if it equals the value
    the axis fit predicts there, so a wrong OCR can never enter the calibration."""
    vals = [parse_number(t) for _, _, t in picked]
    if len(picked) < 2 or any(v is None for v in vals):
        return picked
    pos = np.array([p[0] if axis == "x" else p[1] for p in picked])
    v = np.array(vals, float)
    log = bool(np.all(v > 0) and len(v) >= 3 and
               np.std(np.diff(np.log10(np.sort(v)))) < 0.05 * np.ptp(np.log10(v)) < np.std(np.diff(np.sort(v))))
    vv = np.log10(v) if log else v
    a, b = np.polyfit(pos, vv, 1)

    exp_vals = sorted({parse_number(e) for e in expected or []} - {None})
    if len(exp_vals) >= 2:
        candidates = exp_vals
    else:
        sv = np.sort(vv)
        step = float(np.min(np.diff(sv))) if len(sv) > 1 else 0
        if step <= 0:
            return picked
        lo, hi = sv[0] - max_extend * step, sv[-1] + max_extend * step
        k = np.arange(round(lo / step), round(hi / step) + 1)
        candidates = [float(10 ** (s * step)) if log else round(float(s * step), 10) for s in k]
    have = set(np.round(v, 9))

    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    clean = _to_ocr_image(img, scale)
    raw = ImageOps.autocontrast(img.convert("L").resize((W * scale, H * scale), Image.LANCZOS))
    hh = 12.0                                              # typical label height in px
    if axis == "x":
        cy = float(np.median([p[1] for p in picked]))
        widths = 2.5 * hh
    else:
        cx = float(np.median([p[0] for p in picked]))
        half_w = max(1.6 * hh, 0.5 * max(len(t) for _, _, t in picked) * 0.7 * hh)

    added = list(picked)
    for val in candidates:
        if round(val, 9) in have:
            continue
        p = ((math.log10(val) if log else val) - b) / a
        if axis == "x":
            if not (0 <= p < W):
                continue
            box = (p - widths / 2, cy - hh, p + widths / 2, cy + hh)
        else:
            if not (0 <= p < H):
                continue
            box = (cx - half_w - 8, p - hh * 0.9, cx + half_w + 4, p + hh * 0.9)
        for txt in _read_crop_variants([clean, raw], box, scale):
            n = parse_number(txt)
            if n is not None and abs(n - val) <= 1e-9 * max(1.0, abs(val)):
                label = next((e for e in expected or [] if parse_number(e) == val), txt)
                added.append((p, cy, label) if axis == "x" else (cx, p, label))
                break
    key = 0 if axis == "x" else 1
    return sorted(added, key=lambda q: q[key])


def find_ticks(image_path: str, stage1: Dict[str, Any], scale: int = 3, fill_gaps: bool = True,
               vlm_labels: Optional[Dict[int, str]] = None, tokens: Optional[List[Token]] = None) -> Dict[str, Any]:
    """OCR + selection + gap filling. Returns {"x": [(cx, cy, text)], "y": [...], "info": {...}}."""
    tokens = tokens if tokens is not None else ocr_tokens(image_path, scale=scale)
    size = Image.open(image_path).size
    res = locate_ticks(tokens, stage1, size, vlm_labels)
    if fill_gaps:
        for axis in ("x", "y"):
            if res["info"].get(axis, {}).get("type") in ("linear", "log"):
                expected = [str(t) for t in (stage1.get(f"{axis}_ticks") or []) if t not in (None, "", "UNKNOWN")]
                before = len(res[axis])
                res[axis] = fill_missing_ticks(image_path, res[axis], axis, expected, scale)
                res["info"][axis]["filled"] = len(res[axis]) - before
    return res

"""
Drop-in replacement for GraphGrounder (same constructor, same extract() signature,
same output format), with three changes:

1. Tick labels: the VLM boxes are snapped to the real text blobs next to the
   axis lines and assigned in order -> removes the systematic offset seen with
   small images (labels 20-40 px off in your Fig. 1 overlay).
2. Data points: read from the pixels by colour (graph_color_digitizer). The VLM
   only gives the series colour name (stage 1) and, in one call each, the
   legend box and an inset box to ignore. VLM point lists are the fallback for
   black/gray series or when no colour match is found, and they are sanity-checked
   (inside the plot, no duplicates, no runaway repetition).
3. Overlay always saved (also when calibration fails) and shows where the tick
   labels really are, the plot box, the excluded boxes and each value.

Use it in extractor_nanoparticles.py:
    from xlmexlab.graph_grounder import GraphGrounder
"""
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

from xlmexlab.ocr_ticks import classify_tokens_with_vlm, find_ticks, ocr_tokens, parse_number
from xlmexlab.graph_color_digitizer import (
    ACHROMATIC, _region_mask, assign_colors, color_mask, dominant_colors, extract_bars,
    extract_line_series, name_to_rgb, plot_box_from_ticks, shared_x_positions, snap_ticks_to_text,
)


def build_region_prompt(what: str) -> dict:
    return _prompt(
        expertise="You are an expert at precisely localizing regions in scientific figures.",
        objective=f"Locate the {what} in the image. If there is no {what}, return [].",
        schema='[{"bbox_2d": [x1, y1, x2, y2], "label": "' + what + '"}]',
        rules=[
            "Return at most one box.",
            "The box must enclose the whole region, including its frame and text.",
            "Return [] if the region does not exist. Do not guess.",
        ],
    )


class GraphGrounder:
    def __init__(self, vlm_model, prompt_template_path: Optional[str],
                 coord_mode: str = "norm1000", refine: bool = True,
                 use_color: bool = True, snap_ticks: bool = True, find_regions: bool = True,
                 tick_source: str = "ocr", ocr_vlm_classify: bool = False):
        """tick_source: "ocr" (Tesseract boxes, VLM only as fallback) or "vlm".
        ocr_vlm_classify: also ask the VLM to label the OCR boxes (x_tick / y_tick /
        title / legend / inset ...) - useful for crowded figures with insets."""
        self.vlm = vlm_model
        self.template = prompt_template_path
        self.coord_mode = coord_mode
        self.refine = refine
        self.use_color = use_color
        self.snap_ticks = snap_ticks
        self.find_regions = find_regions
        self.tick_source = tick_source
        self.ocr_vlm_classify = ocr_vlm_classify
        self._ocr_axes: Optional[Dict[str, Any]] = None
        self.last_diagnostics: Dict[str, Any] = {}
        self.last_calibration: Dict[str, Any] = {}

    # ───────────────────────── VLM ─────────────────────────
    def _ask(self, prompt_dict: dict, image_path: str, scale: float) -> str:
        fm = PromptFormatter(**prompt_dict)
        fm.model_post_init(self.template)
        prompt = fm.format_prompt("<image>")
        out = self.vlm.run_image_single_prompt_rescale(prompt, image_path, scale=scale)
        print(f"\n  [GraphGrounder] RAW:\n{out}")
        return out

    # ───────────────────────── axes ─────────────────────────
    @staticmethod
    def _axis_from_pairs(pairs, axis):
        pos = [(p[0] if axis == "x" else p[1], p[2]) for p in pairs]
        nums = [(p, parse_number(t) if parse_number(t) is not None else to_float(t)) for p, t in pos]
        if len(nums) >= 2 and all(v is not None for _, v in nums):
            return NumericAxis([p for p, _ in nums], [v for _, v in nums]), pairs
        if pos:
            return CategoricalAxis([p for p, _ in pos], [t for _, t in pos]), pairs
        return None, []

    def _build_axis(self, image_path, axis, ticks, space, scale, gray):
        ocr_pairs = (self._ocr_axes or {}).get(axis) or []
        if len(ocr_pairs) >= 2:                               # exact OCR boxes: no VLM call
            self._tick_source_used[axis] = "ocr"
            return self._axis_from_pairs(ocr_pairs, axis)
        self._tick_source_used[axis] = "vlm"
        ticks = [str(t) for t in (ticks or []) if t not in (None, "", "UNKNOWN")]
        if not ticks:
            return None, []
        items = parse_json_list(self._ask(build_tick_locator_prompt(axis, ticks), image_path, scale))

        by_label = {}
        for it in items:
            if "bbox_2d" in it and len(it["bbox_2d"]) == 4:
                by_label.setdefault(_norm_label(it.get("label", "")), it["bbox_2d"])

        pairs: List[Tuple[float, float, str]] = []           # (cx, cy, tick text)
        for t in ticks:
            b = by_label.get(_norm_label(t))
            if b:
                x1, y1, x2, y2 = space.bbox(b)
                pairs.append(((x1 + x2) / 2, (y1 + y2) / 2, t))
        if len(pairs) < 2 and len(items) == len(ticks):      # labels ignored, boxes in order
            pairs = []
            for t, it in zip(ticks, items):
                if "bbox_2d" in it and len(it["bbox_2d"]) == 4:
                    x1, y1, x2, y2 = space.bbox(it["bbox_2d"])
                    pairs.append(((x1 + x2) / 2, (y1 + y2) / 2, t))

        if self.snap_ticks:
            h, w = gray.shape
            if len(pairs) < 2 and len(ticks) >= 2:           # VLM failed: try the text blobs alone
                n = len(ticks)
                dummy = ([(w * (i + 1) / (n + 1), h - 1) for i in range(n)] if axis == "x"
                         else [(1, h * (i + 1) / (n + 1)) for i in range(n)])
                snapped = snap_ticks_to_text(gray, dummy, axis)
                if snapped != dummy:
                    order = ticks if axis == "x" else ticks       # stage-1 order: left->right / as listed
                    key = 0 if axis == "x" else 1
                    snapped = sorted(snapped, key=lambda c: c[key])
                    if axis == "y" and to_float(order[0]) is not None and to_float(order[-1]) is not None \
                            and to_float(order[0]) < to_float(order[-1]):
                        order = list(reversed(order))             # top of image = largest value
                    pairs = [(c[0], c[1], t) for c, t in zip(snapped, order)]
            elif len(pairs) >= 2:
                snapped = snap_ticks_to_text(gray, [(p[0], p[1]) for p in pairs], axis)
                pairs = [(c[0], c[1], p[2]) for c, p in zip(snapped, pairs)]

        return self._axis_from_pairs(pairs, axis)

    def _region(self, what, image_path, space, scale):
        if not self.find_regions:
            return None
        items = parse_json_list(self._ask(build_region_prompt(what), image_path, scale))
        for it in items:
            if len(it.get("bbox_2d", [])) == 4:
                return space.bbox(it["bbox_2d"])
        return None

    @staticmethod
    def _is_bar(s: dict) -> bool:
        m = str(s.get("marker", "")).strip().lower()
        l = str(s.get("line", "")).strip().lower()
        return m in ("none", "", "unknown", "-") and l in ("none", "", "unknown", "-")

    @staticmethod
    def _sanitize(px_pts, box, exclude, n_x_ticks):
        """Drop VLM points outside the plot / inside legend or inset, duplicates, runaway lists."""
        x1, y1, x2, y2 = box
        keep = []
        for x, y in px_pts:
            if not (x1 - 5 <= x <= x2 + 5 and y1 - 5 <= y <= y2 + 5):
                continue
            if any(a <= x <= c and b <= y <= d for a, b, c, d in exclude):
                continue
            if any(abs(x - kx) < 3 and abs(y - ky) < 3 for kx, ky in keep):
                continue
            keep.append((x, y))
        limit = max(30, 4 * n_x_ticks)
        if len(keep) > limit:
            return [], "rejected: too many points (generation loop)"
        if len(keep) >= 5:                                     # evenly stepped lattice = repetition loop
            k = sorted(keep)
            dx, dy = np.diff([p[0] for p in k]), np.diff([p[1] for p in k])
            if np.std(dx) < 1.5 and np.std(dy) < 1.5:
                return [], "rejected: evenly spaced points (generation loop)"
        return keep, None

    # ───────────────────────── extraction ─────────────────────────
    def extract(self, image_path: str, stage1: dict, scale: float = 1.0,
                debug_dir: Optional[str] = None) -> Dict[str, Any]:
        img = Image.open(image_path).convert("RGB")
        arr = np.array(img)
        gray = np.array(img.convert("L")).astype(int)
        space = CoordSpace(img.width, img.height, self.coord_mode, scale)

        self._ocr_axes, self._tick_source_used = None, {}
        if self.tick_source == "ocr":
            try:
                toks = ocr_tokens(image_path)
                labels = None
                if self.ocr_vlm_classify and toks:
                    work = os.path.join(debug_dir or os.path.dirname(image_path) or ".",
                                        os.path.splitext(os.path.basename(image_path))[0] + "_ocr_marks.png")
                    labels = classify_tokens_with_vlm(self.vlm, self.template, image_path, toks, work)
                self._ocr_axes = find_ticks(image_path, stage1, tokens=toks, vlm_labels=labels)
                print(f"  [GraphGrounder] OCR ticks: {self._ocr_axes['info']}")
            except Exception as e:                            # tesseract missing, etc. -> VLM path
                print(f"  [GraphGrounder] OCR failed ({e}); using VLM tick boxes")

        x_map, x_anc = self._build_axis(image_path, "x", stage1.get("x_ticks"), space, scale, gray)
        y_map, y_anc = self._build_axis(image_path, "y", stage1.get("y_ticks"), space, scale, gray)

        result: Dict[str, Any] = {"x_axis": stage1.get("x_axis"), "y_axis": stage1.get("y_axis")}
        self.last_diagnostics = {"x_axis": x_map.info() if x_map else None,
                                 "y_axis": y_map.info() if y_map else None,
                                 "tick_source": dict(self._tick_source_used)}
        overlay: Dict[str, Any] = {"x": x_anc, "y": y_anc, "box": None, "exclude": [], "series": {}}

        if x_map is None or y_map is None:
            print("  [GraphGrounder] axis calibration failed -> no series extracted")
            self.last_diagnostics["error"] = "axis calibration failed"
            if debug_dir:
                self._save_overlay(img, overlay, debug_dir, image_path)
            return result

        box = plot_box_from_ticks([a[0] for a in x_anc], [a[1] for a in y_anc], arr.shape,
                                  x_label_y=float(np.median([a[1] for a in x_anc])),
                                  y_label_x=float(np.median([a[0] for a in y_anc])), gray=gray)
        exclude = []
        if self.use_color:
            for what in ("legend", "inset plot"):
                r = self._region(what, image_path, space, scale)
                if r and not (r[2] < box[0] or r[0] > box[2] or r[3] < box[1] or r[1] > box[3]):
                    exclude.append(r)
        overlay["box"], overlay["exclude"] = box, exclude
        region = _region_mask(arr.shape, box, exclude)
        zero_px = y_map.pixel(0.0) if isinstance(y_map, NumericAxis) and not y_map.log else None

        series = stage1.get("series", [])
        masks: Dict[str, np.ndarray] = {}
        if self.use_color and series:
            palette = dominant_colors(arr, box, exclude)
            cmap = assign_colors([(s.get("name", "UNKNOWN"), s.get("color", "")) for s in series], palette)
            pal = [c for c, _ in palette]
            for s in series:
                rgb = cmap.get(s.get("name", "UNKNOWN"))
                if rgb is not None:
                    masks[s.get("name", "UNKNOWN")] = color_mask(arr, rgb, region, palette=pal)
        line_masks = {s.get("name", "UNKNOWN"): masks[s.get("name", "UNKNOWN")]
                      for s in series if s.get("name", "UNKNOWN") in masks and not self._is_bar(s)}
        lines = extract_line_series(line_masks, [a[0] for a in x_anc]) if line_masks else {}
        # kept for grid_overlay.grid_extract (direct reading with a value grid)
        shared_px = shared_x_positions([pts for pts, kind in lines.values() if kind != "ticks"])
        self.last_calibration = {
            "x_map": x_map if isinstance(x_map, NumericAxis) else None, "y_map": y_map,
            "box": box, "exclude": exclude,
            "data_x": [round(x_map.value(p), 6) for p in shared_px] if isinstance(x_map, NumericAxis) else [],
        }

        for s in series:
            name = s.get("name", "UNKNOWN")
            color = s.get("color", "UNKNOWN")
            bar = self._is_bar(s)
            px_pts, method, note = [], None, None

            if name in masks:                                   # 1) pixels
                if bar:
                    px_pts, method = extract_bars(masks[name], zero_px), "color_bars"
                else:
                    px_pts, method = lines[name]
                    method = f"color_{method}"

            cname = str(color).strip().lower()
            chromatic = name_to_rgb(cname) is not None and cname not in ACHROMATIC
            if not px_pts and self.use_color and chromatic and name not in masks:
                # a coloured series whose colour is absent from the main plot is
                # only in the legend/inset -> do not let the VLM invent points
                px_pts, method, note = [], "skipped", "colour not found in main plot (inset/legend only?)"
            elif not px_pts:                                    # 2) VLM fallback
                if bar:
                    items = parse_json_list(self._ask(build_bars_prompt(name, color), image_path, scale))
                else:
                    items = parse_json_list(self._ask(
                        build_points_prompt(name, color, s.get("marker", "UNKNOWN"), s.get("line", "UNKNOWN")),
                        image_path, scale))
                for it in items:
                    if bar and len(it.get("bbox_2d", [])) == 4:
                        x1, y1, x2, y2 = space.bbox(it["bbox_2d"])
                        cy = y2 if (zero_px is not None and (y1 + y2) / 2 > zero_px) else y1
                        px_pts.append(((x1 + x2) / 2, cy))
                    elif "point_2d" in it and len(it["point_2d"]) == 2:
                        px_pts.append((space.px(it["point_2d"][0], "x"), space.px(it["point_2d"][1], "y")))
                    elif len(it.get("bbox_2d", [])) == 4:
                        x1, y1, x2, y2 = space.bbox(it["bbox_2d"])
                        px_pts.append(((x1 + x2) / 2, (y1 + y2) / 2))
                if self.refine and not bar:
                    px_pts = [snap_to_marker(arr, x, y) for x, y in px_pts]
                px_pts, note = self._sanitize(px_pts, box, exclude, len(x_anc))
                method = "vlm"

            px_pts = sorted(px_pts, key=lambda p: p[0])
            out = []
            for x, y in px_pts:
                xv, yv = x_map.value(x), y_map.value(y)
                out.append([round(xv, 4) if isinstance(xv, float) else xv, round(yv, 4)])
            result[name] = out
            overlay["series"][name] = (px_pts, out, method)
            self.last_diagnostics[name] = {"kind": "bars" if bar else "points", "method": method,
                                           "n": len(out), **({"note": note} if note else {})}

        if debug_dir:
            self._save_overlay(img, overlay, debug_dir, image_path)
        return result

    # ───────────────────────── debug image ─────────────────────────
    @staticmethod
    def _save_overlay(img, ov, debug_dir, image_path, zoom: int = 2):
        os.makedirs(debug_dir, exist_ok=True)
        im = img.resize((img.width * zoom, img.height * zoom), Image.LANCZOS)
        d = ImageDraw.Draw(im)
        Z = lambda *v: [c * zoom for c in v]
        if ov.get("box"):
            d.rectangle(Z(*ov["box"]), outline=(150, 150, 150))
        for ex in ov.get("exclude", []):
            d.rectangle(Z(*ex), outline=(255, 140, 0), width=2)
        for x, y, t in ov.get("x", []) + ov.get("y", []):          # tick labels where they really are
            d.ellipse(Z(x - 4, y - 4, x + 4, y + 4), outline=(0, 0, 255), width=2)
            d.text((x * zoom + 8, y * zoom - 14), str(t), fill=(0, 0, 255))
        for name, (pts, vals, method) in ov.get("series", {}).items():
            col = (0, 160, 0) if str(method).startswith("color") else (220, 0, 0)
            for (x, y), (xv, yv) in zip(pts, vals):
                d.ellipse(Z(x - 4, y - 4, x + 4, y + 4), outline=col, width=2)
                d.text((x * zoom + 7, y * zoom + 4), f"{yv:g}", fill=col)
        base = os.path.splitext(os.path.basename(image_path))[0]
        im.save(os.path.join(debug_dir, f"{base}_grounded.png"))