import copy
import difflib
import json
import Levenshtein
import math
import numpy as np
import os
import re
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Set, Tuple
from PIL import Image, ImageDraw, ImageFont, ImageOps
from pydantic import BaseModel, Field
from scipy import ndimage

from xlmexlab.prompt import PromptFormatter

class Evaluator_Graphs(BaseModel):
    reference_data: Dict[str, Any]
    threshold: float = Field(default=0.999)
    distance_threshold: float = Field(default=0.1)
    all_label_matches: ClassVar[List[Tuple[str, str, str, float]]] = []
    all_series_matches: ClassVar[List[Tuple[str, str, str, float]]] = []
    unmatched_items: List[Tuple[str, str, str, str]] = []



    @staticmethod
    def normalize_text(text: str) -> str:
        return text.lower().replace(" ", "")

    def evaluate(self, tp: int, fp: int, fn: int) -> Dict[str, float]:

        if tp + fp == 0:
            #ND = 0.0
            CoWP = 0.0
        else:
            #ND = fp / (tp + fp)
            CoWP = tp / (tp + fp)
        if tp + fn == 0:
            #MD = 0.0
            CoSP = 0.0
        else:
            #MD = fn / (tp + fn)
            CoSP = tp / (tp + fn)
        if tp + fn + fp == 0:
            #ID = 1.0
            CD = 1.0
        else:
            #ID = (fn + fp) / (tp + fn + fp)
            CD = tp / (tp + fn + fp)
        #return {"New detections": ND, "Miss detections": MD, "Incorrect detections": ID}
        return {"Correct detections over Should have been Predicted": CoSP, "Correct detections over what Was Predicted": CoWP,"Correct Detections": CD}
    @staticmethod
    def load_json(file_path: str) -> Dict[str, Any]:
        with open(file_path, 'r') as file:
            return json.load(file)

    @staticmethod
    def extract_labels(data: Dict[str, Any]) -> List[str]:
        labels = []
        for plot_data in data.values():
            for data_values in plot_data.values():
                for key in data_values.keys():
                    norm_key = Evaluator_Graphs.normalize_text(key)
                    if norm_key not in labels:
                        labels.append(norm_key)
        return labels

    @staticmethod
    def extract_series(data: Dict[str, Any]) -> List[str]:
        series = []
        for plot_data in data.values():
            for key in plot_data.keys():
                norm_key = Evaluator_Graphs.normalize_text(key)
                if norm_key not in series:
                    series.append(norm_key)
        return series

    def match_references_tests(
        self, references: List[str], tests: List[str]
    ) -> Tuple[int, int, int, List[Tuple[str, str, float]], Set[int], Set[int]]:

        match_candidates = []
        for ref_idx, ref in enumerate(references):
            for test_idx, test in enumerate(tests):
                ratio = Levenshtein.ratio(ref, test)
                if ratio >= self.threshold:
                    match_candidates.append((ratio, ref_idx, test_idx, ref, test))

        match_candidates.sort(reverse=True)

        matched_refs = set()
        matched_tests = set()
        matches = []

        for ratio, ref_idx, test_idx, ref, test in match_candidates:
            if ref_idx not in matched_refs and test_idx not in matched_tests:
                matched_refs.add(ref_idx)
                matched_tests.add(test_idx)
                matches.append((ref, test, ratio))

        TP = len(matches)
        FN = len(references) - TP
        FP = len(tests) - len(matched_tests)

        return TP, FP, FN, matches, matched_refs, matched_tests

    @staticmethod
    def euclidean_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5


    def overall_point_matching(self, ref_data: Dict[str, Any], test_data: Dict[str, Any]) -> Tuple[int, int, int]:
        ref_points = []
        test_points = []

        # Collect all points from all series
        for plot_data in ref_data.values():
            for series_data in plot_data.values():
                x_vals, y_vals = list(series_data.values())
                ref_points.extend(list(zip(x_vals, y_vals)))

        for plot_data in test_data.values():
            for series_data in plot_data.values():
                x_vals, y_vals = list(series_data.values())
                test_points.extend(list(zip(x_vals, y_vals)))

        if not ref_points and not test_points:
            return 0, 0, 0

        # Normalize based on global scale
        all_x = [p[0] for p in ref_points + test_points]
        all_y = [p[1] for p in ref_points + test_points]
        max_x = abs(max(all_x, key=abs, default=1.0)) or 1.0
        max_y = abs(max(all_y, key=abs, default=1.0)) or 1.0
        ref_points = [(x / max_x, y / max_y) for x, y in ref_points]
        test_points = [(x / max_x, y / max_y) for x, y in test_points]

        TP, FP, FN = 0, 0, len(ref_points)
        test_points_used = set()

        for i, ref_pt in enumerate(ref_points):
            for j, test_pt in enumerate(test_points):
                if j in test_points_used:
                    continue
                dist = self.euclidean_distance(ref_pt, test_pt)
                if dist <= self.distance_threshold:
                    TP += 1
                    FN -= 1
                    test_points_used.add(j)
                    break

        FP = len(test_points) - len(test_points_used)
        return TP, FP, FN


    def point_matching_accuracy(
        self,
        ref_data: Dict[str, Any],
        test_data: Dict[str, Any],
        matches: List[Tuple[str, str, float]],
        matched_ref_series: Set[int],
        matched_test_series: Set[int],
    ) -> Tuple[int, int, int, Dict[str, Any]]: # add

        overall_TP = 0
        overall_FP = 0
        overall_FN = 0
        per_image_points = {} #add

        #add
        for plot_name in ref_data.keys() | test_data.keys():
            per_image_points[plot_name] = {
                "TP": [], 
                "FP": {
                    "matched_series":[],
                    "unmatched_series":[]
                }, 
                "FN":{
                    "matched_series":[],
                    "unmatched_series":[]
                }
            }


        for ref_series, test_series, _ in matches:
            print(f"\nMatching series '{ref_series}' ↔ '{test_series}'")
            ref_points = []
            test_points = []

            ref_series_norm = self.normalize_text(ref_series)
            test_series_norm = self.normalize_text(test_series)

            

            for plot_name, plot_data in ref_data.items():
                for key, value in plot_data.items():
                    if self.normalize_text(key) == ref_series_norm:
                        ref_values = list(value.values())
                        ref_x = ref_values[0]
                        ref_y = ref_values[1]
                        ref_points = list(zip(ref_x, ref_y))
                        break

            for plot_name, plot_data in test_data.items():
                for key, value in plot_data.items():
                    if self.normalize_text(key) == test_series_norm:
                        test_values = list(value.values())
                        test_x = test_values[0]
                        test_y = test_values[1]
                        test_points = list(zip(test_x, test_y))
                        break

            print(f"Ref Points ({len(ref_points)}): {ref_points}")
            print(f"Test Points ({len(test_points)}): {test_points}")

            max_x_value = abs(max(ref_x + test_x, key=abs, default=1.0))
            max_y_value = abs(max(ref_y + test_y, key=abs, default=1.0))

            # Avoid dividing by zero (just in case)
            max_x_value = max_x_value if max_x_value != 0 else 1.0
            max_y_value = max_y_value if max_y_value != 0 else 1.0

            ref_points_raw = list(zip(ref_x, ref_y))
            test_points_raw = list(zip(test_x, test_y))

            ref_points = [(x / max_x_value, y / max_y_value) for x, y in ref_points_raw]
            test_points = [(x / max_x_value, y / max_y_value) for x, y in test_points_raw]

            FN = len(ref_points_raw)
            TP = 0

            while ref_points:
                ref_point = ref_points.pop(0)
                ref_point_raw = ref_points_raw.pop(0)
                closest_distance = float('inf')
                closest_test_point = None
                closest_test_index = None

                for test_index, test_point in enumerate(test_points):
                    distance = self.euclidean_distance(ref_point, test_point)
                    if distance < closest_distance and distance <= self.distance_threshold:
                        closest_distance = distance
                        closest_test_point = test_point
                        closest_test_index = test_index

                if closest_test_point is not None:
                    print(f"Matched point {ref_point} ↔ {closest_test_point} (distance={closest_distance:.4f})")
                    test_point_raw = test_points_raw.pop(closest_test_index)
                    test_points.pop(closest_test_index)
                    FN -= 1
                    TP += 1
                    per_image_points[plot_name]["TP"].append(
                        {"ref_point":ref_point_raw,
                        "test_point":test_point_raw, 
                        "distance":closest_distance}
                    )
                else:
                    print(f"Unmatched reference point: {ref_point}")
                    per_image_points[plot_name]["FN"]["matched_series"].append({"ref_point": ref_point_raw})

            for leftover in test_points_raw:
                per_image_points[plot_name]["FP"]["matched_series"].append({"test_point": leftover})


            FP = len(test_points_raw)
            print(f"Remaining test points (FP): {test_points}")
            print(f"Series summary: TP={TP}, FP={FP}, FN={FN}")

            overall_TP += TP
            overall_FP += FP
            overall_FN += FN

        for plot_name, plot_data in ref_data.items():
            for ref_idx, ref_series in enumerate(plot_data.keys()):
                if ref_idx not in matched_ref_series:
                    print(f"Unmatched reference series: {ref_series}")
                    ref_values = list(plot_data[ref_series].values())
                    ref_points = list(zip(ref_values[0], ref_values[1]))
                    print(f"  Adding FN for {len(ref_points)} unmatched reference points")
                    ref_x, ref_y = list(plot_data[ref_series].values())
                    ref_points_raw = list(zip(ref_x, ref_y))
                    for pt in ref_points_raw:
                        per_image_points[plot_name]["FN"]["unmatched_series"].append({"ref_point": pt})
                    overall_FN += len(ref_points_raw)

        for plot_name, plot_data in test_data.items():
            for test_idx, test_series in enumerate(plot_data.keys()):
                if test_idx not in matched_test_series:
                    print(f"Unmatched test series: {test_series}")
                    test_values = list(plot_data[test_series].values())
                    test_points = list(zip(test_values[0], test_values[1]))
                    print(f"  Adding FP for {len(test_points)} unmatched test points")
                    test_x, test_y = list(plot_data[test_series].values())
                    test_points_raw = list(zip(test_x, test_y))
                    for pt in test_points_raw:
                        per_image_points[plot_name]["FP"]["unmatched_series"].append({"test_point": pt})
                    overall_FP += len(test_points_raw)

        print(f"\nPoint Matching Totals: TP={overall_TP}, FP={overall_FP}, FN={overall_FN}")
        return overall_TP, overall_FP, overall_FN, per_image_points
            
    

    def process_plots(self, test_data: Dict[str, Any]) -> Dict[str, int]:
        total_point_TP, total_point_FP, total_point_FN = 0, 0, 0
        total_label_TP, total_label_FP, total_label_FN = 0, 0, 0
        total_series_TP, total_series_FP, total_series_FN = 0, 0, 0

        skipped_images = 0
        total_images = len(self.reference_data)

        all_label_matches = []
        all_series_matches = []
        unmatched_items = []

        per_image_results = {}

        for plot_name in self.reference_data.keys():
            test_plot = test_data.get(plot_name)

            if not test_plot:
                print(f"Skipping image '{plot_name}' – no data in test set.")
                skipped_images += 1
                continue

            ref_plot = self.reference_data[plot_name]

            ref_labels = self.extract_labels({plot_name: ref_plot})
            test_labels = self.extract_labels({plot_name: test_plot})

            label_TP, label_FP, label_FN, label_matches, matched_ref_labels, matched_test_labels = self.match_references_tests(
                ref_labels,
                test_labels
            )

            print(f"{plot_name} - Label Matching: TP: {label_TP}, FP: {label_FP}, FN: {label_FN}")
            if label_matches:
                print(f"{plot_name} - Matched Labels:")
                for ref_l, test_l, ratio in label_matches:
                    print(f"  '{ref_l}' ↔ '{test_l}' (Similarity: {ratio:.3f})")
                    all_label_matches.extend([(plot_name, ref_l, test_l, ratio) for ref_l, test_l, ratio in label_matches])

            unmatched_ref_labels = [label for idx, label in enumerate(ref_labels) if idx not in matched_ref_labels]
            if unmatched_ref_labels:
                print(f"{plot_name} - Unmatched Reference Labels:")
                for label in unmatched_ref_labels:
                    print(f"  '{label}'")
                    unmatched_items.append((plot_name, label, "ref", "label"))

            unmatched_test_labels = [label for idx, label in enumerate(test_labels) if idx not in matched_test_labels]
            if unmatched_test_labels:
                print(f"{plot_name} - Unmatched Test Labels:")
                for label in unmatched_test_labels:
                    print(f"  '{label}'")
                    unmatched_items.append((plot_name, label, "test", "label"))
     
            total_label_TP += label_TP
            total_label_FP += label_FP
            total_label_FN += label_FN

            ref_series = self.extract_series({plot_name: ref_plot})
            test_series = self.extract_series({plot_name: test_plot})
            series_TP, series_FP, series_FN, series_matches, matched_ref_series, matched_test_series = self.match_references_tests(
                ref_series,
                test_series
            )
            print(f"{plot_name} - Series Matching: TP: {series_TP}, FP: {series_FP}, FN: {series_FN}")
            print(f"{plot_name} - Matched Series:")
            
            

            for ref_s, test_s, ratio in series_matches:
                print(f"  '{ref_s}' ↔ '{test_s}' (Similarity: {ratio:.3f})")
                all_series_matches.extend([(plot_name, ref_s, test_s, ratio) for ref_s, test_s, ratio in series_matches])

            unmatched_refs = [s for i, s in enumerate(ref_series) if i not in matched_ref_series]
            for s in unmatched_refs:
                print(f"{plot_name} - Unmatched Reference Series: '{s}'")
                unmatched_items.append((plot_name, s, "ref", "series"))

            unmatched_tests = [s for i, s in enumerate(test_series) if i not in matched_test_series]
            for s in unmatched_tests:
                print(f"{plot_name} - Unmatched Test Series: '{s}'")
                unmatched_items.append((plot_name, s, "test", "series"))


            total_series_TP += series_TP
            total_series_FP += series_FP
            total_series_FN += series_FN

            point_TP, point_FP, point_FN, per_image_points = self.point_matching_accuracy(
                {plot_name: ref_plot},
                {plot_name: test_plot},
                series_matches,
                matched_ref_series,
                matched_test_series
            )

            per_image_results[plot_name] = per_image_points.get(plot_name, {})
            print(f"{plot_name} - Point Matching: TP: {point_TP}, FP: {point_FP}, FN: {point_FN}")
            total_point_TP += point_TP
            total_point_FP += point_FP
            total_point_FN += point_FN

        print(f"\nSkipped Images: {skipped_images}/{total_images} ({(skipped_images / total_images) * 100:.2f}%)")
        print(f"Total Label Matching - TP: {total_label_TP}, FP: {total_label_FP}, FN: {total_label_FN}")
        print(f"Total Series Matching - TP: {total_series_TP}, FP: {total_series_FP}, FN: {total_series_FN}")
        print(f"Total Point Matching - TP: {total_point_TP}, FP: {total_point_FP}, FN: {total_point_FN}")

        return {
            "Label_TP": total_label_TP,
            "Label_FP": total_label_FP,
            "Label_FN": total_label_FN,
            "Series_TP": total_series_TP,
            "Series_FP": total_series_FP,
            "Series_FN": total_series_FN,
            "Point_TP": total_point_TP,
            "Point_FP": total_point_FP,
            "Point_FN": total_point_FN,
            "Skipped_Images": skipped_images,
            "Total_Images": total_images,
            "Skipped_Percent": round((skipped_images / total_images) * 100, 2),
            "Matched_Labels": all_label_matches,
            "Matched_Series": all_series_matches,
            "Unmatched_Items": unmatched_items,
            "Per_Image_Results": per_image_results
        }


"""
Grid-assisted direct extraction with a small VLM.

Idea: small VLMs read a value much better when there is a labelled line next to
the point than when they must interpolate between two far-apart tick labels.
So, after calibrating the axes (in code), we draw a dense grid IN DATA UNITS
and print the value of every major line at the image border, then ask the VLM
to read the series.

Rules that make the grid help instead of hurt
- non-data ink is painted white first (see "annotation filter" below), so neither
  the digitizer nor the VLM can mistake an annotation for a data point;
- the grid is painted only on background pixels, so markers, lines and error
  bars are never covered;
- gridlines are neutral gray (never a series colour), minor lines lighter;
- labels go in a white margin outside the figure, never on top of the data;
- the image is upscaled first (small figures are the main source of error);
- ask ONE series per call (or one x position per call), never "all series".

The annotation filter removes, inside the plot area only:
- significance brackets (thin isolated horizontal rules + the legs at their ends);
- '*', '**', '***', '#', 'n.s.', 'p<0.05';
- rows of identical glyphs at a constant y (a '+ + + +' row is an annotation,
  not a flat series: a flat series is drawn with a connecting line);
- tall solid coloured glyphs such as dosing arrows, which otherwise steal a
  series colour and become its "data points".

Typical use (after GraphGrounder has calibrated the axes):
    g = draw_value_grid(image_path, out_path, x_map, y_map, box)
    prompt_dict = build_grid_read_prompt(series, g)
    raw = vlm.run_image_single_prompt_rescale(prompt, out_path)
    pts = parse_grid_answer(raw)

or, end to end, grid_extract(grounder, image_path, stage1, out_dir).
"""
import math
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

FONT_PATHS = ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "arial.ttf"]

Box = Tuple[float, float, float, float]


def _font(size: int):
    for p in FONT_PATHS:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


# ═════════════════════════ annotation filter ═════════════════════════
# Non-data ink. Everything here works on the ORIGINAL figure, before the value
# grid is drawn: run it on a gridded image and the drawn gridlines look like
# brackets.
# token texts that are never data
SIG_TEXT_RX = re.compile(
    r"^(?:"
    r"[*∗٭⁎#†‡§]{1,6}"      # * ** *** # ## dagger
    r"|n\.?\s*s\.?"                                       # n.s.  ns
    r"|p\s*[<>=≤≥]\s*0?[.,]\d+"                 # p<0.05
    r")$", re.I)

PLUS_TEXT_RX = re.compile(r"^[+†]{1,3}$")
PLUS_MARKERS = ("plus", "cross", "+", "x", "star", "asterisk")


# ───────────────────────── low level ─────────────────────────
def _dark(arr: np.ndarray, dark: int = 135, max_chroma: int = 45) -> np.ndarray:
    """Dark and (nearly) achromatic pixels: ink that is not a coloured series."""
    a = arr.astype(int)
    return (a.mean(2) < dark) & ((a.max(2) - a.min(2)) <= max_chroma)


def _box_mask(shape, box: Box, shrink: float = 0.0) -> np.ndarray:
    h, w = shape[:2]
    x1, y1, x2, y2 = box
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    m = np.zeros((h, w), bool)
    m[max(0, int(y1 + shrink)):min(h, int(y2 - shrink) + 1),
      max(0, int(x1 + shrink)):min(w, int(x2 - shrink) + 1)] = True
    return m


def _run_thickness(d: np.ndarray, axis: int, win: int) -> np.ndarray:
    """Number of dark pixels in a window of length `win` along `axis`, per pixel."""
    k = ndimage.uniform_filter1d(d.astype(np.float32), size=win, axis=axis, mode="constant")
    return k * win


# ───────────────────────── brackets ─────────────────────────
def _runs_1d(flags: np.ndarray, max_gap: int = 1) -> List[Tuple[int, int]]:
    idx = np.flatnonzero(flags)
    if len(idx) == 0:
        return []
    out, s, p = [], int(idx[0]), int(idx[0])
    for i in idx[1:]:
        if i - p > max_gap + 1:
            out.append((s, p))
            s = int(i)
        p = int(i)
    out.append((s, p))
    return out


def _free_sides(d: np.ndarray, leg: Sequence[int], x: int, gap: int = 3,
                frac: float = 0.75) -> bool:
    """True when the column `x` is an isolated vertical hairline along `leg`.

    A bracket leg has white on both sides. A steep data curve does not: it keeps
    going sideways, so `x - gap` or `x + gap` is ink for most of the run. Without
    this test a near-vertical black curve is read as a bracket leg and erased.
    """
    w = d.shape[1]
    free = 0
    for yy in leg:
        l_ = x - gap
        r_ = x + gap
        if (l_ < 0 or not d[yy, l_]) and (r_ >= w or not d[yy, r_]):
            free += 1
    return free >= frac * len(leg)


def bracket_mask(d: np.ndarray, inside: np.ndarray, min_len_px: int, max_len_px: int,
                 max_thick: int = 4, edge: int = 6, max_leg_frac: float = 0.35) -> np.ndarray:
    """Significance brackets: an isolated thin horizontal rule plus the short legs at its ends.

    Run-based on purpose. An earlier connected-component version grew through every
    gridline the bracket legs crossed and wiped the whole grid.

    A gridline is rejected because the dark coverage of its ROW exceeds `max_len_px`,
    which stays true when bars chop the gridline into bracket-sized pieces.
    An error-bar cap is rejected by `min_len_px`, a bar top by it as well.
    """
    out = np.zeros_like(d)
    di = d & inside
    if not di.any():
        return out
    rows_in = np.flatnonzero(inside.any(1))
    if len(rows_in) < 3:
        return out
    y_lo, y_hi = int(rows_in[0]) + edge, int(rows_in[-1]) - edge
    plot_h = int(rows_in[-1] - rows_in[0])
    max_leg = max(4, int(max_leg_frac * plot_h))

    thin_v = _run_thickness(di, 0, 2 * max_thick + 3) <= max_thick
    cand = di & thin_v

    for y in range(max(0, y_lo), min(d.shape[0], y_hi + 1)):
        row = cand[y]
        if not row.any():
            continue
        if int(di[y].sum()) > max_len_px:          # gridline / frame / reference line row
            continue
        for a, b in _runs_1d(row, max_gap=1):
            if not (min_len_px <= b - a + 1 <= max_len_px):
                continue
            out[y, a:b + 1] = True
            cols_in = np.flatnonzero(inside[y])
            x_lo, x_hi = int(cols_in[0]) + edge, int(cols_in[-1]) - edge
            for x0 in (a, b):                       # the legs, at the two ends only
                if not (x_lo <= x0 <= x_hi):        # at the frame: that is an axis, not a leg
                    continue
                for x in range(max(0, x0 - 2), min(d.shape[1], x0 + 3)):
                    for step in (1, -1):
                        leg, yy = [], y + step
                        while 0 <= yy < d.shape[0] and inside[yy, x] and d[yy, x] \
                                and len(leg) <= max_leg:
                            leg.append(yy)
                            yy += step
                        if not (4 <= len(leg) <= max_leg):   # a leg ends; an axis does not
                            continue
                        if _free_sides(d, leg, x):           # a hairline, not a steep curve
                            out[leg, x] = True
    return out


# ───────────────────────── glyphs ─────────────────────────
def _blobs(d: np.ndarray, inside: np.ndarray, max_h: int, max_w: int) -> List[Tuple[slice, slice]]:
    lab, n = ndimage.label(d & inside, structure=np.ones((3, 3), bool))
    out = []
    for sl in ndimage.find_objects(lab):
        if sl is None:
            continue
        h_, w_ = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if 2 <= h_ <= max_h and 2 <= w_ <= max_w:
            out.append(sl)
    return out


def glyphs_above_brackets(d: np.ndarray, inside: np.ndarray, brackets: np.ndarray,
                          text_h: float) -> np.ndarray:
    """Small blobs within ~2 text heights ABOVE a bracket: the `***` that labels it."""
    out = np.zeros_like(d)
    if not brackets.any():
        return out
    reach = int(max(6, 2.2 * text_h))
    # a pixel is "under consideration" if a bracket pixel exists below it within `reach`
    below = np.zeros_like(brackets)
    for s in range(1, reach + 1):
        below[:-s] |= brackets[s:]
    for sl in _blobs(d, inside, max_h=int(2.0 * text_h) + 2, max_w=int(3.0 * text_h) + 4):
        if below[sl].any():
            out[sl] |= d[sl]
    return out


def token_glyphs(shape, tokens: Sequence[Any], inside: np.ndarray,
                 drop_plus: bool) -> np.ndarray:
    """Pixels of OCR tokens whose TEXT is a significance mark (and `+` when allowed)."""
    out = np.zeros(shape[:2], bool)
    for t in tokens or []:
        txt = str(getattr(t, "text", "")).strip()
        if not txt:
            continue
        hit = bool(SIG_TEXT_RX.match(txt)) or (drop_plus and bool(PLUS_TEXT_RX.match(txt)))
        if not hit:
            continue
        x1, y1 = int(getattr(t, "x1", 0)), int(getattr(t, "y1", 0))
        x2, y2 = int(getattr(t, "x2", 0)), int(getattr(t, "y2", 0))
        pad = 1
        sl = (slice(max(0, y1 - pad), y2 + pad), slice(max(0, x1 - pad), x2 + pad))
        if inside[sl].any():
            out[sl] = True
    return out


def _same_shape(d: np.ndarray, boxes: Sequence[Tuple[slice, slice]],
                min_iou: float = 0.72, grid: int = 9) -> bool:
    """Do these blobs draw the SAME glyph? Each patch is rasterised onto a small
    square and compared by IoU with the median patch. A row of '+' passes; a row of
    tick labels ('10', '12', '14' ... one blob per digit) does not."""
    pats = []
    for sl in boxes:
        patch = d[sl]
        if patch.size == 0:
            return False
        ys = np.linspace(0, patch.shape[0] - 1, grid).round().astype(int)
        xs = np.linspace(0, patch.shape[1] - 1, grid).round().astype(int)
        pats.append(patch[np.ix_(ys, xs)].astype(np.float32))
    ref = np.median(np.stack(pats), axis=0) >= 0.5
    if not ref.any():
        return False
    ious = []
    for p in pats:
        b = p >= 0.5
        inter = float((b & ref).sum())
        union = float((b | ref).sum()) or 1.0
        ious.append(inter / union)
    return float(np.mean(ious)) >= min_iou


def annotation_rows(d: np.ndarray, inside: np.ndarray, text_h: float,
                    min_count: int = 4, min_span: float = 0.35,
                    y_tol: float = 3.0, size_tol: int = 2) -> np.ndarray:
    """A row of IDENTICAL small glyphs at a constant y, spread across the plot.

    This is the ``+  +  +  +  +`` row of Fig. 4B: an annotation, not a series.
    A real series that happened to be flat is drawn with a connecting line, so its
    markers are one wide connected component, not many separate equal blobs.

    The search is done per glyph SIZE class first, so a row of equal glyphs is not
    diluted by the ordinary text and data labels that share its y band.
    """
    out = np.zeros_like(d)
    cols = np.flatnonzero(inside.any(0))
    if len(cols) < 2:
        return out
    width = cols[-1] - cols[0]
    lim = int(1.8 * text_h) + 2
    boxes = _blobs(d, inside, max_h=lim, max_w=lim)
    if len(boxes) < min_count:
        return out

    hs = np.array([sl[0].stop - sl[0].start for sl in boxes])
    ws = np.array([sl[1].stop - sl[1].start for sl in boxes])
    cy = np.array([(sl[0].start + sl[0].stop) / 2.0 for sl in boxes])
    cx = np.array([(sl[1].start + sl[1].stop) / 2.0 for sl in boxes])

    used = np.zeros(len(boxes), bool)
    for i in np.argsort(-(hs * ws)):                       # biggest glyph first
        if used[i] or hs[i] < 4 or ws[i] < 4:
            continue
        same = (np.abs(hs - hs[i]) <= size_tol) & (np.abs(ws - ws[i]) <= size_tol) & ~used
        for j in np.flatnonzero(same):
            row = same & (np.abs(cy - cy[j]) <= max(y_tol, 0.3 * text_h))
            k = np.flatnonzero(row)
            if len(k) < min_count:
                continue
            if (cx[k].max() - cx[k].min()) < min_span * width:
                continue
            if not _same_shape(d, [boxes[m] for m in k]):
                continue                 # digits of a tick-label row are the same SIZE but
            for m in k:                  # not the same SHAPE; a row of '+' is both
                out[boxes[m]] |= d[boxes[m]]
                used[m] = True
            break
    return out


# ───────────────────────── public API ─────────────────────────
def clean_chart(arr: np.ndarray, box: Box, exclude: Sequence[Box] = (),
                tokens: Sequence[Any] = (), stage1: Optional[dict] = None,
                text_h: float = 11.0, margin: int = 3,
                min_bracket_frac: float = 0.06,
                max_bracket_frac: float = 0.75,
                glyph_dark: int = 135, line_dark: int = 190,
                drop_blobs: bool = True,
                zero_px: Optional[float] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Return (image with the annotations painted white, what was removed).

    box     : plot area in pixels, as GraphGrounder computes it
    exclude : legend / inset boxes (left untouched, the legend must stay readable)
    tokens  : OCR tokens (``GraphGrounder._ocr_tokens``); optional but much safer
    stage1  : used only to decide whether `+` is a real marker of some series
    """
    arr = np.asarray(arr)
    inside = _box_mask(arr.shape, box, shrink=margin)
    for ex in exclude or []:
        inside &= ~_box_mask(arr.shape, ex)
    # glyphs are printed black; the bracket rules themselves are often thin gray
    d = _dark(arr, dark=glyph_dark)
    d_line = _dark(arr, dark=line_dark, max_chroma=40)

    x1, _, x2, _ = box
    min_len = int(max(10, min_bracket_frac * abs(x2 - x1)))
    max_len = int(max_bracket_frac * abs(x2 - x1))

    markers = " ".join(str((s or {}).get("marker", "")).lower()
                       for s in ((stage1 or {}).get("series") or []))
    drop_plus = not any(p in markers for p in PLUS_MARKERS)
    if any(str((s or {}).get("marker", "")).strip().lower() in ("none", "", "unknown", "-")
           and str((s or {}).get("line", "")).strip().lower() in ("none", "", "unknown", "-")
           for s in ((stage1 or {}).get("series") or [])):
        drop_blobs = False                 # a bar chart: a bar is tall and solid too

    br = bracket_mask(d_line, inside, min_len_px=min_len, max_len_px=max_len)
    gl = glyphs_above_brackets(d, inside, br, text_h)
    tk = token_glyphs(arr.shape, tokens, inside, drop_plus)
    # a row of tick labels sits just outside the plot and looks exactly like an
    # annotation row, so glyph rows are searched in the eroded interior only
    pad = int(max(4, 1.2 * text_h))
    core = ndimage.binary_erosion(inside, np.ones((2 * pad + 1, 1), bool))
    rw = annotation_rows(d, core, text_h) if drop_plus else np.zeros_like(d)

    bl = solid_blob_mask(arr, inside, zero_px=zero_px) if drop_blobs \
        else np.zeros(arr.shape[:2], bool)

    kill = (br | gl | tk | rw) & inside
    kill = ndimage.binary_dilation(kill, np.ones((3, 3), bool)) & _dark(arr, dark=200, max_chroma=60) & inside
    # coloured blobs are erased with their anti-aliased rim, not through the dark test
    kill |= ndimage.binary_dilation(bl, np.ones((3, 3), bool)) & inside

    out = arr.copy()
    out[kill] = 255
    info = {"brackets_px": int(br.sum()), "glyphs_px": int(gl.sum()),
            "token_px": int(tk.sum()), "row_px": int(rw.sum()),
            "blob_px": int(bl.sum()),
            "total_px": int(kill.sum()), "plus_removed": bool(drop_plus)}
    return out, info


def solid_blob_mask(arr: np.ndarray, inside: np.ndarray, min_chroma: int = 60,
                    min_aspect: float = 2.2, min_fill: float = 0.40,
                    min_h_frac: float = 0.06, zero_px: Optional[float] = None,
                    foot: float = 0.02) -> np.ndarray:
    """Tall solid coloured glyphs inside the plot: dosing arrows, pointers, big triangles.

    They are the worst kind of non-data ink, because they are the SAME colour family as a
    series. In 10.1016/j.actbio.2017.08.034 Fig. 4B the five blue dosing arrows take the
    'free Combo' colour, so the digitizer reads the arrows instead of the curve and
    `shared_x_positions` returns the arrow positions as the data x.

    A curve is thin: it fills ~5% of its bounding box. A marker is small and square.
    An arrow is tall (h >= 2.2 w), solid (fill >= 0.40) and a real fraction of the plot.

    Pass `zero_px` (and keep `drop_blobs` off for bar charts): a BAR is also tall and
    solid, but it stands on the baseline, so anything whose foot touches the baseline or
    the bottom of the plot is kept.
    """
    out = np.zeros(arr.shape[:2], bool)
    a = arr.astype(int)
    chrom = ((a.max(2) - a.min(2)) >= min_chroma) & inside
    if not chrom.any():
        return out
    rows_in = np.flatnonzero(inside.any(1))
    plot_h = float(rows_in[-1] - rows_in[0]) or 1.0
    bottom = float(rows_in[-1])
    lab, n = ndimage.label(chrom, structure=np.ones((3, 3), bool))
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        if sl is None:
            continue
        bh = sl[0].stop - sl[0].start
        bw = sl[1].stop - sl[1].start
        if bh < min_h_frac * plot_h or bw < 3:
            continue
        if bh < min_aspect * bw:
            continue
        m = (lab[sl] == i)
        if m.sum() < min_fill * bh * bw:
            continue
        base = sl[0].stop - 1
        if abs(base - bottom) <= foot * plot_h:              # a bar on the axis
            continue
        if zero_px is not None and abs(base - zero_px) <= foot * plot_h:
            continue
        out[sl] |= m
    return out


def text_height(tokens: Sequence[Any], default: float = 11.0) -> float:
    """Median height of the OCR tokens: the natural length unit of a figure."""
    hs = [float(getattr(t, "y2", 0)) - float(getattr(t, "y1", 0)) for t in tokens or []]
    hs = [h for h in hs if 4 <= h <= 60]
    return float(np.median(hs)) if hs else default


def achromatic_mask(arr: np.ndarray, region: np.ndarray,
                    other_masks: Sequence[np.ndarray] = (),
                    dark: int = 135, min_size: int = 6) -> np.ndarray:
    """Pixels of a BLACK / GRAY series: dark achromatic ink inside the plot, minus the
    axis lines, the frame, the text and every coloured series.

    Use it after `clean_chart`, so the significance ink is already gone. It lets the
    colour digitizer handle a black control series (PBS, Saline, Control) instead of
    falling back to the VLM, which is where most of the instability comes from.
    """
    d = _dark(arr, dark=dark) & region
    if not d.any():
        return d
    for m in other_masks or ():
        d &= ~ndimage.binary_dilation(np.asarray(m, bool), np.ones((3, 3), bool))
    # long straight runs = axis lines / frame / gridlines
    h, w = d.shape
    lines = ndimage.binary_opening(d, np.ones((1, max(25, w // 12)), bool)) | \
            ndimage.binary_opening(d, np.ones((max(25, h // 12), 1), bool))
    d &= ~ndimage.binary_dilation(lines, np.ones((3, 3), bool))
    lab, n = ndimage.label(d, structure=np.ones((3, 3), bool))
    if n:
        sizes = ndimage.sum(d, lab, range(1, n + 1))
        d &= np.isin(lab, np.flatnonzero(sizes >= min_size) + 1)
    return d

# ═════════════════════════ axis helpers ═════════════════════════
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


# ═════════════════════════ drawing ═════════════════════════
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


# ═════════════════════════ prompting ═════════════════════════
def build_grid_read_prompt(series: Dict[str, str], grid: Dict[str, Any],
                           x_values: Optional[Sequence[Any]] = None, faded: bool = False) -> dict:
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
    if faded:
        task += (" All other series are drawn very faint (almost white); read ONLY the strongly coloured "
                 "series. The legend is unchanged.")
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


# ═════════════════════════ fading the other series ═════════════════════════
def fade_other_series(image_path: str, palette: Sequence[Sequence[float]], keep_rgb: Optional[Sequence[float]],
                      box: Tuple[float, float, float, float], exclude: Sequence[Tuple[float, float, float, float]],
                      out_path: str, strength: float = 0.85, min_chroma: int = 25,
                      same_tol: float = 20.0) -> Optional[str]:
    """Copy of the figure where every COLOURED pixel of the plot area that does not belong to the
    target series is blended towards white. Axes, text, black error bars and the legend
    (excluded box) are untouched, so the VLM still sees the scale and which series it reads.
    Each coloured pixel goes to the nearest palette colour, measured to the segment
    colour -> white, so the anti-aliased (lighter) edges of lines and markers count too.
    palette  : all series colours detected in the plot (GraphGrounder.last_palette)
    keep_rgb : colour of the series being read (None for a black/gray series -> fade all colours)

    Palette entries within `same_tol` of keep_rgb are dropped before the comparison. Without
    that, a figure with two clusters of one hue (a blue curve and blue dosing arrows) fades
    the target series itself: its own pixels are nearer the OTHER cluster than to keep_rgb.
    """
    arr = np.asarray(Image.open(image_path).convert("RGB")).astype(float)
    h, w = arr.shape[:2]
    region = np.zeros((h, w), bool)
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    region[max(0, y1):min(h, y2 + 1), max(0, x1):min(w, x2 + 1)] = True
    for a, b, c, d in exclude or []:
        region[max(0, int(b)):min(h, int(d) + 1), max(0, int(a)):min(w, int(c) + 1)] = False
    coloured = region & ((arr.max(2) - arr.min(2)) >= min_chroma)
    if not coloured.any() or not palette:
        return None
    px = arr[coloured]                                    # (n, 3)

    def seg_dist(c):
        c = np.asarray(c, float)
        v = 255.0 - c
        t = np.clip(((px - c) @ v) / max(1e-6, float(v @ v)), 0, 0.9)
        return np.linalg.norm(px - (c + t[:, None] * v), axis=1)

    if keep_rgb is None:
        others = list(palette)
    else:
        k = np.asarray(keep_rgb, float)
        others = [c for c in palette if np.linalg.norm(np.asarray(c, float) - k) > same_tol]
    if not others:
        return None
    d_all = np.stack([seg_dist(c) for c in others], axis=1)
    if keep_rgb is None:
        fade_px = np.ones(len(px), bool)
    else:
        d_keep = seg_dist(keep_rgb)
        fade_px = d_all.min(axis=1) < d_keep - 1e-6       # another series' colour is nearer
    if not fade_px.any():
        return None
    fade = np.zeros((h, w), bool)
    fade[coloured] = fade_px
    arr[fade] = arr[fade] + (255.0 - arr[fade]) * strength
    Image.fromarray(arr.round().astype(np.uint8)).save(out_path)
    return out_path


# ═════════════════════════ pipeline ═════════════════════════
def _zero_pixel(y_map) -> Optional[float]:
    try:
        if y_map is None or getattr(y_map, "log", False):
            return None
        return float(y_map.pixel(0.0))
    except Exception:
        return None


def grid_extract(grounder, image_path: str, stage1: dict, out_dir: str, scale: float = 1.0,
                 ask_at_data_x: bool = True, fade_others: bool = True,
                 clean_annotations: bool = True) -> Dict[str, Any]:
    """Calibrate with GraphGrounder, remove the non-data ink, draw the value grid, ask the
    VLM ONE series per call, and compare with the pixel result.

    grounder : graph_grounder.GraphGrounder (its VLM and prompt template are reused)
    clean_annotations : after the first calibration pass the plot box is known, so the
                  annotations can be painted white and the digitizer RE-RUN on the clean
                  image (<base>_clean.png). This is what stops a dosing arrow from being
                  read as a series and poisoning `data_x`.
    fade_others : one grid image PER SERIES in which the other coloured series are faded
                  (<base>_grid_s<i>.png); the plain <base>_grid.png is kept for the overlay.
    Returns {"grid": {series: [[x, y], ...]}, "pixel": {...}, "agreement": {...}, "grid_image": path}
    """
    from xlmexlab.prompt import PromptFormatter

    try:
        from .graph_grounder import normalize_stage1
    except ImportError:
        from graph_grounder import normalize_stage1
    stage1 = normalize_stage1(stage1)

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]

    # calibration only: debug_dir=None, so C's *_grounded.png is NOT written into D's folder
    pixel = grounder.extract(image_path, stage1, scale=scale, debug_dir=None)
    cal = getattr(grounder, "last_calibration", {}) or {}

    # ── remove the non-data ink and read the pixels again on the clean image ──
    src, ann = image_path, None
    if clean_annotations and cal.get("box"):
        tokens = getattr(grounder, "_ocr_tokens", ()) or ()
        arr = np.asarray(Image.open(image_path).convert("RGB"))
        cleaned, ann = clean_chart(arr, cal["box"], exclude=cal.get("exclude", ()),
                                   tokens=tokens, stage1=stage1,
                                   text_h=text_height(tokens),
                                   zero_px=_zero_pixel(cal.get("y_map")))
        print(f"  [grid_extract] annotation filter: {ann}")
        if ann["total_px"] > 0:
            src = os.path.join(out_dir, f"{base}_clean.png")
            Image.fromarray(cleaned).save(src)
            pixel = grounder.extract(src, stage1, scale=scale, debug_dir=None)
            cal = getattr(grounder, "last_calibration", {}) or cal

    stage1 = getattr(grounder, "last_stage1", None) or stage1    # colours read from the legend symbols
    palette = getattr(grounder, "last_palette", []) or []
    series_rgb = getattr(grounder, "last_series_rgb", {}) or {}
    if not cal.get("y_map"):
        return {"grid": {}, "pixel": pixel, "agreement": {}, "error": "axis calibration failed",
                "annotations": ann}

    # ── the x positions to ask at, if the pixel pass found enough of them ──
    data_x = cal.get("data_x") if ask_at_data_x else None
    n_ticks = len(cal.get("x_ticks_px") or [])
    if data_x and len(data_x) < max(3, 0.8 * n_ticks):
        # asking at 2 positions when the axis has 6 ticks caps EVERY series at 2 points
        print(f"  [grid_extract] only {len(data_x)} pixel x positions for {n_ticks} x ticks "
              f"-> reading the series open-ended instead")
        data_x = None

    grid_path = os.path.join(out_dir, f"{base}_grid.png")
    g = draw_value_grid(src, grid_path, cal.get("x_map"), cal["y_map"], cal["box"],
                        exclude=cal.get("exclude", ()), x_positions=data_x or None)

    out: Dict[str, Any] = {}
    faded_paths: Dict[str, str] = {}
    for i, s in enumerate(stage1.get("series", [])):
        name = s.get("name", "UNKNOWN")
        ask_path, faded = grid_path, False
        if fade_others and len(stage1.get("series", [])) > 1 and palette:
            tmp = os.path.join(out_dir, f"{base}_faded_s{i}.png")
            if fade_other_series(src, palette, series_rgb.get(name), cal["box"],
                                 cal.get("exclude", ()), tmp):
                ask_path = os.path.join(out_dir, f"{base}_grid_s{i}.png")
                draw_value_grid(tmp, ask_path, cal.get("x_map"), cal["y_map"], cal["box"],
                                exclude=cal.get("exclude", ()), x_positions=data_x or None)
                os.remove(tmp)
                faded, faded_paths[name] = True, ask_path
        fm = PromptFormatter(**build_grid_read_prompt(s, g, data_x or None, faded=faded))
        fm.model_post_init(grounder.template)
        raw = grounder.vlm.run_image_single_prompt_rescale(fm.format_prompt("<image>"), ask_path, scale=1.0)
        print(f"\n  [grid_extract] {s.get('name')} RAW:\n{raw}")
        out[s.get("name", "UNKNOWN")] = parse_grid_answer(raw)

    read_path = os.path.join(out_dir, f"{base}_grid_read.png")
    try:
        draw_grid_answers(grid_path, read_path, out, g, cal)
    except Exception as e:                                  # an overlay must never break the run
        print(f"  [grid_extract] overlay not saved: {type(e).__name__}: {e}")
    return {"grid": out, "pixel": pixel, "agreement": compare_series(out, pixel, cal["y_map"]),
            "grid_image": grid_path, "grid_read_image": read_path, "faded_grid_images": faded_paths,
            "clean_image": src if src != image_path else None, "annotations": ann}


_PALETTE = [(220, 0, 0), (0, 110, 230), (0, 160, 0), (200, 0, 200), (230, 120, 0), (0, 170, 170)]


def draw_grid_answers(grid_path: str, out_path: str, answers: Dict[str, List], grid: Dict[str, Any],
                      cal: Dict[str, Any]) -> str:
    """Draw what the VLM READ on the grid image (config D's own overlay).
    original pixel -> grid-image pixel:  p * grid['scale'] + grid['offset']"""
    im = Image.open(grid_path).convert("RGB")
    d = ImageDraw.Draw(im)
    k = float(grid.get("scale", 1.0))
    ox, oy = grid.get("offset", (0, 0))
    x_map, y_map = cal.get("x_map"), cal["y_map"]
    cats = {str(l).strip().lower(): p for p, l in cal.get("x_cat", [])}
    font = _font(max(12, int(im.height / 60)))
    cp = sorted(cats.values())
    step = float(np.median(np.diff(cp))) if len(cp) > 1 else 0.0
    n = max(1, len(answers))
    for i, (name, pts) in enumerate(answers.items()):
        dx = (i - (n - 1) / 2) * (0.8 * step / n)          # grouped bars: one slot per series
        col = _PALETTE[i % len(_PALETTE)]
        d.text((8, 8 + 16 * i), f"o {name}", fill=col, font=font)
        for x, y in pts:
            if not isinstance(y, (int, float)):
                continue
            if isinstance(x, (int, float)) and x_map is not None:
                px = x_map.pixel(float(x))
            else:
                px = cats.get(str(x).strip().lower())
                px = None if px is None else px + dx
            if px is None:
                continue
            try:
                py = y_map.pixel(float(y))
            except Exception:
                continue
            gx, gy = px * k + ox, py * k + oy
            d.ellipse([gx - 6, gy - 6, gx + 6, gy + 6], outline=col, width=3)
            d.text((gx + 8, gy - 18), f"{y:g}", fill=col, font=font)
    im.save(out_path)
    return out_path


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