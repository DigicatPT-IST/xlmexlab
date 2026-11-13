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
graph_extraction.py - everything needed for configs C_ocr_color and D_grid_vlm, in ONE file.

Merged from (no code changes, only the cross-file imports removed):
    1. grounding_utils.py        prompts, JSON/number parsing, coordinates, axis calibration
    2. ocr_ticks.py              tick labels from OCR (tesseract or rapidocr)
    3. graph_color_digitizer.py  data points read from the pixels by colour
    4. graph_grounder.py         GraphGrounder (calibration + config C) + normalize_stage1
    5. grid_overlay.py           value grid + VLM reading (config D: grid_extract)

Use in extractor_nanoparticles.py:
    from xlmexlab.graph_extraction import GraphGrounder, grid_extract

Requirements: numpy, pillow, scipy, and one OCR backend:
    pip install rapidocr_onnxruntime            (or: tesseract program + pip install pytesseract)
"""


import copy
import difflib
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy import ndimage

from xlmexlab.prompt import PromptFormatter

try:
    import pytesseract
except ImportError:                                       # pragma: no cover
    pytesseract = None


# 1. GROUNDING UTILS - prompts, parsing, coordinates, axis calibration

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


# parsing
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


#  axis calibration
class NumericAxis:
    def __init__(self, pix: List[float], vals: List[float], log_hint: Optional[bool] = None):
        """log_hint=True: the OCR / stage 1 says the axis is logarithmic (needed when only
        2 ticks are readable: 2 points always fit a straight line perfectly)."""
        lin = self._fit(pix, vals, log=False)
        best, self.log = lin, False
        if min(vals) > 0:
            lg = self._fit(pix, vals, log=True)
            if log_hint and max(vals) / min(vals) >= 10:
                best, self.log = lg, True
            elif len(vals) >= 3 and lin["err"] > 0.01 and lg["err"] < 0.5 * lin["err"]:
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


# refinement 
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



# 2. OCR TICKS - tick labels with exact text boxes (tesseract or rapidocr)
NUM_WHITELIST = "0123456789.,-%eE^x×"
_RAPID = None


def _rapid():
    global _RAPID
    if _RAPID is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError:
            from rapidocr import RapidOCR                  # newer package name
        _RAPID = RapidOCR()
    return _RAPID


def _tesseract_ok() -> bool:
    if pytesseract is None:
        return False
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def resolve_backend(backend: str = "auto") -> str:
    if backend != "auto":
        return backend
    if _tesseract_ok():
        return "tesseract"
    try:
        _rapid()
        return "rapidocr"
    except Exception as e:
        raise ImportError("No OCR backend: install the tesseract program + pytesseract, "
                          "or `pip install rapidocr_onnxruntime`") from e


def _rapid_run(arr: np.ndarray) -> List[Tuple[List, str, float]]:
    """RapidOCR on an RGB array -> [(polygon, text, score)], for old and new package versions."""
    out = _rapid()(arr)
    if isinstance(out, tuple):                           # rapidocr_onnxruntime: (result, elapse)
        res = out[0] or []
        return [(r[0], str(r[1]), float(r[2])) for r in res]
    boxes = getattr(out, "boxes", None)                  # rapidocr >= 2: RapidOCROutput
    if boxes is None:
        return []
    return [(b, str(t), float(sc)) for b, t, sc in zip(boxes, out.txts, out.scores)]


def _engine_words(ocr_img: Image.Image, backend: str) -> List[Tuple[str, float, float, float, float, float]]:
    """(text, x1, y1, x2, y2, conf 0-100) in ocr_img pixels."""
    words = []
    if backend == "tesseract":
        d = pytesseract.image_to_data(ocr_img, config="--psm 11", output_type=pytesseract.Output.DICT)
        for i, txt in enumerate(d["text"]):
            if txt.strip():
                x, y, w, h = d["left"][i], d["top"][i], d["width"][i], d["height"][i]
                words.append((txt.strip(), x, y, x + w, y + h, float(d["conf"][i])))
        return words
    arr = np.stack([np.array(ocr_img.convert("L"))] * 3, axis=-1)
    for poly, txt, sc in _rapid_run(arr):
        xs, ys = [p[0] for p in poly], [p[1] for p in poly]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        parts = txt.split()
        # RapidOCR returns whole lines: split "10 20 30" into separate numbers
        if len(parts) > 1 and all(parse_number(p) is not None for p in parts) and (x2 - x1) > (y2 - y1):
            n_chars = len(txt)
            pos = 0
            for ptxt in parts:
                start = txt.index(ptxt, pos)
                pos = start + len(ptxt)
                words.append((ptxt, x1 + (x2 - x1) * start / n_chars, y1,
                              x1 + (x2 - x1) * pos / n_chars, y2, sc * 100))
        else:
            words.append((txt.strip(), x1, y1, x2, y2, sc * 100))
    return words


def _read_crop_texts(crop: Image.Image, backend: str) -> List[Tuple[str, float]]:
    """Read a small crop as a number -> [(text, conf)]."""
    crop = ImageOps.expand(crop, border=16, fill=255)
    if backend == "tesseract":
        out = []
        for psm in (7, 8, 10):
            d = pytesseract.image_to_data(crop, config=f"--psm {psm} -c tessedit_char_whitelist={NUM_WHITELIST}",
                                          output_type=pytesseract.Output.DICT)
            ws = [(d["text"][i], float(d["conf"][i])) for i in range(len(d["text"])) if d["text"][i].strip()]
            if ws:
                out.append(("".join(w for w, _ in ws), float(np.mean([c for _, c in ws]))))
        return out
    out = []
    rec = getattr(_rapid(), "text_recognizer", None)     # recogniser only: no detector, so single
    if rec is not None:                                  # digits ('0', '4') are read too
        try:
            tight = ImageOps.expand(crop.crop((8, 12, crop.width - 8, crop.height - 12)), border=(8, 4), fill=255)
            r = rec([np.stack([np.array(tight.convert("L"))] * 3, axis=-1)])
            r = r[0] if isinstance(r, tuple) else r
            out += [(str(t), 100 * float(sc)) for t, sc in r if str(t).strip()]
        except Exception:
            pass
    res = _rapid_run(np.stack([np.array(crop.convert("L"))] * 3, axis=-1))
    if res:
        out.append(("".join(t for _, t, _ in res), 100 * float(np.mean([sc for _, _, sc in res]))))
    return out


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


#  number parsing 
_SUP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")
_FIX = str.maketrans({"O": "0", "o": "0", "l": "1", "I": "1", "|": "1", "S": "5", "−": "-", "–": "-", "—": "-"})


def parse_number(text: str) -> Optional[float]:
    """'1,000' '0.5' '0,5' '-20' '−20' '75%' '10³' '10^3' '1e3' '1×10^3' -> float."""
    t = text.strip().replace(" ", "")
    if not t:
        return None
    t = t.translate(_FIX).replace("%", "")
    # powers of ten need an explicit marker: 10^3, 10^{3}, 10³, 10⁻², 2x10^3, or '10-2' (OCR of 10⁻²).
    # A plain '100' / '1000' / '105' is an ordinary number (the old pattern read '100' as 10^0 = 1).
    m = (re.fullmatch(r"(?:(-?\d+(?:\.\d+)?)[x×*])?10(?:\^\{?|\{)(-?\d{1,2})\}?", t)
         or re.fullmatch(r"(?:(-?\d+(?:\.\d+)?)[x×*])?10([⁰¹²³⁴⁵⁶⁷⁸⁹⁻]{1,3})", t)
         or re.fullmatch(r"()10(-\d{1,2})", t))
    if m:
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


def _read_number_crop(ocr_img: Image.Image, t: Token, scale: int, backend: str) -> Tuple[Optional[str], float]:
    """Re-read one token as a single number."""
    pad = max(4, int(0.5 * t.h * scale))
    box = (max(0, int(t.x1 * scale) - pad), max(0, int(t.y1 * scale) - pad),
           min(ocr_img.width, int(t.x2 * scale) + pad), min(ocr_img.height, int(t.y2 * scale) + pad))
    reads = [r for r in _read_crop_texts(ocr_img.crop(box), backend) if parse_number(r[0]) is not None]
    if not reads:
        return None, -1
    return max(reads, key=lambda r: r[1])


def ocr_tokens(image_path: str, scale: int = 3, min_conf: float = 0, backend: str = "auto") -> List[Token]:
    """All text tokens with exact boxes in ORIGINAL image pixels.
    Adjacent words on one line are merged ('Free DOX', '2,000'); number-like tokens
    are re-read on their own, which fixes '10¢'->'100', 'O'->'0', etc."""
    backend = resolve_backend(backend)
    img = Image.open(image_path).convert("RGB")
    ocr_img = _to_ocr_image(img, scale)

    words: List[Token] = []
    for txt, x1, y1, x2, y2, conf in _engine_words(ocr_img, backend):
        if conf < min_conf:
            continue
        x1, y1, x2, y2 = x1 / scale, y1 / scale, x2 / scale, y2 / scale
        if (y2 - y1) < 4 or (x2 - x1) < 2:
            continue
        words.append(Token(txt, x1, y1, x2, y2, conf))

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
            txt2, conf2 = _read_number_crop(ocr_img, t, scale, backend)
            n2 = parse_number(txt2) if txt2 else None
            if n2 is not None and (t.num is None or conf2 > t.conf):
                t.text, t.num, t.conf = txt2, n2, conf2
    return merged


# tick selection
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
        picked = [(t.cx, t.cy, _canonical_text(t, expected)) for t in toks]
        if len(picked) < min(3, len(expected)):
            by_order = _assign_by_order(tokens, axis, expected, size)
            if len(by_order) > len(picked):
                picked, info = by_order, {"type": "by_order", "n": len(by_order)}
        out[axis] = picked
        out["info"][axis] = info
    return out


def _assign_by_order(tokens: List[Token], axis: str, expected: Sequence[str],
                     size: Tuple[int, int]) -> List[Tuple[float, float, str]]:
    """Fallback for unreadable but well-placed labels (log axes: '10⁻²' read as '1077').
    Takes the row (x) / column (y) of number-like tokens with exactly as many labels as
    stage 1 listed, evenly spaced, and assigns the stage-1 values in order."""
    vals = [(parse_number(e), e) for e in expected or []]
    vals = [v for v in vals if v[0] is not None]
    if len(vals) < 3:
        return []
    W, H = size
    numeric = [t for t in tokens if re.match(r"^[-−–]?\d", t.text.strip())]
    if len(numeric) < len(vals):
        return []
    hmed = float(np.median([t.h for t in numeric]))
    groups = _cluster([t.cy for t in numeric], 0.5 * hmed) if axis == "x" else \
        _cluster([t.x2 for t in numeric], max(4.0, 0.7 * hmed))
    best, best_score = [], None
    for g in groups:
        toks = sorted((numeric[i] for i in g), key=lambda t: t.cx if axis == "x" else t.cy)
        if len(toks) != len(vals):
            continue
        pos = np.array([t.cx if axis == "x" else t.cy for t in toks])
        d = np.diff(pos)
        if d.mean() <= 0 or d.std() / d.mean() > 0.08:        # labels must be evenly spaced
            continue
        where = np.mean([t.cy for t in toks]) / H if axis == "x" else 1 - np.mean([t.cx for t in toks]) / W
        if best_score is None or where > best_score:
            best, best_score = toks, where
    if not best:
        return []
    ordered = sorted(vals, key=lambda v: v[0])                # ascending values
    if axis == "y":
        ordered = ordered[::-1]                               # top of the image = largest value
    return [(t.cx, t.cy, e) for t, (_, e) in zip(best, ordered)]


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
    marked = draw_token_marks(image_path, tokens, work_path)
    fm = PromptFormatter(**build_token_classification_prompt(tokens))
    fm.model_post_init(prompt_template_path)
    raw = vlm.run_image_single_prompt_rescale(fm.format_prompt("<image>"), marked, scale=1.0)
    print(f"\n  [ocr_ticks] token labels RAW:\n{raw}")
    return parse_token_labels(raw, len(tokens))


# gap filling
def _read_crop_variants(images: List[Image.Image], box_px: Tuple[float, float, float, float],
                        scale: int, backend: str = "auto") -> List[str]:
    """Read one small region as a number, on the cleaned image and on the raw one
    (line removal can cut a '0' at the origin)."""
    backend = resolve_backend(backend)
    out = []
    x1, y1, x2, y2 = [int(round(v * scale)) for v in box_px]
    for im in images:
        crop = im.crop((max(0, x1), max(0, y1), min(im.width, x2), min(im.height, y2)))
        if crop.width < 4 or crop.height < 4:
            continue
        out += [t for t, _ in _read_crop_texts(crop, backend)]
    return out


def fill_missing_ticks(image_path: str, picked: List[Tuple[float, float, str]], axis: str,
                       expected: Sequence[str], scale: int = 3, max_extend: int = 3,
                       backend: str = "auto") -> List[Tuple[float, float, str]]:
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
        for txt in _read_crop_variants([clean, raw], box, scale, backend):
            n = parse_number(txt)
            if n is not None and abs(n - val) <= 1e-9 * max(1.0, abs(val)):
                label = next((e for e in expected or [] if parse_number(e) == val), txt)
                added.append((p, cy, label) if axis == "x" else (cx, p, label))
                break
    key = 0 if axis == "x" else 1
    return sorted(added, key=lambda q: q[key])


def find_ticks(image_path: str, stage1: Dict[str, Any], scale: int = 3, fill_gaps: bool = True,
               vlm_labels: Optional[Dict[int, str]] = None, tokens: Optional[List[Token]] = None,
               backend: str = "auto") -> Dict[str, Any]:
    """OCR + selection + gap filling. Returns {"x": [(cx, cy, text)], "y": [...], "info": {...}}."""
    backend = resolve_backend(backend)
    tokens = tokens if tokens is not None else ocr_tokens(image_path, scale=scale, backend=backend)
    size = Image.open(image_path).size
    res = locate_ticks(tokens, stage1, size, vlm_labels)
    if fill_gaps:
        for axis in ("x", "y"):
            if res["info"].get(axis, {}).get("type") in ("linear", "log"):
                expected = [str(t) for t in (stage1.get(f"{axis}_ticks") or []) if t not in (None, "", "UNKNOWN")]
                before = len(res[axis])
                res[axis] = fill_missing_ticks(image_path, res[axis], axis, expected, scale, backend=backend)
                res["info"][axis]["filled"] = len(res[axis]) - before
    res["info"]["backend"] = backend
    return res



# 3. COLOUR DIGITIZER - data points read from the pixels by colour
Box = Tuple[float, float, float, float]          # x1, y1, x2, y2 (pixels)

NAMED_RGB = {
    "red": (220, 40, 40), "dark red": (150, 30, 30), "maroon": (128, 0, 0),
    "brown": (150, 80, 50), "orange": (245, 140, 20), "yellow": (240, 200, 0),
    "gold": (230, 180, 20), "olive": (130, 140, 40), "green": (40, 160, 60),
    "light green": (150, 200, 90), "dark green": (20, 100, 40), "lime": (120, 220, 40),
    "teal": (30, 100, 110), "dark teal": (30, 90, 100), "cyan": (0, 190, 220),
    "turquoise": (60, 200, 200), "light blue": (100, 170, 230), "sky blue": (110, 180, 235),
    "blue": (30, 70, 220), "dark blue": (20, 40, 140), "navy": (20, 30, 110),
    "purple": (130, 50, 160), "violet": (150, 80, 200), "magenta": (230, 0, 230),
    "pink": (240, 120, 170), "salmon": (240, 130, 110), "gray": (128, 128, 128),
    "grey": (128, 128, 128), "black": (0, 0, 0),
}
ACHROMATIC = {"black", "gray", "grey", "white", "dark gray", "dark grey", "light gray", "light grey"}


#colour helpers
def _css_colors() -> Dict[str, str]:
    try:
        from matplotlib.colors import CSS4_COLORS
        return CSS4_COLORS
    except Exception:
        return {}


_CSS = _css_colors()


def name_to_rgb(name: str) -> Optional[Tuple[int, int, int]]:
    """Colour as stage 1 wrote it -> RGB. Accepts our names ('dark teal'), hex codes
    ('#B22222', '#fa0'), rgb(178,34,34) and CSS names ('firebrick', 'dark slate gray')."""
    n = (name or "").strip().lower()
    if not n or n in ("unknown", "none", "-"):
        return None
    m = re.search(r"#([0-9a-f]{6}|[0-9a-f]{3})\b", n)        # '#1F77B4 (blue)', 'blue (#1f77b4)'
    if not m:
        m = re.fullmatch(r"#?([0-9a-f]{6}|[0-9a-f]{3})", n)
        m = m if m and re.search(r"\d", n) else None
    if m:
        h = m.group(1)
        if len(h) == 3:
            h = "".join(c * 2 for c in h)
        return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    m = re.fullmatch(r"rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+).*\)", n)
    if m:
        return tuple(min(255, int(v)) for v in m.groups())
    if n in NAMED_RGB:
        return NAMED_RGB[n]
    css = _CSS.get(n.replace(" ", "").replace("-", "").replace("grey", "gray"))
    if css:
        return name_to_rgb(css)
    hits = [k for k in sorted(NAMED_RGB, key=len, reverse=True) if k in n]
    return NAMED_RGB[hits[0]] if hits else None


def is_achromatic(name: str) -> bool:
    """Black / gray / white, whether written as a name or as a code like '#333333'."""
    n = (name or "").strip().lower()
    if n in ACHROMATIC or any(a == n for a in ACHROMATIC):
        return True
    rgb = name_to_rgb(n)
    if rgb is None:
        return False
    import colorsys
    _, sat, val = colorsys.rgb_to_hsv(*[c / 255 for c in rgb])
    return sat < 0.15 or val < 0.15


def _region_mask(shape, box: Box, exclude: Sequence[Box]) -> np.ndarray:
    h, w = shape[:2]
    m = np.zeros((h, w), bool)
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    m[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = True
    for ex in exclude or []:
        a, b, c, d = [int(round(v)) for v in ex]
        m[max(0, b):min(h, d), max(0, a):min(w, c)] = False
    return m


def rgb_to_lab(rgb) -> np.ndarray:
    """sRGB (0-255, shape (..., 3)) -> CIELAB. Distances in Lab follow what the eye sees:
    pastel yellow vs pastel orange are far apart, a colour vs its anti-aliased edge are close."""
    c = np.asarray(rgb, float) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = c @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def delta_e(a, b) -> float:
    """Perceptual distance between two RGB colours (CIE76 Delta E: ~2 just visible, >25 clearly different)."""
    return float(np.linalg.norm(rgb_to_lab(a) - rgb_to_lab(b)))


def dominant_colors(arr: np.ndarray, box: Box, exclude: Sequence[Box] = (),
                    min_px: int = 40, merge_de: float = 12.0, min_chroma: int = 20) -> List[Tuple[np.ndarray, int]]:
    """Chromatic colours present in the plot area, most frequent first.
    Bins are merged when they are perceptually close (Lab Delta E < merge_de): the old RGB
    distance merged pastel yellow into pastel orange (Fig 56: Conj-HSLN lost its colour)
    and the chroma cut-off of 50 dropped lavender (HSLN) entirely."""
    region = _region_mask(arr.shape, box, exclude)
    px = arr[region].astype(int)
    px = px[(px.max(1) - px.min(1)) >= min_chroma]   # skip white / gray / black, keep pastel fills
    if len(px) == 0:
        return []
    q = (px // 8) * 8 + 4
    keys, counts = np.unique(q, axis=0, return_counts=True)
    labs = rgb_to_lab(keys)
    clusters: List[List] = []                                   # [seed rgb, count, seed lab]
    for i in np.argsort(-counts):
        if counts[i] < 2:
            break
        c = keys[i].astype(float)
        for cl in clusters:
            if np.linalg.norm(cl[2] - labs[i]) < merge_de:
                cl[1] += int(counts[i])
                break
        else:
            clusters.append([c, int(counts[i]), labs[i]])  # seed = most frequent bin
    clusters = [[c[0], c[1]] for c in sorted(clusters, key=lambda c: -c[1])]
    # anti-aliased edges are blends of a series colour with the white background:
    # fold every cluster that lies on the segment strong-colour -> white into it
    merged: List[List] = []
    for c, n in clusters:
        for m in merged:
            s_ = m[0]
            t = float(np.dot(c - s_, 255 - s_) / max(1e-6, np.dot(255 - s_, 255 - s_)))
            if 0 <= t <= 0.85 and np.linalg.norm(s_ + t * (255 - s_) - c) < 28:
                m[1] += n
                break
        else:
            merged.append([c, n])
    merged = [m for m in merged if m[1] >= min_px]
    return [(m[0], m[1]) for m in sorted(merged, key=lambda c: -c[1])]


def _hsv(rgb) -> Tuple[float, float, float]:
    import colorsys
    r, g, b = [float(v) / 255 for v in rgb]
    return colorsys.rgb_to_hsv(r, g, b)


def _color_dist(a, b) -> float:
    """Hue-first distance between two colours (robust to light/dark variants)."""
    ha, sa, va = _hsv(a)
    hb, sb, vb = _hsv(b)
    dh = min(abs(ha - hb), 1 - abs(ha - hb)) * 360
    return dh + 40 * abs(sa - sb) + 30 * abs(va - vb)


def colour_match_score(ref, plot_rgb, n_px: int, max_px: int, max_hue: float = 50.0,
                       max_de: float = 90.0) -> Optional[float]:
    """How well a plot colour fits the colour stage 1 gave (lower = better, None = no match).
    Mostly HUE: a VLM says 'orange' / 'blue' or a memorised hex for what is really a pastel or a
    darker fill, so lightness is unreliable but the hue is right. Lab Delta E breaks ties between
    colours of the same hue, and small clusters (anti-aliased bar edges, borders) are penalised so
    the real fill wins over its own dark edge."""
    hr, sr, _ = _hsv(ref)
    hp, sp, _ = _hsv(plot_rgb)
    dh = min(abs(hr - hp), 1 - abs(hr - hp)) * 360
    de = delta_e(ref, plot_rgb)
    if dh > max_hue or de > max_de:
        return None
    small = 15.0 if n_px < 0.15 * max_px else 0.0
    return dh + 0.25 * de + small


def assign_colors(series_names_colors: List[Tuple[str, str]],
                  palette: List[Tuple[np.ndarray, int]]) -> Dict[str, Optional[np.ndarray]]:
    """One-to-one assignment of the stage-1 colours (VLM hex / name, or the legend swatch) to the
    colours REALLY drawn in the plot: the best (series, plot colour) pair is fixed first
    (colour_match_score), then the next best among what is left, so when two plot colours are
    similar each series gets the one closest to its own colour. No acceptable plot colour -> None
    (no pixels, no fading). Black/gray series get None (read by the VLM)."""
    out: Dict[str, Optional[np.ndarray]] = {}
    cands = []
    big = max((n for _, n in palette), default=1)
    for name, cname in series_names_colors:
        rgb = name_to_rgb(cname)
        if rgb is None or is_achromatic(cname):
            out[name] = None
            continue
        for j, (c, n) in enumerate(palette):
            sc = colour_match_score(rgb, c, n, big)
            if sc is not None:
                cands.append((sc, name, j))
    used = set()
    for d, name, j in sorted(cands, key=lambda t: t[0]):
        if name in out or j in used:
            continue
        out[name] = palette[j][0]
        used.add(j)
    for name, _ in series_names_colors:
        out.setdefault(name, None)
    return out


def choose_series_colors(series: List[Dict[str, Any]], palette: List[Tuple[np.ndarray, int]],
                         legend_max_de: float = 15.0) -> None:
    """Which colour describes each series (in place)? The VLM's colour (hex or name) is the default.
    The legend swatch read from the pixels replaces it ONLY when that swatch colour is really drawn
    in the plot (Delta E <= legend_max_de to a plot colour): a swatch read next to the wrong text
    (an x label, a border) is then ignored. Sets s["color"], keeps s["color_vlm"], s["color_source"]."""
    pal = [c for c, _ in palette]
    for s in series:
        s.setdefault("color_vlm", s.get("color"))
        leg = s.get("color_legend")
        rgb = name_to_rgb(leg) if leg else None
        if rgb is not None and not is_achromatic(leg) and pal and min(delta_e(rgb, c) for c in pal) <= legend_max_de:
            s["color"], s["color_source"] = leg, "legend"
        else:
            s["color"], s["color_source"] = s.get("color_vlm"), "vlm"


def color_mask(arr: np.ndarray, rgb: np.ndarray, region: np.ndarray, tol: float = 70.0,
               palette: Optional[List[np.ndarray]] = None) -> np.ndarray:
    """Pixels of this series' colour. With `palette` (all detected series colours) a
    pixel is kept only if this colour is its nearest one, which separates
    neighbouring hues (red vs orange) without a tight tolerance."""
    a = arr.astype(float)
    d = np.linalg.norm(a - np.asarray(rgb, float), axis=2)
    m = (d < tol) & region & ((a.max(2) - a.min(2)) >= 20)
    if palette:
        others = [np.linalg.norm(a - np.asarray(c, float), axis=2)
                  for c in palette if np.linalg.norm(np.asarray(c, float) - rgb) > 1]
        if others:
            m &= d <= np.min(others, axis=0)
    lab, n = ndimage.label(m)                                   # drop specks, keep thin lines
    if n:
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        m &= np.isin(lab, np.flatnonzero(sizes >= 4) + 1)
    return m


#extractionv
def _runs(flags: np.ndarray, min_len: int = 1, max_gap: int = 0) -> List[Tuple[int, int]]:
    idx = np.flatnonzero(flags)
    if len(idx) == 0:
        return []
    runs, s, p = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - p > max_gap + 1:
            if p - s + 1 >= min_len:
                runs.append((s, p))
            s = i
        p = i
    if p - s + 1 >= min_len:
        runs.append((s, p))
    return runs


def extract_bars(mask: np.ndarray, zero_px: Optional[float]) -> List[Tuple[float, float]]:
    cols = mask.sum(0)
    if cols.max() == 0:
        return []
    filled = cols >= 3
    pts = []
    for a, b in _runs(filled, min_len=4, max_gap=1):
        core = slice(a + (b - a) // 4, b - (b - a) // 4 + 1)      # ignore anti-aliased edges
        rows = [np.flatnonzero(mask[:, c]) for c in range(core.start, core.stop)]
        rows = [r for r in rows if len(r)]
        if not rows:
            continue
        top = float(np.median([r.min() for r in rows]))
        bot = float(np.median([r.max() for r in rows]))
        y = bot if (zero_px is not None and abs(top - zero_px) < abs(bot - zero_px)) else top
        pts.append(((a + b) / 2.0, y))
    return pts


def extract_markers(mask: np.ndarray, x_fallback: Sequence[float] = ()) -> Tuple[List[Tuple[float, float]], str]:
    """Return marker centres; if the series has no visible markers, sample the
    curve at x_fallback positions."""
    cols = mask.sum(0).astype(float)
    present = cols > 0
    if not present.any():
        return [], "empty"
    line_w = float(np.median(cols[present]))
    # expected count of a plain line in each column grows with its slope
    ys = np.arange(mask.shape[0])[:, None]
    centre = np.where(present, (mask * ys).sum(0) / np.maximum(cols, 1), np.nan)
    xi = np.arange(len(cols))
    centre = np.interp(xi, xi[present], centre[present])
    slope = np.abs(np.gradient(ndimage.uniform_filter1d(centre, 9)))
    expected = line_w * np.sqrt(1 + slope ** 2)
    thick = present & (ndimage.uniform_filter1d(cols, 3) >= np.maximum(expected * 1.5, expected + 1.5))
    pts = []
    for a, b in _runs(thick, min_len=3, max_gap=1):
        if b - a > 25:                        # a long thick stretch is a filled area, not a marker
            continue
        sub = mask[:, a:b + 1]
        lab, n = ndimage.label(sub)
        if n == 0:
            continue
        sizes = ndimage.sum(sub, lab, range(1, n + 1))
        k = int(np.argmax(sizes)) + 1
        ys, xs = np.nonzero(lab == k)
        pts.append((a + xs.mean(), ys.mean()))
    if len(pts) >= 2:
        return pts, "markers"
    # plain line: read the curve centre at the requested x positions
    out = []
    for x in x_fallback:
        xi = int(round(x))
        win = mask[:, max(0, xi - 1):xi + 2]
        ys = np.nonzero(win)[0]
        if len(ys):
            out.append((float(xi), float(np.median(ys))))
    return out, "curve"


def plot_box_from_ticks(x_px: Sequence[float], y_px: Sequence[float], shape,
                        x_label_y: Optional[float] = None, y_label_x: Optional[float] = None,
                        gray: Optional[np.ndarray] = None) -> Box:
    """Plot area from the tick-label positions. The outermost labels are often missing
    (e.g. '0' at the origin, '100' at the top), so the box is bounded by the axis lines
    / the label rows when known, and otherwise extended by about one tick step."""
    h, w = shape[:2]
    xs, ys = sorted(x_px), sorted(y_px)
    dx = (xs[-1] - xs[0]) / max(1, len(xs) - 1) if len(xs) > 1 else 20
    dy = (ys[-1] - ys[0]) / max(1, len(ys) - 1) if len(ys) > 1 else 20
    left, right = xs[0] - 1.0 * dx, xs[-1] + 1.0 * dx
    top, bottom = ys[0] - 1.0 * dy, ys[-1] + 1.0 * dy
    if x_label_y is not None:                     # plot ends above the x tick labels
        bottom = min(bottom, x_label_y - 7)
    if y_label_x is not None:                     # and right of the y tick labels
        left = max(left, y_label_x + 12)
    if gray is not None:
        y_ax, x_ax = find_axis_lines(gray)
        if y_ax is not None and ys[-1] - 2 <= y_ax <= bottom + 2:
            bottom = y_ax - 1
        if x_ax is not None and left - 2 <= x_ax <= xs[0] + 2:
            left = x_ax + 1
    left, right = sorted((left, right))
    top, bottom = sorted((top, bottom))
    return (max(0, left), max(0, top), min(w, right), min(h, bottom))


def find_axis_lines(gray: np.ndarray, dark: int = 120) -> Tuple[Optional[int], Optional[int]]:
    """(row of the x-axis line, column of the y-axis line), or None when absent."""
    h, w = gray.shape
    d = gray < dark
    def longest(v):
        r = _runs(v)
        return max((b - a + 1 for a, b in r), default=0)
    rows = [longest(d[y, :]) for y in range(h // 2, h)]
    cols = [longest(d[:, x]) for x in range(0, w // 2)]
    y_ax = (h // 2 + int(np.argmax(rows))) if max(rows, default=0) > 0.4 * w else None
    x_ax = int(np.argmax(cols)) if max(cols, default=0) > 0.4 * h else None
    return y_ax, x_ax


def _text_blobs(gray: np.ndarray, dark: int) -> List[Tuple[float, float, float, float]]:
    """Bounding boxes (x1, y1, x2, y2) of dark connected components, characters merged per word."""
    d = gray < dark
    lines = ndimage.binary_opening(d, structure=np.ones((1, 35))) | \
            ndimage.binary_opening(d, structure=np.ones((35, 1)))
    d &= ~ndimage.binary_dilation(lines, iterations=1)          # remove axis lines / frame / grid
    d = ndimage.binary_opening(d, structure=np.ones((2, 1))) | ndimage.binary_opening(d, structure=np.ones((1, 2)))
    lab, n = ndimage.label(ndimage.binary_dilation(d, structure=np.ones((3, 5))))
    out = []
    for sl in ndimage.find_objects(lab):
        if sl is None:
            continue
        y1, y2, x1, x2 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
        if 5 <= (y2 - y1) <= 30 and (x2 - x1) <= 80:              # drop lines, frames, long titles
            out.append((x1, y1, x2, y2))
    return out


def snap_ticks_to_text(gray: np.ndarray, centers: List[Tuple[float, float]], axis: str,
                       dark: int = 120) -> List[Tuple[float, float]]:
    """Correct VLM tick-label centres using the real text blobs.
    x labels = blobs just below the x-axis line and right of the y-axis line;
    y labels = blobs whose right edge touches the y-axis line.
    If the number of blobs equals the number of ticks they are assigned in order,
    which removes any systematic VLM offset; otherwise the VLM centres are kept."""
    if len(centers) < 2:
        return centers
    y_ax, x_ax = find_axis_lines(gray, dark)
    blobs_all = _text_blobs(gray, dark)
    if axis == "x":
        ref = y_ax if y_ax is not None else float(np.median([c[1] for c in centers])) - 8
        left = (x_ax - 4) if x_ax is not None else -1
        cand = [b for b in blobs_all if ref + 3 <= b[1] <= ref + 24 and b[2] >= left]
        if cand:                                               # keep the first text row only
            top = min(b[1] for b in cand)
            cand = [b for b in cand if b[1] <= top + 6]
        blobs = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in cand]
        key = 0
    else:
        if x_ax is not None:
            cand = [b for b in blobs_all if x_ax - 14 <= b[2] <= x_ax + 4 and b[0] < x_ax - 4]
        else:
            xc = float(np.median([c[0] for c in centers]))
            cand = [b for b in blobs_all if b[0] - 10 <= xc <= b[2] + 10]
        blobs = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in cand]
        key = 1
    if len(blobs) > len(centers):          # keep the run of blobs that best matches the guess
        span = np.array(sorted(c[key] for c in centers))
        bl = sorted(blobs, key=lambda b: b[key])
        blobs = min((bl[s:s + len(centers)] for s in range(len(bl) - len(centers) + 1)),
                    key=lambda cand: np.var(np.array([c[key] for c in cand]) - span)
                    + 0.01 * np.mean(np.abs(np.array([c[key] for c in cand]) - span)))
    if len(blobs) != len(centers):
        return centers
    order = np.argsort([c[key] for c in centers])
    blobs = sorted(blobs, key=lambda b: b[key])
    out = list(centers)
    for i, j in enumerate(order):
        out[j] = blobs[i]
    return out


def sample_curve(mask: np.ndarray, xs: Sequence[float], win: int = 2) -> List[Tuple[float, float]]:
    """Centre of the series' pixels at each requested x (skipped where the series is absent)."""
    out = []
    for x in xs:
        xi = int(round(x))
        sub = mask[:, max(0, xi - win):xi + win + 1]
        ys = np.nonzero(sub)[0]
        if len(ys):
            out.append((float(x), float(np.median(ys))))
    return out


def shared_x_positions(marker_sets: Sequence[Sequence[Tuple[float, float]]], tol: float = 4.0) -> List[float]:
    """Scientific plots usually measure every series at the same x (doses, time points).
    Merge the marker x positions found in all series into one sorted list."""
    xs = sorted(x for pts in marker_sets for x, _ in pts)
    groups: List[List[float]] = []
    for x in xs:
        if groups and x - groups[-1][-1] <= tol:
            groups[-1].append(x)
        else:
            groups.append([x])
    return [float(np.mean(g)) for g in groups]


def extract_line_series(masks: Dict[str, np.ndarray], x_fallback: Sequence[float]) -> Dict[str, Tuple[List[Tuple[float, float]], str]]:
    """Two passes over all line series of one chart:
    1) markers per series; 2) x positions shared across series are used to read
    the series whose markers were too small/occluded to detect."""
    first = {n: extract_markers(m, ()) for n, m in masks.items()}
    marker_sets = [pts for pts, kind in first.values() if kind == "markers"]
    shared = shared_x_positions(marker_sets) if marker_sets else list(x_fallback)
    out = {}
    for n, m in masks.items():
        pts, kind = first[n]
        if kind == "markers" and len(pts) >= len(shared):
            out[n] = (pts, "markers")
        else:
            sampled = sample_curve(m, shared)
            out[n] = (sampled, "shared_x" if marker_sets else "ticks") if len(sampled) > len(pts) else (pts, kind)
    return out


def match_categories(pts: List[List[Any]], cats: Sequence[str], extra: Sequence[str] = ()) -> List[List[Any]]:
    """VLM category names -> the axis category names.
    1) exact (case-insensitive) matches first - against the calibrated axis categories AND the
       stage-1 x ticks (OCR can miss a label: 'HSLN' was then fuzzily turned into 'Conj-HSLN');
    2) then a close spelling match (ratio >= 0.8) to a category that no other answer uses;
    3) otherwise the VLM's label is KEPT as written - never moved onto a different category."""
    allc, seen = [], set()
    for c in list(cats) + list(extra):
        k = str(c).strip().lower()
        if k and k not in seen:
            seen.add(k)
            allc.append(str(c))
    if not allc:
        return pts
    low = [c.strip().lower() for c in allc]
    keys = [str(x).strip().lower() for x, _ in pts]
    used = {k for k in keys if k in low}
    out = []
    for (x, y), k in zip(pts, keys):
        if k in low:
            out.append([allc[low.index(k)], y])
            continue
        free = [c for c in low if c not in used]
        m = difflib.get_close_matches(k, free, n=1, cutoff=0.8)
        if m:
            used.add(m[0])
            out.append([allc[low.index(m[0])], y])
        else:
            out.append([x, y])
    return out


def complete_categories(pairs: List[Tuple[float, float, str]], ticks: Sequence[Any],
                        max_resid: float = 0.2) -> List[Tuple[float, float, str]]:
    """OCR found only some category labels: place the missing stage-1 categories by their index.
    Category centres are evenly spaced, so position = a * index + b is fitted on the labels that
    were found (exact stage-1 spelling, in the right order); used only if that fit is good."""
    cats = [str(t) for t in ticks or [] if t not in (None, "", "UNKNOWN")]
    if len(cats) < 3 or len(pairs) < 2 or len(pairs) >= len(cats):
        return pairs
    low = [c.strip().lower() for c in cats]
    idx = []
    for cx, cy, t in pairs:
        k = str(t).strip().lower()
        if k not in low:
            return pairs
        idx.append(low.index(k))
    if len(set(idx)) != len(idx) or idx != sorted(idx):
        return pairs
    xs = np.array([p[0] for p in pairs], float)
    a, b = np.polyfit(np.array(idx, float), xs, 1)
    if a <= 0 or np.max(np.abs(a * np.array(idx) + b - xs)) > max_resid * a:
        return pairs
    cy = float(np.median([p[1] for p in pairs]))
    out = list(pairs)
    for i, c in enumerate(cats):
        if i not in idx:
            out.append((float(a * i + b), cy, c))
    print(f"  [GraphGrounder] categories added by position: {[c for i, c in enumerate(cats) if i not in idx]}")
    return sorted(out, key=lambda p: p[0])


def snap_stage1_colors(stage1: dict, image_path: str, box: Optional[Box] = None) -> dict:
    """Post-processing of stage 1: replace each series colour by the closest colour that is
    REALLY drawn in the image (same greedy hue matching the digitizer uses).
    Adds s["color_px"] = "#rrggbb" (None for black/gray series or when nothing matches);
    s["color"] (what the VLM said) is kept. Small VLMs often answer a CSS colour from memory
    ('#00008B' for a dark-teal line); snapping fixes most of those."""
    s1 = copy.deepcopy(stage1)
    arr = np.asarray(Image.open(image_path).convert("RGB"))
    h, w = arr.shape[:2]
    pal = dominant_colors(arr, box or (0, 0, w - 1, h - 1))
    series = [s for s in s1.get("series", []) or [] if isinstance(s, dict)]
    cmap = assign_colors([(str(i), s.get("color", "")) for i, s in enumerate(series)], pal)
    for i, s in enumerate(series):
        c = cmap.get(str(i))
        s["color_px"] = None if c is None else "#%02x%02x%02x" % tuple(int(round(v)) for v in c)
    return s1



# ═══════════════════════════════════════════════════════════════════════════
# 3b. ANNOTATION FILTER - black non-data ink inside the plot area
#
# What it removes, inside the plot box and OUTSIDE the legend / inset boxes:
#   * significance brackets  - an isolated thin black rule plus the short legs
#                              at its two ends;
#   * significance symbols   - '*', '**', '***', '#', 'n.s.', 'p<0.05', found
#                              either above a bracket or by their OCR text;
#   * rows of identical symbols at a constant y (a '+ + + +' row is an
#     annotation; a flat series is drawn with a connecting line, so its markers
#     form ONE wide component, not many separate equal blobs).
#
# What it never removes:
#   * anything coloured - every coloured series is safe by construction;
#   * THE GRIDLINES: a gridline row carries more ink than `max_len_px` even when
#     bars chop it into bracket-sized pieces, so the whole row is skipped;
#   * the axis lines and the frame (same test, plus the `edge` margin);
#   * thick shapes, so a black BAR survives (a bar is not thin);
#   * short segments, so error-bar caps survive;
#   * the tick labels, which sit outside the eroded interior.
#
# It must run on the ORIGINAL figure. On an image that already carries the
# measurement grid, the drawn gridlines are read as brackets.
# ═══════════════════════════════════════════════════════════════════════════

# OCR texts that are never data
SIG_TEXT_RX = re.compile(
    r"^(?:"
    r"[*∗٭⁎#†‡§]{1,6}"     # * ** *** # ## dagger
    r"|n\.?\s*s\.?"                                      # n.s.  ns
    r"|p\s*[<>=≤≥]\s*0?[.,]\d+"                # p<0.05
    r")$", re.I)

PLUS_TEXT_RX = re.compile(r"^[+†]{1,3}$")
PLUS_MARKERS = ("plus", "cross", "+", "x", "star", "asterisk")


def _ink(arr: np.ndarray, dark: int = 135, max_chroma: int = 45) -> np.ndarray:
    """Dark and (nearly) achromatic pixels: black ink that is not a coloured series."""
    a = arr.astype(int)
    return (a.mean(2) < dark) & ((a.max(2) - a.min(2)) <= max_chroma)


def _inside_box(shape, box: Box, shrink: float = 0.0) -> np.ndarray:
    h, w = shape[:2]
    x1, y1, x2, y2 = box
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    m = np.zeros((h, w), bool)
    m[max(0, int(y1 + shrink)):min(h, int(y2 - shrink) + 1),
      max(0, int(x1 + shrink)):min(w, int(x2 - shrink) + 1)] = True
    return m


def _thickness(d: np.ndarray, axis: int, win: int) -> np.ndarray:
    """Number of ink pixels in a window of length `win` along `axis`, per pixel."""
    k = ndimage.uniform_filter1d(d.astype(np.float32), size=win, axis=axis, mode="constant")
    return k * win


def _free_sides(d: np.ndarray, leg: Sequence[int], x: int, gap: int = 3, frac: float = 0.75) -> bool:
    """True when column `x` is an isolated vertical hairline along `leg`.
    A bracket leg has white on both sides; a steep data curve keeps going sideways,
    so without this test a near-vertical black curve is erased as a bracket leg."""
    w = d.shape[1]
    free = 0
    for yy in leg:
        l_, r_ = x - gap, x + gap
        if (l_ < 0 or not d[yy, l_]) and (r_ >= w or not d[yy, r_]):
            free += 1
    return free >= frac * len(leg)


def annotation_brackets(d: np.ndarray, inside: np.ndarray, min_len_px: int, max_len_px: int,
                        max_thick: int = 4, edge: int = 6, max_leg_frac: float = 0.35) -> np.ndarray:
    """Significance brackets: an isolated thin horizontal rule + the legs at its ends.

    Run-based on purpose. A connected-component version grows through every gridline
    the legs cross and wipes the whole grid.
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

    cand = di & (_thickness(di, 0, 2 * max_thick + 3) <= max_thick)

    for y in range(max(0, y_lo), min(d.shape[0], y_hi + 1)):
        row = cand[y]
        if not row.any():
            continue
        if int(di[y].sum()) > max_len_px:          # GRIDLINE / frame / reference-line row
            continue
        cols_in = np.flatnonzero(inside[y])
        if len(cols_in) < 2:
            continue
        x_lo, x_hi = int(cols_in[0]) + edge, int(cols_in[-1]) - edge
        for a, b in _runs(row, min_len=1, max_gap=1):
            if not (min_len_px <= b - a + 1 <= max_len_px):
                continue
            out[y, a:b + 1] = True
            for x0 in (a, b):                       # the legs, at the two ends only
                if not (x_lo <= x0 <= x_hi):        # at the frame: an axis, not a leg
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
                        if _free_sides(d, leg, x):
                            out[leg, x] = True
    return out


def _glyph_boxes(d: np.ndarray, inside: np.ndarray, max_h: int, max_w: int) -> List[Tuple[slice, slice]]:
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
    """Small blobs within ~2 text heights ABOVE a bracket: the '***' that labels it."""
    out = np.zeros_like(d)
    if not brackets.any():
        return out
    reach = int(max(6, 2.2 * text_h))
    below = np.zeros_like(brackets)
    for s in range(1, reach + 1):
        below[:-s] |= brackets[s:]
    for sl in _glyph_boxes(d, inside, max_h=int(2.0 * text_h) + 2, max_w=int(3.0 * text_h) + 4):
        if below[sl].any():
            out[sl] |= d[sl]
    return out


def token_glyphs(shape, tokens: Sequence[Any], inside: np.ndarray, drop_plus: bool) -> np.ndarray:
    """Pixels of OCR tokens whose TEXT is a significance mark (and '+' when allowed)."""
    out = np.zeros(shape[:2], bool)
    for t in tokens or []:
        txt = str(getattr(t, "text", "")).strip()
        if not txt:
            continue
        if not (SIG_TEXT_RX.match(txt) or (drop_plus and PLUS_TEXT_RX.match(txt))):
            continue
        x1, y1 = int(getattr(t, "x1", 0)), int(getattr(t, "y1", 0))
        x2, y2 = int(getattr(t, "x2", 0)), int(getattr(t, "y2", 0))
        sl = (slice(max(0, y1 - 1), y2 + 1), slice(max(0, x1 - 1), x2 + 1))
        if inside[sl].any():
            out[sl] = True
    return out


def _same_shape(d: np.ndarray, boxes: Sequence[Tuple[slice, slice]],
                min_iou: float = 0.72, grid: int = 9) -> bool:
    """Do these blobs draw the SAME glyph? Each patch is rasterised onto a small square
    and compared by IoU with the median patch. A row of '+' passes; a row of tick labels
    (one blob per digit: '1', '0', '1', '2' ...) does not."""
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
        union = float((b | ref).sum()) or 1.0
        ious.append(float((b & ref).sum()) / union)
    return float(np.mean(ious)) >= min_iou


def annotation_rows(d: np.ndarray, inside: np.ndarray, text_h: float, min_count: int = 4,
                    min_span: float = 0.35, y_tol: float = 3.0, size_tol: int = 2) -> np.ndarray:
    """A row of IDENTICAL small symbols at a constant y, spread across the plot."""
    out = np.zeros_like(d)
    cols = np.flatnonzero(inside.any(0))
    if len(cols) < 2:
        return out
    width = cols[-1] - cols[0]
    lim = int(1.8 * text_h) + 2
    boxes = _glyph_boxes(d, inside, max_h=lim, max_w=lim)
    if len(boxes) < min_count:
        return out
    hs = np.array([sl[0].stop - sl[0].start for sl in boxes])
    ws = np.array([sl[1].stop - sl[1].start for sl in boxes])
    cy = np.array([(sl[0].start + sl[0].stop) / 2.0 for sl in boxes])
    cx = np.array([(sl[1].start + sl[1].stop) / 2.0 for sl in boxes])
    used = np.zeros(len(boxes), bool)
    for i in np.argsort(-(hs * ws)):                       # biggest symbol first
        if used[i] or hs[i] < 4 or ws[i] < 4:
            continue
        same = (np.abs(hs - hs[i]) <= size_tol) & (np.abs(ws - ws[i]) <= size_tol) & ~used
        for j in np.flatnonzero(same):
            k = np.flatnonzero(same & (np.abs(cy - cy[j]) <= max(y_tol, 0.3 * text_h)))
            if len(k) < min_count:
                continue
            if (cx[k].max() - cx[k].min()) < min_span * width:
                continue
            if not _same_shape(d, [boxes[m] for m in k]):
                continue
            for m in k:
                out[boxes[m]] |= d[boxes[m]]
                used[m] = True
            break
    return out


def text_height(tokens: Sequence[Any], default: float = 11.0) -> float:
    """Median height of the OCR tokens: the natural length unit of a figure."""
    hs = [float(getattr(t, "y2", 0)) - float(getattr(t, "y1", 0)) for t in tokens or []]
    hs = [h for h in hs if 4 <= h <= 60]
    return float(np.median(hs)) if hs else default


def clean_annotations(arr: np.ndarray, box: Box, exclude: Sequence[Box] = (),
                      tokens: Sequence[Any] = (), stage1: Optional[dict] = None,
                      text_h: float = 11.0, margin: int = 3,
                      min_bracket_frac: float = 0.06, max_bracket_frac: float = 0.75,
                      glyph_dark: int = 135, line_dark: int = 190) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Return (image with the black annotations painted white, what was removed).

    box     : plot area in pixels, as plot_box_from_ticks computes it
    exclude : legend / inset boxes - left untouched, the legend must stay readable
    tokens  : OCR tokens (GraphGrounder._ocr_tokens); optional but much safer
    stage1  : used only to decide whether '+' is a real marker of some series
    """
    arr = np.asarray(arr)
    inside = _inside_box(arr.shape, box, shrink=margin)
    for ex in exclude or []:
        inside &= ~_inside_box(arr.shape, ex)

    # symbols are printed black; the bracket rules themselves are often thin gray
    d = _ink(arr, dark=glyph_dark)
    d_line = _ink(arr, dark=line_dark, max_chroma=40)

    x1, _, x2, _ = box
    min_len = int(max(10, min_bracket_frac * abs(x2 - x1)))
    max_len = int(max_bracket_frac * abs(x2 - x1))

    markers = " ".join(str((s or {}).get("marker", "")).lower()
                       for s in ((stage1 or {}).get("series") or []))
    drop_plus = not any(p in markers for p in PLUS_MARKERS)

    br = annotation_brackets(d_line, inside, min_len_px=min_len, max_len_px=max_len)
    gl = glyphs_above_brackets(d, inside, br, text_h)
    tk = token_glyphs(arr.shape, tokens, inside, drop_plus)
    # a row of tick labels sits just outside the plot and looks exactly like an
    # annotation row, so symbol rows are searched in the eroded interior only
    pad = int(max(4, 1.2 * text_h))
    core = ndimage.binary_erosion(inside, np.ones((2 * pad + 1, 1), bool))
    rw = annotation_rows(d, core, text_h) if drop_plus else np.zeros_like(d)

    kill = (br | gl | tk | rw) & inside
    kill = ndimage.binary_dilation(kill, np.ones((3, 3), bool)) & _ink(arr, dark=200, max_chroma=60) & inside

    out = arr.copy()
    out[kill] = 255
    info = {"brackets_px": int(br.sum()), "symbols_px": int(gl.sum()),
            "token_px": int(tk.sum()), "row_px": int(rw.sum()),
            "total_px": int(kill.sum()), "plus_removed": bool(drop_plus)}
    return out, info


# ═══════════════════════════════════════════════════════════════════════════
# 4. GRAPH GROUNDER - axis calibration + pixel extraction (config C)
# ═══════════════════════════════════════════════════════════════════════════
def _looks_log(ticks) -> bool:
    """Stage-1 tick labels written as powers of ten (10^-2, 10⁻², 1e-2, 0.01 0.1 1 10)."""
    vals = [parse_number(str(t)) for t in (ticks or [])]
    vals = [v for v in vals if v is not None]
    if len(vals) < 3 or min(vals) <= 0:
        return False
    lg = sorted(math.log10(v) for v in vals)
    steps = [b - a for a, b in zip(lg, lg[1:])]
    return all(abs(st - steps[0]) < 0.05 for st in steps) and steps[0] >= 0.9


# small VLMs drop / swap letters in the usual units: '%D/g', '% ID/g', '%lD/g', '%1D/g', 'ID%/g', '%ID/9'
_UNIT_FIXES = [
    (re.compile(r"%\s*[IlL1|]?\s*D\s*/\s*[g9]\b"), "%ID/g"),
    (re.compile(r"\bI?D\s*%\s*/\s*[g9]\b"), "%ID/g"),
    (re.compile(r"%\s*I\s*D\b(?!\s*/)"), "%ID"),
    (re.compile(r"\b(m|c)m\s*\^?\s*3\b|\b(m|c)m³"), lambda m: (m.group(1) or m.group(2)) + "m³"),
    (re.compile(r"\b[uµμ]\s*g\b"), "µg"),
    (re.compile(r"\b[uµμ]\s*M\b"), "µM"),
    (re.compile(r"\b[uµμ]\s*L\b"), "µL"),
]
NO_LEGEND_NAMES = {"no_legend", "nolegend", "no legend", "none", "unknown", "n/a", "-", ""}


def normalize_units(text):
    """'%D/g' -> '%ID/g', 'mm^3' -> 'mm³', 'ug' -> 'µg' ... (axis titles and series names)."""
    if not isinstance(text, str):
        return text
    t = text.replace("μ", "µ")
    for rx, rep in _UNIT_FIXES:
        t = rx.sub(rep, t)
    return t


_NUM_TICK = re.compile(r"\s*[-+−–]?\s*\d[\d.,]*\s*(%|h|hr|hrs|d|day|days|min|s|wk|week|weeks|"
                       r"nm|µm|mm|cm|mg/kg|µg|mg|g|ml|mL|µM|nM|mM|°C)?\s*", re.I)


def _tick_number(t) -> Optional[float]:
    """Number of a tick label, or None for a category. '4T1', 'A549', 'PC-3', 'HeLa' are cell
    lines, not numbers (the old fallback read them as 4, 549, -3 and built a numeric x axis)."""
    v = parse_number(str(t))
    if v is not None:
        return v
    v = to_float(t)                       # powers of ten '10^-2', '$10^{2}$' ...
    if v is not None and ("10^" in str(t) or re.search(r"[⁰¹²³⁴⁵⁶⁷⁸⁹]", str(t)) or _NUM_TICK.fullmatch(str(t))):
        return v
    return None


def normalize_stage1(stage1: dict) -> dict:
    """Fix common small-VLM stage-1 habits before extraction:
    * unit typos in the axis titles ('%D/g' -> '%ID/g', see normalize_units);
    * a chart without legend (series 'NO_LEGEND' / 'unknown' / empty name) -> one series
      named after the y-axis title (the name is not evaluated for single-series charts);
    * a single-series bar chart listed as one series PER BAR (series names = x categories,
      all with the same colour) -> merged into one series;
    * the same series listed twice -> kept once."""
    if not isinstance(stage1, dict):
        return stage1
    s1 = copy.deepcopy(stage1)
    for k in ("x_axis", "y_axis"):
        if k in s1:
            s1[k] = normalize_units(s1[k])
    series = [s for s in s1.get("series", []) or [] if isinstance(s, dict)]
    for s in series:
        s["name"] = normalize_units(str(s.get("name", "")).strip().strip("\"'`"))
        if s["name"].lower() in NO_LEGEND_NAMES:
            s["name"] = s1.get("y_axis") or "value"
    seen, uniq = set(), []
    for s in series:
        key = str(s.get("name", "")).strip().lower()
        if key not in seen:
            seen.add(key)
            uniq.append(s)
    series = uniq
    cats = [str(t).strip().lower() for t in s1.get("x_ticks") or []]
    def is_bar(s):
        return str(s.get("marker", "")).lower() in ("none", "", "unknown", "-") and \
            str(s.get("line", "")).lower() in ("none", "", "unknown", "-")
    if len(series) >= 2 and cats and all(is_bar(s) for s in series):
        hits = sum(bool(difflib.get_close_matches(str(s.get("name", "")).strip().lower(), cats, n=1, cutoff=0.8))
                   for s in series)
        colours = [str(s.get("color", "")).strip().lower() for s in series]
        common = max(set(colours), key=colours.count)
        # a grouped bar chart whose LEGEND names also appear on the x axis (e.g. 'Control')
        # has a different colour per series -> only merge when (almost) all colours are the same
        same_colour = colours.count(common) >= max(2, 0.8 * len(series))
        if hits >= 0.8 * len(series) and (same_colour or len(series) > len(cats)):
            common = max(set(colours), key=colours.count)
            series = [{"name": s1.get("y_axis") or "value", "color": common, "marker": "none", "line": "none"}]
    s1["series"] = series
    return s1


def legend_box_from_tokens(tokens, names, img_w: int) -> Optional[Tuple[float, float, float, float]]:
    """Legend box from OCR text: the tokens that spell the series names (>= 2 of them, close
    together), extended to the LEFT by ~3.5 text heights for the symbol / line sample.
    Exact pixels and no VLM call; returns None when the legend text is not found."""
    names = [str(n) for n in names if str(n).strip()]
    if len(names) < 2 or not tokens:
        return None
    norm = lambda t: re.sub(r"\s+", "", str(t)).lower()
    targets = [norm(n) for n in names]
    hits, used = [], set()
    for t in tokens:
        m = difflib.get_close_matches(norm(t.text), targets, n=1, cutoff=0.75)
        if m and m[0] not in used:
            used.add(m[0])
            hits.append(t)
    if len(hits) < 2:
        return None
    h = float(np.median([t.y2 - t.y1 for t in hits]))
    x1, y1 = min(t.x1 for t in hits), min(t.y1 for t in hits)
    x2, y2 = max(t.x2 for t in hits), max(t.y2 for t in hits)
    if (x2 - x1) > 0.8 * img_w or (y2 - y1) > (len(hits) + 1) * 2.5 * h:   # scattered: not one legend
        return None
    return (max(0.0, x1 - 3.5 * h), y1 - 0.4 * h, x2 + 0.4 * h, y2 + 0.4 * h)


def legend_entries(tokens, names) -> Dict[str, List[Any]]:
    """{series name: OCR tokens that spell it, best match first}. A name can appear twice
    (legend AND x-axis category, e.g. 'Control'); the swatch test picks the legend one."""
    norm = lambda t: re.sub(r"\s+", "", str(t)).lower()
    out: Dict[str, List[Any]] = {}
    for n in [str(n) for n in names if str(n).strip()]:
        # a number is a tick label, not legend text (series '1','2','3' = week numbers under the
        # bars: their "swatch" was read next to the x labels and made week 3 gray)
        c = [(difflib.SequenceMatcher(None, norm(t.text), norm(n)).ratio(), i, t) for i, t in enumerate(tokens or [])
             if getattr(t, "num", None) is None]
        c = [x for x in c if x[0] >= 0.75]
        if c:
            out[n] = [t for _, _, t in sorted(c, key=lambda x: (-x[0], x[1]))]
    return out


def _text_start_x(t, name: str) -> float:
    """OCR often glues the legend symbol to the name ('mm Conj-HSLN', 'a -@GEM-LNP'): find where
    the name starts inside the token text and convert that character offset to a pixel x."""
    txt = str(t.text)
    if not name or len(txt) <= len(name):
        return float(t.x1)
    best_k = max(range(len(txt) - len(name) // 2),
                 key=lambda k: difflib.SequenceMatcher(None, txt[k:].lower(), name.lower()).ratio() - 0.001 * k)
    return float(t.x1 + (t.x2 - t.x1) * best_k / len(txt))


def _swatch(arr: np.ndarray, t, name: str = "") -> Optional[Dict[str, Any]]:
    """Colour of the symbol just LEFT of one legend text token."""
    h_img = arr.shape[0]
    h = max(4.0, t.y2 - t.y1)
    tx = _text_start_x(t, name)
    x1, x2 = int(max(0, tx - 3.5 * h)), int(max(0, tx - 1))
    y1, y2 = int(max(0, t.y1 - 0.2 * h)), int(min(h_img, t.y2 + 0.2 * h))
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None
    px = arr[y1:y2, x1:x2].reshape(-1, 3).astype(int)
    chroma = px[(px.max(1) - px.min(1)) >= 20]
    if len(chroma) >= 4:
        sat = chroma.max(1) - chroma.min(1)
        core = chroma[sat >= np.median(sat)]          # core of the symbol, not its anti-aliased rim
        c = np.median(core, axis=0)
        return {"hex": "#%02x%02x%02x" % tuple(int(v) for v in c), "kind": "chromatic", "n_px": int(len(chroma)),
                "token": t}
    dark = px[px.max(1) < 170]
    if len(dark) >= 4:
        c = np.median(dark, axis=0)
        return {"hex": "#%02x%02x%02x" % tuple(int(v) for v in c), "kind": "achromatic", "n_px": int(len(dark)),
                "token": t}
    return None


def legend_swatch_colors(arr: np.ndarray, entries: Dict[str, List[Any]]) -> Dict[str, Dict[str, Any]]:
    """Colour of each legend symbol, read from the pixels just LEFT of its legend text
    (marker / line sample / bar swatch). For every name the candidate texts are tried in
    order and the first one with a COLOURED symbol wins (an x-axis label has none).
    Returns {name: {"hex", "kind", "n_px"}}; kind "achromatic" = black/gray symbol.
    Small VLMs name colours from memory ('#00008B' for dark teal); the pixels do not lie."""
    out = {}
    for name, toks in entries.items():
        if not isinstance(toks, (list, tuple)):
            toks = [toks]
        found = [sw for sw in (_swatch(arr, t, name) for t in toks) if sw]
        chrom = [sw for sw in found if sw["kind"] == "chromatic"]
        if chrom:
            out[name] = chrom[0]
        elif found:
            out[name] = found[0]
    return out


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
                 tick_source: str = "ocr", ocr_vlm_classify: bool = False, ocr_backend: str = "auto",
                 legend_colors: bool = True, clean_annotations: bool = True):
        """tick_source: "ocr" (OCR boxes, VLM only as fallback) or "vlm".
        ocr_backend: "auto" (tesseract if installed, else rapidocr), "tesseract" or "rapidocr".
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
        self.ocr_backend = ocr_backend          # "auto" | "tesseract" | "rapidocr"
        self.legend_colors = legend_colors      # read series colours from the legend symbols (pixels)
        self.clean_annotations = clean_annotations   # paint the black annotations white first
        self.last_stage1: Dict[str, Any] = {}
        self.last_masks: Dict[str, np.ndarray] = {}   # stage 1 actually used (normalised, colours from legend)
        self._ocr_axes: Optional[Dict[str, Any]] = None
        self.last_diagnostics: Dict[str, Any] = {}
        self.last_calibration: Dict[str, Any] = {}
        self.last_points_px: Dict[str, List[Dict[str, Any]]] = {}   # {series: [{x_px, y_px, x, y}]}
        self.last_clean_arr: Optional[np.ndarray] = None   # figure with the black annotations removed
        self.last_annotations: Dict[str, Any] = {}

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
    def _axis_from_pairs(pairs, axis, log_hint: Optional[bool] = None):
        pos = [(p[0] if axis == "x" else p[1], p[2]) for p in pairs]
        nums = [(p, _tick_number(t)) for p, t in pos]
        if len(nums) >= 2 and all(v is not None for _, v in nums):
            return NumericAxis([p for p, _ in nums], [v for _, v in nums], log_hint=log_hint), pairs
        if pos:
            return CategoricalAxis([p for p, _ in pos], [t for _, t in pos]), pairs
        return None, []

    def _build_axis(self, image_path, axis, ticks, space, scale, gray):
        ocr_pairs = (self._ocr_axes or {}).get(axis) or []
        log_hint = _looks_log(ticks) or \
            ((self._ocr_axes or {}).get("info", {}).get(axis, {}).get("type") == "log")
        exp = [v for v in (_tick_number(t) for t in (ticks or []) if t not in (None, "", "UNKNOWN")) if v is not None]
        if len(ocr_pairs) == 2 and len(exp) >= 3:
            # two points always fit a line, so a misread pair ('90' + '7' read from '0') would be
            # accepted silently: with only 2 OCR ticks both values must be in the stage-1 list
            vals = [_tick_number(p[2]) for p in ocr_pairs]
            if not all(v is not None and any(abs(v - e) <= 1e-6 * max(1.0, abs(e)) for e in exp) for v in vals):
                print(f"  [GraphGrounder] OCR {axis} ticks {[p[2] for p in ocr_pairs]} not in stage 1 -> not used")
                ocr_pairs = []
        if len(ocr_pairs) >= 2:                               # exact OCR boxes: no VLM call
            self._tick_source_used[axis] = "ocr"
            if axis == "x" and not exp:                       # categories: OCR can miss one
                ocr_pairs = complete_categories(ocr_pairs, ticks)
            return self._axis_from_pairs(ocr_pairs, axis, log_hint)
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

        return self._axis_from_pairs(pairs, axis, log_hint)

    def _region(self, what, image_path, space, scale):
        if not self.find_regions:
            return None
        items = parse_json_list(self._ask(build_region_prompt(what), image_path, scale))
        for it in items:
            if len(it.get("bbox_2d", [])) == 4:
                return space.bbox(it["bbox_2d"])
        return None

    def _categories_from_bars(self, arr, gray, stage1, y_anc):
        """Bar charts whose category labels OCR cannot read (rotated, long names): find the
        bars by colour, group them per category, and give the groups the stage-1 category
        names in order (left to right). Used only if the group count matches exactly."""
        cats = [str(t) for t in (stage1.get("x_ticks") or []) if t not in (None, "", "UNKNOWN")]
        bars = [s for s in stage1.get("series", []) if self._is_bar(s)]
        if len(cats) < 2 or not bars or sum(parse_number(c) is not None for c in cats) > len(cats) / 2:
            return None, []
        h, w = gray.shape
        y_ax, x_ax = find_axis_lines(gray)
        ys = sorted(a[1] for a in y_anc)
        left = (x_ax + 2) if x_ax is not None else float(np.median([a[0] for a in y_anc])) + 15
        bottom = y_ax if y_ax is not None else min(h - 1, ys[-1] + 2)
        box = (left, max(0, ys[0] - 0.5 * (ys[-1] - ys[0]) / max(1, len(ys) - 1)), w - 2, bottom)
        palette = dominant_colors(arr, box, [])
        cmap = assign_colors([(s.get("name", ""), s.get("color", "")) for s in bars], palette)
        region = _region_mask(arr.shape, box, [])
        zero_px = None
        centers = []
        for s in bars:
            rgb = cmap.get(s.get("name", ""))
            if rgb is None:
                continue
            m = color_mask(arr, rgb, region, palette=[c for c, _ in palette])
            centers += [c[0] for c in extract_bars(m, zero_px)]
        if len(centers) < len(cats):
            return None, []
        centers = sorted(centers)
        gaps = np.diff(centers)
        if len(cats) > 1:                                  # split at the largest gaps -> categories
            cut = sorted(np.argsort(gaps)[-(len(cats) - 1):])
            groups, start = [], 0
            for c in cut:
                groups.append(centers[start:c + 1])
                start = c + 1
            groups.append(centers[start:])
        else:
            groups = [centers]
        if len(groups) != len(cats) or any(len(g) == 0 for g in groups):
            return None, []
        sizes = [len(g) for g in groups]
        if max(sizes) - min(sizes) > 1:                     # uneven groups -> unreliable
            return None, []
        pos = [float(np.mean(g)) for g in groups]
        pairs = [(p, float(bottom) + 10, c) for p, c in zip(pos, cats)]
        return CategoricalAxis(pos, cats), pairs

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
        stage1 = normalize_stage1(stage1)
        img = Image.open(image_path).convert("RGB")
        arr = np.array(img)
        gray = np.array(img.convert("L")).astype(int)
        space = CoordSpace(img.width, img.height, self.coord_mode, scale)

        self._ocr_axes, self._tick_source_used, self._ocr_tokens = None, {}, []
        self.last_points_px, self.last_calibration, self.last_masks = {}, {}, {}
        self.last_palette, self.last_series_rgb = [], {}
        self.last_clean_arr, self.last_annotations = None, {}
        if self.tick_source == "ocr":
            try:
                toks = ocr_tokens(image_path, backend=self.ocr_backend)
                self._ocr_tokens = toks
                labels = None
                if self.ocr_vlm_classify and toks:
                    work_dir = debug_dir or os.path.dirname(image_path) or "."
                    os.makedirs(work_dir, exist_ok=True)          # debug_dir may not exist yet
                    work = os.path.join(work_dir, os.path.splitext(os.path.basename(image_path))[0] + "_ocr_marks.png")
                    try:                                          # a failed labelling must not lose the OCR
                        labels = classify_tokens_with_vlm(self.vlm, self.template, image_path, toks, work)
                    except Exception as e:
                        print(f"  [GraphGrounder] token labelling failed ({type(e).__name__}: {e}); geometry only")
                        labels = None
                self._ocr_axes = find_ticks(image_path, stage1, tokens=toks, vlm_labels=labels,
                                            backend=self.ocr_backend)
                print(f"  [GraphGrounder] OCR ticks: {self._ocr_axes['info']}")
            except Exception as e:                            # tesseract missing, etc. -> VLM path
                print(f"  [GraphGrounder] OCR failed ({e}); using VLM tick boxes")

        legend_cols = {}
        if self.use_color and self.legend_colors and self._ocr_tokens and len(stage1.get("series", [])) >= 2:
            legend_cols = legend_swatch_colors(arr, legend_entries(self._ocr_tokens,
                                                                   [s.get("name", "") for s in stage1["series"]]))
            for s in stage1["series"]:
                c = legend_cols.get(s.get("name", ""))
                if c and c["kind"] == "chromatic":        # decided below, once the plot colours are known
                    s["color_legend"] = c["hex"]
        self.last_stage1 = stage1

        x_map, x_anc = self._build_axis(image_path, "x", stage1.get("x_ticks"), space, scale, gray)
        y_map, y_anc = self._build_axis(image_path, "y", stage1.get("y_ticks"), space, scale, gray)

        result: Dict[str, Any] = {"x_axis": stage1.get("x_axis"), "y_axis": stage1.get("y_axis")}
        self.last_diagnostics = {"x_axis": x_map.info() if x_map else None,
                                 "y_axis": y_map.info() if y_map else None,
                                 "tick_source": dict(self._tick_source_used),
                                 "legend_colors": {n: c["hex"] for n, c in legend_cols.items()}}
        overlay: Dict[str, Any] = {"x": x_anc, "y": y_anc, "box": None, "exclude": [], "series": {}}

        if x_map is None and y_map is not None and self.use_color:
            x_map, x_anc = self._categories_from_bars(arr, gray, stage1, y_anc)
            if x_map is not None:
                self._tick_source_used["x"] = "bar_positions"
                self.last_diagnostics["x_axis"] = x_map.info()
                self.last_diagnostics["tick_source"] = dict(self._tick_source_used)
                overlay["x"] = x_anc

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
                r = None
                if what == "legend":          # exact box from the OCR'd legend text, no VLM call
                    sw_toks = [c["token"] for c in legend_cols.values() if c["kind"] == "chromatic"]
                    names = [s.get("name", "") for s in stage1.get("series", [])]
                    # the texts that have a coloured symbol next to them ARE the legend (not x labels)
                    r = (legend_box_from_tokens(sw_toks, names, img.width) if len(sw_toks) >= 2 else None) \
                        or legend_box_from_tokens(self._ocr_tokens, names, img.width)
                    self.last_diagnostics["legend_source"] = "ocr" if r else "vlm"
                if r is None:
                    r = self._region(what, image_path, space, scale)
                if r and not (r[2] < box[0] or r[0] > box[2] or r[3] < box[1] or r[1] > box[3]):
                    exclude.append(r)
        overlay["box"], overlay["exclude"] = box, exclude

        # black non-data ink (brackets, '*', '#', rows of symbols) inside the plot box and
        # outside the legend / inset. The gridlines, the axes and the bars are kept.
        if self.clean_annotations:
            cleaned, ann = clean_annotations(arr, box, exclude=exclude, tokens=self._ocr_tokens,
                                             stage1=stage1, text_h=text_height(self._ocr_tokens))
            self.last_annotations = ann
            self.last_diagnostics["annotations"] = ann
            if ann["total_px"] > 0:
                print(f"  [GraphGrounder] annotation filter: {ann}")
                arr = cleaned                     # the digitizer reads the clean pixels
                self.last_clean_arr = cleaned     # grid_extract draws the grid on them

        region = _region_mask(arr.shape, box, exclude)
        zero_px = y_map.pixel(0.0) if isinstance(y_map, NumericAxis) and not y_map.log else None

        series = stage1.get("series", [])
        masks: Dict[str, np.ndarray] = {}
        if self.use_color and series:
            palette = dominant_colors(arr, box, exclude)
            choose_series_colors(series, palette)
            self.last_diagnostics["series_colors"] = {s.get("name"): {k: s.get(k) for k in
                                                      ("color", "color_vlm", "color_legend", "color_source")}
                                                      for s in series}
            cmap = assign_colors([(s.get("name", "UNKNOWN"), s.get("color", "")) for s in series], palette)
            pal = [c for c, _ in palette]
            # for fade_other_series: every colour drawn in the plot + which one is whose
            self.last_palette = [tuple(float(v) for v in c) for c in pal]
            self.last_series_rgb = {n: tuple(float(v) for v in c) for n, c in cmap.items() if c is not None}
            for s in series:
                rgb = cmap.get(s.get("name", "UNKNOWN"))
                if rgb is not None:
                    masks[s.get("name", "UNKNOWN")] = color_mask(arr, rgb, region, palette=pal)
        line_masks = {s.get("name", "UNKNOWN"): masks[s.get("name", "UNKNOWN")]
                      for s in series if s.get("name", "UNKNOWN") in masks and not self._is_bar(s)}
        lines = extract_line_series(line_masks, [a[0] for a in x_anc]) if line_masks else {}
        # kept for grid_extract (direct reading with a value grid)
        shared_px = shared_x_positions([pts for pts, kind in lines.values() if kind != "ticks"])
        self.last_masks = masks                 # {series: bool mask of its pixels}
        self.last_calibration = {
            "x_map": x_map if isinstance(x_map, NumericAxis) else None, "y_map": y_map,
            "box": box, "exclude": exclude,
            "data_x": [round(x_map.value(p), 6) for p in shared_px] if isinstance(x_map, NumericAxis) else [],
            # categorical x (bars): [(pixel, label), ...] so other methods can draw at the category
            "x_cat": [(float(p), str(l)) for p, l in zip(x_map.pix, x_map.labels)]
                     if isinstance(x_map, CategoricalAxis) else [],
            # tick labels found by OCR / VLM, in ORIGINAL-image pixels: [(cx, cy, "text"), ...]
            "x_ticks_px": [(round(float(a), 2), round(float(b), 2), t) for a, b, t in x_anc],
            "y_ticks_px": [(round(float(a), 2), round(float(b), 2), t) for a, b, t in y_anc],
        }
        self.last_points_px = {}

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
            chromatic = name_to_rgb(cname) is not None and not is_achromatic(cname)
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
            # data points in ORIGINAL-image pixels, same order as result[name]
            self.last_points_px[name] = [{"x_px": round(float(x), 2), "y_px": round(float(y), 2),
                                          "x": v[0], "y": v[1]} for (x, y), v in zip(px_pts, out)]
            self.last_diagnostics[name] = {"kind": "bars" if bar else "points", "method": method,
                                           "n": len(out), **({"note": note} if note else {})}

        if debug_dir:
            self._save_overlay(img, overlay, debug_dir, image_path)
        return result

    # debug image 
    @staticmethod
    def _save_overlay(img, ov, debug_dir, image_path, zoom: int = 2):
        os.makedirs(debug_dir, exist_ok=True)
        im = img.resize((img.width * zoom, img.height * zoom), Image.LANCZOS)
        d = ImageDraw.Draw(im)
        Z = lambda *v: [c * zoom for c in v]
        def _box(b):                                       # PIL needs x0<=x1, y0<=y1
            x1, y1, x2, y2 = b
            return Z(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        if ov.get("box"):
            d.rectangle(_box(ov["box"]), outline=(150, 150, 150))
        for ex in ov.get("exclude", []):
            d.rectangle(_box(ex), outline=(255, 140, 0), width=2)
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


# 5. GRID OVERLAY - value grid + one VLM reading per series (config D)
FONT_PATHS = ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "arial.ttf"]


def _font(size: int):
    for p in FONT_PATHS:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


# axis helpers 
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
    def from_axis(cls, axis):                       # NumericAxis
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


#  drawing 
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


# fading the other series 
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
    that, a figure with two clusters of ONE hue (a blue curve and blue dosing arrows) fades
    the target series itself: its own pixels are nearer the other cluster than to keep_rgb."""
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
    else:                                                 # the nearest colour must be ANOTHER series,
        k = np.asarray(keep_rgb, float)                   # not a second cluster of the same hue
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


def series_file_tags(names: Sequence[str], max_len: int = 40) -> List[str]:
    """Series name -> safe file-name part ('Conj-HSLN' -> 'Conj-HSLN', '% ID/g' -> 'ID_g').
    Two series that give the same tag get _2, _3 ...; an empty tag becomes s<index>."""
    out, seen = [], {}
    for i, n in enumerate(names):
        t = re.sub(r"[^A-Za-z0-9.+-]+", "_", str(n)).strip("._")[:max_len] or f"s{i}"
        if t.lower() in ("read", "raw", "clean"):         # would clash with <base>_grid_read.png
            t = f"series_{t}"
        k = t.lower()
        seen[k] = seen.get(k, 0) + 1
        out.append(t if seen[k] == 1 else f"{t}_{seen[k]}")
    return out


def fade_target_ok(series: Dict[str, Any], keep_rgb: Optional[Sequence[float]]) -> Tuple[bool, str]:
    """May the other series be faded for this one? Never when it could fade the target itself:
    * a coloured series whose plot colour was not found (keep_rgb None would fade ALL colours);
    * the plot colour assigned to it is far from its legend swatch (wrong colour match)."""
    cname = str(series.get("color", ""))
    if is_achromatic(cname):
        return True, "black/gray series: all colours faded"
    want = name_to_rgb(cname)
    if keep_rgb is None:
        return False, "colour of the series not found in the plot"
    if want is not None and colour_match_score(want, keep_rgb, 1, 1) is None:
        return False, f"plot colour {tuple(int(v) for v in keep_rgb)} far from legend colour {cname}"
    return True, "ok"


# pipeline 
def grid_extract(grounder, image_path: str, stage1: dict, out_dir: str, scale: float = 1.0,
                 ask_at_data_x: bool = True, fade_others: bool = True) -> Dict[str, Any]:
    """Calibrate with GraphGrounder, draw the value grid, ask the VLM ONE series per
    call, and compare with the pixel result.

    grounder : GraphGrounder (its VLM and prompt template are reused)
    fade_others : one grid image PER SERIES in which the other coloured series are faded
                  (<base>_grid_s<i>.png); the plain <base>_grid.png is kept for the overlay.
    Returns {"grid": {series: [[x, y], ...]}, "pixel": {...}, "agreement": {...}, "grid_image": path}
    """
    stage1 = normalize_stage1(stage1)
    # calibration only: debug_dir=None, so C's *_grounded.png is NOT written into D's folder
    pixel = grounder.extract(image_path, stage1, scale=scale, debug_dir=None)
    cal = getattr(grounder, "last_calibration", {}) or {}
    stage1 = getattr(grounder, "last_stage1", None) or stage1    # colours read from the legend symbols
    palette = getattr(grounder, "last_palette", []) or []
    series_rgb = getattr(grounder, "last_series_rgb", {}) or {}
    if not cal.get("y_map"):
        return {"grid": {}, "pixel": pixel, "agreement": {}, "error": "axis calibration failed"}

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]
    grid_path = os.path.join(out_dir, f"{base}_grid.png")

    # the grounder already removed the black annotations: read the grid from those pixels,
    # so the VLM cannot mistake a bracket or a row of symbols for a data point either
    src = image_path
    clean = getattr(grounder, "last_clean_arr", None)
    if clean is not None:
        src = os.path.join(out_dir, f"{base}_clean.png")
        Image.fromarray(np.asarray(clean).astype(np.uint8)).save(src)

    data_x = cal.get("data_x") if ask_at_data_x else None
    g = draw_value_grid(src, grid_path, cal.get("x_map"), cal["y_map"], cal["box"],
                        exclude=cal.get("exclude", ()), x_positions=data_x or None)

    out: Dict[str, Any] = {}
    faded_paths: Dict[str, str] = {}
    all_series = stage1.get("series", [])
    tags = series_file_tags([s.get("name", "UNKNOWN") for s in all_series])
    for i, s in enumerate(all_series):
        name = s.get("name", "UNKNOWN")
        tag = tags[i]
        ask_path, faded = grid_path, False
        ok_fade, fade_note = fade_target_ok(s, series_rgb.get(name))
        if fade_others and not ok_fade:
            print(f"  [grid_extract] {name}: NOT fading the others ({fade_note})")
        if fade_others and ok_fade and len(stage1.get("series", [])) > 1 and palette:
            tmp = os.path.join(out_dir, f"{base}_faded_{tag}.png")
            if fade_other_series(src, palette, series_rgb.get(name), cal["box"], cal.get("exclude", ()), tmp):
                ask_path = os.path.join(out_dir, f"{base}_grid_{tag}.png")
                draw_value_grid(tmp, ask_path, cal.get("x_map"), cal["y_map"], cal["box"],
                                exclude=cal.get("exclude", ()), x_positions=data_x or None)
                os.remove(tmp)
                faded, faded_paths[name] = True, ask_path
        fm = PromptFormatter(**build_grid_read_prompt(s, g, data_x or None, faded=faded))
        fm.model_post_init(grounder.template)
        raw = grounder.vlm.run_image_single_prompt_rescale(fm.format_prompt("<image>"), ask_path, scale=1.0)
        print(f"\n  [grid_extract] {s.get('name')} RAW:\n{raw}")
        pts = parse_grid_answer(raw)
        if cal.get("x_map") is None:          # categorical x: 'HSLN' must not become 'Conj-HSLN'
            pts = match_categories(pts, [lab for _, lab in cal.get("x_cat", [])],
                                   extra=[str(t) for t in stage1.get("x_ticks") or [] if t not in (None, "")])
        out[s.get("name", "UNKNOWN")] = pts

    read_path = os.path.join(out_dir, f"{base}_grid_read.png")
    try:
        draw_grid_answers(grid_path, read_path, out, g, cal)
    except Exception as e:                                  # an overlay must never break the run
        print(f"  [grid_extract] overlay not saved: {type(e).__name__}: {e}")
    return {"grid": out, "pixel": pixel, "agreement": compare_series(out, pixel, cal["y_map"]),
            "grid_image": grid_path, "grid_read_image": read_path, "faded_grid_images": faded_paths,
            "clean_image": src if src != image_path else None,
            "annotations": getattr(grounder, "last_annotations", {})}


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