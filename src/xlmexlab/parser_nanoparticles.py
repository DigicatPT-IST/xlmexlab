#parser
import re
from collections import OrderedDict
from pydantic import BaseModel, PrivateAttr
import Levenshtein
from typing import List, Dict, Any, Optional
from xlmexlab.nanoparticle_paragraph import CARGO_DB, cargo_map

cargo_list = sorted({
    x for vals in CARGO_DB.values() for x in vals
})


# 2. HELPERS
SEP = r"[-_\s/]+"          # flexible separator
RNA = r"(?:rna)?"
WORD = r"\b"

def rx(term):
    return WORD + term + WORD

COMPILED_MAP = [
    (re.compile(pattern, re.I), canonical)
    for pattern, canonical in cargo_map.items()
]


ORGAN_MAP = {
    # Bladder
    "bladder": "bladder",
    "urinary bladder": "bladder",

    # Brain / CNS
    "brain": "brain",
    "cerebral": "brain",
    "cortex": "brain",
    "hippocampus": "brain",
    "hypothalamus": "brain",
    "pineal": "brain",
    "pituitary": "brain",
    "pituitary gland": "brain",
    "spinal cord": "brain",
    "neural": "brain",
    "nerve": "brain",

    #bone marrow
    "bone marrow": "bone marrow",

    #bone
    "bone": "bone",

    # Heart
    "heart": "heart",
    "cardiac": "heart",
    "myocardial": "heart",

    #Intestine
    "intestine": "intestine",
    "intestinal": "intestine",
    "gut": "intestine",
    "bowel": "intestine",
    "duodenum": "intestine",
    "jejunum": "intestine",
    "ileum": "intestine",
    "colon": "intestine",
    "colonic": "intestine",
    "rectum": "intestine",
    "rectal": "intestine",
    "appendix": "intestine",
    "appendiceal": "intestine",

    #Kidney
    "kidney": "kidney",
    "renal": "kidney",
    "nephric": "kidney",

    # Heart
    "heart": "heart",
    "cardiac": "heart",
    "myocardial": "heart",
    
    #Liver
    "liver": "liver",
    "hepatic": "liver",
    "hepatocyte": "liver",
    "hepatocellular": "liver",

    # Lung
    "lung": "lung",
    "pulmonary": "lung",
    "bronchus": "lung",
    "bronchial": "lung",
    "alveolar": "lung",
    
    #Muscle
    "muscle": "muscle",
    "skeletal muscle": "muscle",

    # Spleen
    "spleen": "spleen",
    "splenic": "spleen",

    # Stomach
    "stomach": "stomach",
    "gastric": "stomach",

    #Thymus
    "thymus": "thymus",

    #Thyroid gland
    "thyroid gland": "thyroid gland",

#não aparecem no paper
    # Pancreas -> nao aparece
    "pancreas": "pancreas",
    "pancreatic": "pancreas",

    # Gallbladder / biliary
    "gallbladder": "gallbladder",
    "biliary": "gallbladder",
    "bile duct": "gallbladder",

    # Blood / circulation
    "blood": "blood",
    "plasma": "blood",
    "serum": "blood",
    "vascular": "blood vessel",
    "vessel": "blood vessel",
    "aorta": "blood vessel",
    "artery": "blood vessel",
    "arterial": "blood vessel",
    "vein": "blood vessel",
    "venous": "blood vessel",
    "capillary": "blood vessel",

    # Lymphatic
    "lymph node": "lymph node",
    "lymphatic": "lymph node",
    "tonsil": "tonsil",
    "tonsillar": "tonsil",

    # Reproductive male
    "prostate": "prostate",
    "prostatic": "prostate",
    "testis": "testis",
    "testes": "testis",
    "testicular": "testis",

    # Reproductive female
    "ovary": "ovary",
    "ovarian": "ovary",
    "uterus": "uterus",
    "uterine": "uterus",
    "cervix": "cervix",
    "cervical": "cervix",
    "vagina": "vagina",
    "vaginal": "vagina",
    "breast": "breast",
    "mammary": "breast",

    # Endocrine
    "thyroid": "thyroid",
    "adrenal": "adrenal gland",
    "adrenal gland": "adrenal gland",

    # Head / ENT
    "eye": "eye",
    "ocular": "eye",
    "retina": "eye",
    "ear": "ear",
    "otic": "ear",
    "nose": "nose",
    "nasal": "nose",
    "sinus": "nose",
    "pharynx": "throat",
    "larynx": "throat",
    "trachea": "throat",
    "esophagus": "esophagus",
    "esophageal": "esophagus",

    # Skin / soft tissue
    "skin": "skin",
    "cutaneous": "skin",
    "fat": "adipose tissue",
    "adipose": "adipose tissue",
    "soft tissue": "soft tissue",
    "cartilage": "cartilage",
    "tendon": "tendon",
    "ligament": "ligament",
    "joint": "joint",
    "synovial": "joint",


    # Body cavities / membranes
    "peritoneum": "peritoneum",
    "peritoneal": "peritoneum",
    "pleura": "pleura",
    "pleural": "pleura",
    "pericardium": "pericardium",
    "pericardial": "pericardium",
    "diaphragm": "diaphragm",

    # Tumor / disease tissue
    "tumor": "tumor",
    "tumour": "tumor",
    "xenograft": "tumor",
    "metastasis": "tumor",
}

DEFAULT_FIELDS = ["parameter_name", "value", "unit", "condition"]

PARAM_META = {

    "size_nm": {
        "fields": [
            "parameter_name", "value", "unit", "formulation", "size_type", 
        ],
    },

    "zeta_potential_mv": {
        "fields": [
            "parameter_name", "value", "unit", "condition"
        ],
    },

    "bioconjugation_nature": {
        "fields": [
            "bioconjugation_nature", "nature", "formulation"
        ],
    },

    "pdi": {
        "fields": [
            "parameter_name", "value", "methodology"
        ],
    },

    "encapsulation_efficiency_pct": {
        "fields": [
            "parameter_name", "value", "unit", "formulation"
        ],
    },

    "ic50": {
        "fields": [
            "parameter_name", "value", "unit", "formulation", "cell_line"
        ],
    },

    "distribution_half_life_h": {
        "fields": [
            "parameter_name", "value", "unit", "condition"
        ],
    },

    "circulation_half_life_h": {
        "fields": [
            "parameter_name", "value", "unit", "condition"
        ],
    },

    "dose_group": {
        "fields": [
            "parameter_name", "value", "unit", "formulation", "schedule",
        ],
    },

    "tumor_reduction": {
        "fields": [
            "parameter_name", "value", "unit", "formulation",
        ],
    },

    "tumor_size_or_volume": {
        "fields": [
            "parameter_name", "value", "unit", "formulation", "state", "comparasion",
        ],
    },

    "delivery_efficiency": {
        "fields": [
            "parameter_name", "value", "unit", "condition"
        ],
    },

    "biodistribution": {
        "fields": [
            "parameter_name", "value", "unit", "condition"
        ],
    },
}

class ParserNanoparticle(BaseModel):
    _parameters: List[str] = PrivateAttr(default_factory=list)

    def correct_param(self, param:str, parameters: List[str])-> str:
        best_match = param
        best_score = 0

        for p in parameters:
            score = Levenshtein.ratio(param, p)

            if score > best_score:
                best_score = score
                best_match = p

        if best_score == 1:
            return param  # já é perfeito
        else:
            return best_match  # substitui pelo mais próximo
        
    def normalize_value(self, field: str, value):
        if value is None:
            return None
        
        if value in ("", "-", "none", "not_extractable"):
            return None

        value = value.strip()

        # convert numeric values automatically
        if field == "value":
            try:
                return float(value)
            except Exception:
                return value

        return value
    
   

    def _is_param_label(self, token: str, known_param: str) -> bool:
        """Heuristic: does this token look like it's naming known_param,
        rather than being a real data value?"""
        if not token:
            return False

        score = Levenshtein.ratio(token.strip().lower(), known_param.strip().lower())
        return score >= 0.6


    def parse_response(self, response: str, known_param: str | None = None) -> dict[str, list[dict]]:

        results = {}

        for raw_line in response.strip().splitlines():

            line = raw_line.strip()

            if not line or line.startswith("#"):
                continue

            parts = [p.strip() for p in line.split("|")]

            if len(parts) < 1:
                continue

            if known_param is not None:
                param = known_param
                meta = PARAM_META.get(param, {})
                fields = meta.get("fields", DEFAULT_FIELDS)
                data_fields = fields[1:]

                # decide per-line whether parts[0] is the label or real data
                if self._is_param_label(parts[0], known_param):
                    start_idx = 1
                else:
                    start_idx = 0
            else:
                raw_param = parts[0]
                param = self.correct_param(raw_param, self._parameters)
                meta = PARAM_META.get(param, {})
                fields = meta.get("fields", DEFAULT_FIELDS)
                data_fields = fields[1:]
                start_idx = 1

            entry = {}
            for offset, field in enumerate(data_fields):
                idx = start_idx + offset
                value = parts[idx] if idx < len(parts) else None
                value = self.normalize_value(field, value)
                entry[field] = value

            results.setdefault(param, []).append(entry)

        return results

    def detect_organ(self, text: str, organ_map: dict) -> str:
        text_low = text.lower()

        # ordenar por comprimento (multi-word first)
        keys = sorted(organ_map.keys(), key=len, reverse=True)

        for kw in keys:
            pattern = r"\b" + re.escape(kw) + r"\b"
            if re.search(pattern, text_low):
                return organ_map[kw]

        return None


    def postprocess_biodistribution(self, entries: list[dict]) -> dict:
        processed = []

        for item in entries:
            raw_val = str(item.get("value", "")).replace("%", "").strip()

            try:
                percent = float(raw_val)
            except:
                percent = None

            cond = item.get("condition") or ""
            
            organ = self.detect_organ(cond, ORGAN_MAP) or "unknown"
            processed.append({
                "organ": organ,
                "percent": percent,
                "unit": item.get("unit"),
            })
        processed = sorted(
            processed,
            key=lambda x: (x["percent"] is not None, x["percent"]),
            reverse=True
        )

        final = {}
        for i, row in enumerate(processed, start=1):
            final[f"of_target_{i}"] = row

        return final

    def extract_cargos(self, text: str) -> list[str]:
        """Return all cargos, normalizing known ones and keeping unknown ones."""

        # Remove text inside parentheses
        text = re.sub(r"\([^)]*\)", "", text)

        # Split on common separators
        parts = re.split(r"\s*(?:,|;|/|\+|\band\b|\bor\b)\s*", text)

        cargos = []

        for part in parts:
            part = part.strip()
            if not part:
                continue

            matched = False

            for pat, name in COMPILED_MAP:
                if pat.fullmatch(part) or pat.search(part):
                    cargos.append(name)
                    matched = True
                    break

            if not matched:
                cargos.append(part)

        # Remove duplicates while preserving order
        return list(dict.fromkeys(cargos))


    def postprocess_IC50(self, entries: list[dict]) -> dict:
        processed = []
        for item in entries:

            cond = item.get("formulation") or ""
            #cargo = self.extract_cargos(cond) or "unknown"

            processed.append({
                "value": item.get("value"),
                "unit": item.get("unit"),
                "cargo/formulation": cond,
                "cell_line": item.get("cell_line"),
            })
        return processed
    
    def postprocess_ratio(self, lipids_list: list[str], lipids_ratio_list: list[str], dimension: str) -> dict:
        if lipids_list is not None and lipids_ratio_list is not None:
            if len(lipids_list) != len(lipids_ratio_list):
                lipids_list = None
                lipids_ratio_list = None
            else: 
                    processed = []
                    for lipid, ratio in zip(lipids_list, lipids_ratio_list):
                        processed.append({
                            "lipid": lipid,
                            "ratio": ratio,
                            "distribution": dimension,
                        })
        return processed
    


    def postprocess_dose_group(self, entries: list[dict]) -> list[dict]:
        processed = []

        # unidades a ignorar
        skip_units = {"", "l", "/l", "/ml"}

        for item in entries:
            unit = item.get("unit")
            value = item.get("value")

            if unit == "µg":
                try:
                    value = str(float(value)/ 30)
                    unit = "mg/kg"
                except (TypeError, ValueError):
                    pass
            
            if unit == "g":
                try:
                    value = str(float(value) / 0.03)
                    unit = "mg/kg"
                except (TypeError, ValueError):
                    pass

            if unit == "kg":
                try:
                    value = str(float(value) / 0.000030)
                    unit = "mg/kg"
                except (TypeError, ValueError):
                    pass

            # remover entradas com unidades inválidas/associadas a volume
            normalized_unit = str(unit).strip().lower() if unit is not None else ""
            if normalized_unit in skip_units:
                continue

            processed.append({
                "value": value,
                "unit": unit,
                "drug_name": item.get("drug_name"),
            })

        return processed
        
    def compute_charge_group(self, entries: list[dict]) -> str | None:
        if not entries:
            return None

        try:
            # take first valid numeric value
            for item in entries:
                val = item.get("value")

                if val is None:
                    continue

                zeta = float(str(val).replace(",", ".").split()[0])

                if zeta > 10:
                    return "positive"
                elif zeta < -10:
                    return "negative"
                else:
                    return "neutral"

        except Exception:
            return None

        return None

    def replace(self, T_and_F_list: dict, simulated_llm_response: str) -> dict:
        known_param = next(k for k, v in T_and_F_list.items() if v)
        parsed = self.parse_response(simulated_llm_response, known_param)

        postprocessors = {
            "biodistribution":
                self.postprocess_biodistribution,

            "ic50":
                self.postprocess_IC50,

            "dose_group":
                self.postprocess_dose_group,
            
        }

        for key, fn in postprocessors.items():

            if key in parsed:
                parsed[key] = fn(parsed[key])

        for key, value in parsed.items():

            if key in T_and_F_list:
                T_and_F_list[key] = value

        if (
            "zeta_potential_mv" in T_and_F_list
            and not T_and_F_list.get("charge_group")
        ):

            charge = self.compute_charge_group(
                T_and_F_list["zeta_potential_mv"]
            )

            T_and_F_list["charge_group"] = charge

        return T_and_F_list
    

    def parse_schedule_response(self, response: str) -> dict[str, str]:
        """
        Parses the schedule LLM response into a dict of {drug_name: schedule}.
        Expected format per line:  DrugName | single_dose
        """
        results = {}

        for raw_line in response.strip().splitlines():
            line = raw_line.strip()

            if not line or line.startswith("#"):
                continue

            parts = [p.strip() for p in line.split("|")]

            if len(parts) < 2:
                continue

            drug_name = parts[0]
            schedule  = parts[1] if parts[1] else "unknown"

            results[drug_name] = schedule

        return results


    def update_schedule(self, T_and_F_list: dict, schedule_raw: str) -> dict:
        """
        Parses the raw schedule LLM response and injects 'schedule'
        into each entry of dose_group, matched by drug_name.
        """
        if "dose_group" not in T_and_F_list or not T_and_F_list["dose_group"]:
            return T_and_F_list

        parsed = self.parse_schedule_response(schedule_raw)
        print(f"  [PARSER.update_schedule] Parsed schedule: {parsed}")

        for entry in T_and_F_list["dose_group"]:
            drug_name = entry.get("drug_name")
            entry["schedule"] = parsed.get(drug_name, "unknown")
            print(f"  [PARSER.update_schedule] {drug_name} → {entry['schedule']}")

        return T_and_F_list

    def parse_missing_lipids(self, response: str) -> list[str]:
        response = response.strip()
        if not response or response.lower() in ("none", "n/a", "-"):
            return []

        items = []
        for raw_line in response.splitlines():
            line = raw_line.strip(" -•\t")
            if not line or line.lower() == "none":
                continue
            items.append(line)
        return items

    @staticmethod
    def _norm_key(s: str) -> str:
        """Lowercase + collapse whitespace, for case/space-insensitive comparisons."""
        return " ".join(s.strip().lower().split())

    def update_lipid_composition(
        self,
        T_and_F_list: dict,
        missing_raw: str,
        normalization_map: dict,
        generic_terms: dict,
    ) -> dict:
        if "lipid_composition" not in T_and_F_list:
            return T_and_F_list

        existing = T_and_F_list.get("lipid_composition") or []
        missing_items = self.parse_missing_lipids(missing_raw)
        print(f"  [PARSER.update_lipid_composition] LLM reported missing: {missing_items}")

        # normalized lookup: norm_key -> canonical form already present
        existing_norm = {self._norm_key(e): e for e in existing}

        # normalized lookup for the normalization map (handles e.g. "Mc3" vs "MC3")
        norm_map_lookup = {self._norm_key(k): v for k, v in normalization_map.items()}

        normalized_new = []
        for item in missing_items:
            item_key = self._norm_key(item)

            # resolve to canonical form via normalization_map (case-insensitive)
            canonical = norm_map_lookup.get(item_key, item)
            canonical_key = self._norm_key(canonical)

            # skip if it already exists (case/space-insensitive) in regex result or in this batch
            if canonical_key in existing_norm:
                print(f"  [PARSER.update_lipid_composition] Skipping '{item}' -> already present as '{existing_norm[canonical_key]}'")
                continue

            already_added = any(self._norm_key(n) == canonical_key for n in normalized_new)
            if already_added:
                continue

            normalized_new.append(canonical)

        merged = list(existing) + normalized_new
        # final de-dup pass, case/space-insensitive, keeps first occurrence
        seen = {}
        deduped = []
        for item in merged:
            key = self._norm_key(item)
            if key not in seen:
                seen[key] = True
                deduped.append(item)
        merged = deduped

        # re-apply generic-term suppression (e.g. drop "phosphatidylcholine" if a specific PC is present)
        for generic, specific_set in generic_terms.items():
            generic_key = self._norm_key(generic)
            specific_keys = {self._norm_key(s) for s in specific_set}
            merged_keys = {self._norm_key(m) for m in merged}

            if generic_key in merged_keys and merged_keys & specific_keys:
                merged = [m for m in merged if self._norm_key(m) != generic_key]

        T_and_F_list["lipid_composition"] = merged if merged else None
        return T_and_F_list
    
    def parse_load_status_response(self, response: str) -> dict[tuple, str]:
        results = {}
        for raw_line in response.strip().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 3:
                continue
            value, unit, status = parts[0], parts[1], parts[2]
            key = self._make_size_key(value, unit)
            results[key] = status if status in ("loaded", "unloaded") else None
        return results

    def update_load_status(self, T_and_F_list: dict, load_raw: str) -> list[dict]:
        size_entries = T_and_F_list.get("size_nm")
        if not size_entries:
            return size_entries

        parsed = self.parse_load_status_response(load_raw)

        for entry in size_entries:
            key = self._make_size_key(entry.get("value"), entry.get("unit"))
            entry["load"] = parsed.get(key)  # None if not matched/unknown

        return size_entries

    @staticmethod
    def _make_size_key(value, unit) -> tuple:
        """
        Normalize size values so that:
        100 == 100.0
        >50 == > 50
        >50 nm == >50
        <100 == < 100.0
        >=25 == >= 25
        """
        u = str(unit).strip().lower() if unit else None

        if value is None:
            normalized = None
        else:
            s = str(value).strip()

            # Remove unit if it appears at the end of the value
            if u and s.lower().endswith(u):
                s = s[:-len(u)].strip()

            # match optional operator + number
            m = re.match(r'^\s*(>=|<=|>|<)?\s*(-?\d+(?:\.\d+)?)\s*$', s)
            if m:
                op = m.group(1) or "="
                num = round(float(m.group(2)), 4)
                normalized = (op, num)
            else:
                # fallback for non-numeric expressions
                normalized = re.sub(r'\s+', '', s).lower()

        return (normalized, u)
    
    def parse_lipid_ratio_units_response(self, response: str) -> str | None:
        for raw_line in response.strip().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 2:
                continue
            param_name, value = parts[0], parts[1]
            if param_name != "lipid_composition_ratio_units":
                continue
            if value.lower() in ("not_extractable", "not extractable", "", "-", "none"):
                return None
            return value

        return None
    
    def parse_formulation_registry(self, response: str) -> dict[str, dict]:
        registry = {}

        for raw_line in response.strip().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 3:
                continue

            code, drug, load = parts[0], parts[1], parts[2]

            if code == "<CODE>":
                continue

            drug_name = None if drug.lower() in ("none", "", "-") else drug
            if drug_name is None:
                continue  # Skip formulations without a drug name

            drug_names = self.extract_cargos(drug_name)

            if not drug_names:
                drug_names=[drug_name]

            registry[code.upper()] = {
                "drug_name": drug_names,
                "load": load if load in ("loaded", "unloaded") else None,
            }

        return registry
    
    def parse_cargo_category_check(self, response: str) -> dict[str, dict]:

        results = {}
        for raw_line in response.strip().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 3:
                continue
            name, is_drug, category = parts[0], parts[1], parts[2]
            results[name] = {
                "is_drug": is_drug.lower() == "yes",
                "category": category,
            }
        return results
    

    def parse_lipid_ratio_response(self, response: str, lipids: list[str]) -> dict | None:
        """
        Parses the lipid ratio LLM response.
        Returns {"ratios": [float|None, ...]} aligned by position with lipids list,
        or None if not extractable at all.
        """
        
        # check for global not_extractable
        lines = [l.strip() for l in response.strip().splitlines() if l.strip() and not l.startswith("#")]
        if len(lines) == 1 and "not_extractable" in lines[0].lower() and "|" in lines[0]:
            parts = [p.strip() for p in lines[0].split("|")]
            if len(parts) == 2:  # lipid_composition_ratio | not_extractable
                return None

        # parse per-lipid lines
        ratio_map = {}
        for raw_line in lines:
            parts = [p.strip() for p in raw_line.split("|")]
            if len(parts) < 3:
                continue
            _, lipid_name, ratio_value = parts[0], parts[1], parts[2]
            if ratio_value.lower() in ("not_extractable", "not extractable", "", "-", "none"):
                ratio_map[lipid_name.strip().lower()] = None
            else:
                try:
                    ratio_map[lipid_name.strip().lower()] = float(ratio_value)
                except ValueError:
                    ratio_map[lipid_name.strip().lower()] = None

        # align by position with lipids list
        ratios = [ratio_map.get(l.strip().lower()) for l in lipids]

        # if all None, treat as not extractable
        if all(r is None for r in ratios):
            return None
        return {"ratios": ratios}



class ImageParserKeys(BaseModel):
    """Parses the raw VLM text output from PromptCreationImageKeys
    into structured axis labels, ticks, and series names."""

    _data: Dict[str, Any] = PrivateAttr(default_factory=dict)

    def parse(self, raw_output: str) -> None:
        self._data = {
            "x_axis": self._extract_field(raw_output, "X_AXIS"),
            "x_ticks": self._extract_list(raw_output, "X_TICKS"),
            "y_axis": self._extract_field(raw_output, "Y_AXIS"),
            "y_ticks": self._extract_list(raw_output, "Y_TICKS"),
            "series": self._extract_series(raw_output),
        }

    def get_data_dict(self) -> Dict[str, Any]:
        return self._data

    @staticmethod
    def _extract_field(text: str, key: str) -> Optional[str]:
        match = re.search(rf"{key}:\s*(.+)", text)
        return match.group(1).strip() if match else None


    @staticmethod
    def _extract_series(text: str) -> List[Dict[str, str]]:
        match = re.search(r"SERIES:\s*(.+)", text, re.DOTALL)
        if not match:
            return []

        series = []

        for line in match.group(1).splitlines():
            line = line.strip()

            if not line.startswith("-"):
                if line == "" and series:
                    break
                continue

            line = line.lstrip("-").strip()

            parsed = {}
            for p in line.split("|"):
                if ":" in p:
                    k, v = p.split(":", 1)
                    parsed[k.strip().lower()] = v.strip()

            series.append(parsed)

        return series
        
    @staticmethod
    def _extract_list(text: str, key: str) -> List[str]:
        match = re.search(rf"{key}:\s*\[(.*?)\]", text, re.DOTALL)
        if not match:
            return []

        content = match.group(1)
        content = re.sub(r"(?<=\S),(?=\S)", "", content)

        return [
            item.strip().strip("\"'")
            for item in content.split(",")
            if item.strip()
        ]

    @staticmethod
    def _extract_points(text: str) -> List[Dict[str, Any]]:
        # Only look inside FINAL_ANSWER section, if present
        final_match = re.search(r"FINAL_ANSWER:\s*(.*)", text, re.DOTALL)
        section = final_match.group(1) if final_match else text

        points = []
        for match in re.finditer(r"\(\s*([^,]+?)\s*,\s*([^)]+?)\s*\)", section):
            x_raw, y_raw = match.group(1).strip(), match.group(2).strip()
            points.append({"x": x_raw, "y": y_raw})
        return points

class SeriesPointsParser(BaseModel):

    @staticmethod
    def parse_points(raw_output: str) -> List[List[Any]]:
        points = []
        for match in re.finditer(r"\(\s*([^,()]+?)\s*,\s*([^,()]+?)\s*\)", raw_output):
            x_raw, y_raw = match.group(1).strip(), match.group(2).strip()
            points.append([
                SeriesPointsParser._to_float(x_raw),
                SeriesPointsParser._to_float(y_raw),
            ])
        return points

    @staticmethod
    def _to_float(value: str) -> Any:
        if value.upper() in {"N/A", "NA", "UNKNOWN"}:
            return None
        try:
            return float(value)
        except ValueError:
            return value 
