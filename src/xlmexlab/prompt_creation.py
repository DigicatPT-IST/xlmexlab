from typing import Any, List
from pydantic import BaseModel


DEFAULT_FIELDS = ["parameter_name", "value", "unit", "condition"]


# Global two-step screening applied to ALL parameters
GLOBAL_SCREENING = (
    "For each numerical candidate found in the text, apply these two steps:\n"
    "  STEP 1 — Is it introduced by 'X et al.', 'reported', 'showed', 'found', "
    "'according to', or a citation [N]? → DISCARD\n"
    "  STEP 2 — Is it from the authors' own experiment?  → KEEP\n"
    "Only KEPT values are extracted."
)

GLOBAL_NULL_RULE = (
    "- If a parameter is not mentioned, has no numerical value, "
    "or all values were DISCARDED by the screening steps:\n"
    "  <parameter_name> | not_extractable"
)


PARAM_META: dict[str, dict] = {
    "size_nm": {
        "description": "Nanoparticle diameter or size explicitly measured in the study",
        "unit_hint": "nm",
        "specific_format": "size_nm | <value> | <unit> | <formulation> | <size_type>",
        "field_rules": {
            "<value>": "Numeric size exactly as reported (include ranges , intervals and deviations).",
            "<unit>": "Unit exactly as written in text.",
            "<formulation>": (
                "Identify the formulation associated to the reported size."
                "If no formulation name is specified, write 'not extractable'."
            ),
            "<size_type>": (
                "Type of particle size measurement reported. "
                "Indicate the method used to determine size, such as 'DLS', "
                "'hydrodynamic diameter', 'TEM', 'z-average' "
                "or other explicitly stated measurement type. "
                "Use only values explicitly mentioned in the text; if not specified, write 'not extractable."
            )
    },
        "exclude": [
            "theoretical sizes",
            "expected sizes",
            "pore sizes",
            "filter sizes",
            "instrument limits",
        ],
    },

    "bioconjugation_nature": {
        "description": "Bioconjugation nature",
        "unit_hint": "-",
        "specific_format": "bioconjugation_nature | <nature> | <formulation>",
        "field_rules": {
            "<nature>": "If its refered if the bioconjagation nature is elestrostatic or covalent. ",
            "<forumulation>": (
                "Identify the formulation associated to the reported nature."
                "If no formulation name is specified, write 'not extractable'."
            ),
    },
        "exclude": [
            "theoretical sizes",
            "expected sizes",
            "pore sizes",
            "filter sizes",
            "instrument limits",
        ],
    },

    "zeta_potential_mv": {
        "description": "Zeta potential (surface charge)",
        "unit_hint": "mV",
        "specific_format": "zeta_potential_mv | <value> | <unit>",
        "field_rules": {
            "<value>": "Numeric zeta potential exactly as reported.",
            "<unit>": "Unit exactly as written.",
        },
        "exclude": [
            "predicted charge",
            "theoretical surface charge",
        ],
    },

    "pdi": {
        "description": "Polydispersity index",
        "unit_hint": "dimensionless",
        "specific_format": "pdi | <value>| <methodology>",
        "field_rules": {
            "<value>": "Numeric PDI exactly as reported.",
            "<methodology>": "Preparation, formulation, or measurement method asssociated to the reported PDI",
        },
        "exclude": [],
    },

    "encapsulation_efficiency_pct": {
        "description": "Encapsulation efficiency or EE",
        "unit_hint": "%",
        "specific_format": "encapsulation_efficiency_pct | <value> | <unit> | <drug_name>",
        "field_rules": {
            "<value>": "ONLY the numeric EE (Encapsulation Efficiency) exactly as reported (include the standard deviation if available).",
            "<unit>": "Percentage or unit as written.",
            "<drug_name>": "Drug or formulation if explicitly mentioned.",
        },
        "exclude": [
            "theoretical efficiency",
        ],
    },

    "ic50": {
        "description": "Half-maximal inhibitory concentration",
        "unit_hint": "uM, nM, mg/mL, etc.",
        "specific_format": "ic50 | <value> | <unit> | <formulation> | <cell_line>",
        "field_rules": {
            "<value>": "Numeric IC50 exactly as reported.",
            "<unit>": "Unit exactly as written.",
            "<formulation>": " Formulation code/ drug name only.",
            "<cell_line>": "Cell line used in the experiment if not stated, write 'not extractable'.",
        },
        "exclude": [],
    },

    "distribution_half_life_h": {
        "description": "Distribution half-life (alpha phase)",
        "unit_hint": "h",
        "specific_format": "distribution_half_life_h | <value> | <unit> | <drug_name>",
        "field_rules": {
            "<value>": "Numeric half-life exactly as reported.",
            "<unit>": "Unit in hours or as stated.",
            "<drug_name>": "Formulation code associated to the nanoparticle.",
        },
        "exclude": [],
    },

    "circulation_half_life_h": {
        "description": "Circulation / elimination half-life",
        "unit_hint": "h",
        "specific_format": "circulation_half_life_h | <value> | <unit> | <drug_name>",
        "field_rules": {
            "<value>": "Numeric half-life exactly as reported.",
            "<unit>": "Unit as written.",
            "<drug_name>": "Drug studied.",
        },
        "exclude": [],
    },

    "dose_group": {
        "description": "Administered treatment doses from the AUTHORS' OWN experiment only",
        "unit_hint": "as reported",
        "specific_format": "dose_group | <value> | <unit> | <formulation>",
        "field_rules": {
            "<value>": "Numeric only, exactly as reported. Do NOT include units or route.",
            "<unit>": "Dose unit exactly as written. Valid formats: mg/kg, μg/kg, mg/animal, μg, mg, g, etc.",
            "<formulation>": "Formulation actually administered in this study.",
        },
        "exclude": [
            "theoretical doses",
        ],
    },

    "tumor_size_or_volume": {
        "description": "Tumor size or volume",
        "unit_hint": "mm, cm, mm³, mL or as reported",
        "specific_format": "tumor_ size_or_volume | <value> | <unit> | <formulation> | <state> | <comparasion>",
        "field_rules": {
            "<value>": "Numeric reduction exactly as reported.",
            "<unit>": "mm, cm, mm³, mL or as reported.",
            "<formulation>": "Drug/formulation or control, if not stated left it 'unknown'.",
            "<state>": "Time associated to the volume or if its control, if not stated left it 'unknown'.",
            "<comparison>": "One of: 'absolute', 'increase', 'decrease'. Use 'increase' or 'decrease' when the value represents a relative change. Use 'absolute' when the reported value is the measured tumor size or volume.",
        },
        "exclude": [
            "predicted inhibition",
        ],
    },

    "tumor_reduction": {
        "description": "Tumor size reduction/ regression",
        "unit_hint": "%",
        "specific_format": "tumor_reduction | <value> | <unit> | <formulation>",
        "field_rules": {
            "<value>": "Numeric reduction exactly as reported.",
            "<unit>": "Percentage.",
            "<formulation>": "Drug or formulation associated to the reduction if explicitly written, if not just left it 'unknown'.",
        },
        "exclude": [
            "predicted inhibition",
        ],
    },

    "delivery_efficiency": {
        "description": "Cellular or in-vivo delivery efficiency",
        "unit_hint": "% or fold-change",
        "specific_format": "delivery_efficiency | <value> | <unit> | <condition>",
        "field_rules": {
            "<value>": "Numeric efficiency exactly as reported.",
            "<unit>": "Unit or fold-change exactly as written.",
            "<condition>": "Experimental condition if present.",
        },
        "exclude": [
            "qualitative statements",
        ],
    },

    "biodistribution": {
        "description": "Organ accumulation/ biodistribution",
        "unit_hint": "%ID, %ID/g, etc.",
        "specific_format": "biodistribution | <value> | <unit> | <organ>",
        "field_rules": {
            "<value>": "Numeric accumulation exactly as reported.",
            "<unit>": "Unit exactly as written.",
            "<organ>": "Organ explicitly mentioned.",
        },
        "exclude": [
            "qualitative targeting",
        ],
    },
}


class PromptCreation(BaseModel):

    def build_extraction_prompt_json(
        self,
        paragraph: str,
        extracted_flags: dict[str, Any],
    ) -> tuple[dict, list[str]]:

        targets = [k for k, v in extracted_flags.items() if v is True]

        if not targets:
            return {}, []

        #  EXPERTISE (system role) 
        expertise = (
            "You are a nanoparticle information-extraction assistant."
            #"You extract data truthfully from scientific text. "
            #"Only extract values explicitly stated as part of the AUTHORS' OWN experiment. "
            #"Do not infer, guess, or hallucinate values."
        )

        # INITIALIZATION 
        initialization = (
            "Only extract values explicitly stated as part of the AUTHORS' OWN experiment."
            "Ignore values mentioned from other studies, literature comparisons, hypotheses, or discussions."
        )

        #  DEFINITIONS 
        definitions: dict[str, str] = {}
        for param in targets:
            meta = PARAM_META[param]
            definitions[param] = (
                f"{meta.get('description', param.replace('_', ' '))}. "
                f"Expected unit: {meta.get('unit_hint', 'as reported')}."
            )

        # OBJECTIVE 
        objective = "Extract ONLY the parameters listed above from the paragraph below."

        # Answer schema construction
        schema_lines: list[str] = []

        # Format per parameter
        override_lines = [
            f"{PARAM_META[param]['specific_format']}"
            for param in targets
            if "specific_format" in PARAM_META.get(param, {})
        ]
        if override_lines:
            schema_lines.append("Format answer:")
            schema_lines.extend(override_lines)

        schema_lines.append("Rules:")
        schema_lines.append("- One line per extracted value, no NOT omit any field of the 'Format answer' given, ALWAYS include the parameter name.")
        schema_lines.append(
            "- For each numerical candidate found in the text, apply these two steps:\n"
            "  STEP 1 \u2014 Is it introduced by 'X et al.', 'reported', 'showed', 'found', "
            "'according to', or a citation [N]? \u2192 DISCARD\n"
            "  STEP 2 \u2014 Is it from the authors' own experiment? \u2192 KEEP\n"
            "  Only KEPT values are extracted."
        )

        # Per-parameter field rules and exclusions (format/parsing only)
        for param in targets:
            meta = PARAM_META[param]
            if "field_rules" in meta:
                schema_lines.append(f"- Field rules for {param}:")
                for field, rule in meta["field_rules"].items():
                    schema_lines.append(f"  - {field}: {rule}")
            if "exclude" in meta:
                schema_lines.append(f"- Do NOT extract for {param}:")
                for ex in meta["exclude"]:
                    schema_lines.append(f"  - {ex}")

        # Global null rule
        schema_lines.append(GLOBAL_NULL_RULE)

        answer_schema: dict[str, str] = {
            "Format answer (include always all fields)": "\n".join(schema_lines),
        }

        # Conclusions
        conclusion = "Return ONLY the extraction lines. No explanations, headers, or comments."

        prompt_json = {
            "expertise": expertise,
            "initialization": initialization,
            "definitions": definitions,
            "objective": objective,
            "answer_schema": answer_schema,
            "conclusion": conclusion,
        }

        return prompt_json, targets
    
class PromptCreationSchedule(BaseModel):

    def build_extraction_prompt_json(
        self,
        drug_names: list[str],
    ) -> tuple[dict, list[str]]:
        
        expertise = (
            "You are an expert assistant for extracting dosing schedule information from scientific text."
        )
        # INITIALIZATION 
        initialization = (
            "Given this text, for each drug below, what is the dosing schedule?"
        )

        #  DEFINITIONS 
        definitions: dict[str, str] = {}

        # OBJECTIVE 
        objective = "Only answer with: single_dose, multi_dose, or unknown."

        # Answer schema construction
        schema_lines: list[str] = []

        schema_lines.append(
            "Rules:"
            "Use ONLY explicit frequency words ('every X days', 'twice', 'q.d.', 'BID', 'once', 'single dose', 'every week', etc).\n"        
            "Do NOT use timing words ('prior to', '15 min before', 'day 0', 'for X days') as evidence.\n"
            "Format: <drug_name> | <schedule>\n"
            "Drug names: " + ", ".join(drug_names)
            )

        answer_schema: dict[str, str] = {
            "Format": "\n".join(schema_lines),
        }

        # Conclusions
        conclusion = "Return ONLY the extraction lines. No explanations, headers, or comments."

        prompt_json = {
            "expertise": expertise,
            "initialization": initialization,
            "definitions": definitions,
            "objective": objective,
            "answer_schema": answer_schema,
            "conclusion": conclusion,
        }

        return prompt_json
    

class PromptCreationLipidComposition(BaseModel):

    def build_extraction_prompt_json(self, lipids_found: list[str]) -> dict:

        expertise = (
            "You are an expert assistant for identifying lipid components in "
            "nanoparticle formulations described in scientific text."
        )

        initialization = (""
        )

        objective = (
            " Find all lipid, lipid derivative, sterol, PEG-lipid, "
            "or ionizable lipid mentioned in the text."

        )


        schema_lines = [
            "Task:",
            "",
            "Phase 1 – Extraction",
            "- Extract every lipid, phospholipid, sterol, PEG-lipid, ionizable lipid, or lipid derivative mentioned in the text.",
            "- Do not exclude any lipid during extraction.",
            "- Normalize obvious OCR artifacts.",
            "- Normalize synonymous names when possible.",
            "Paragraph:",
        ]

        conclusions = "\n".join([
            "Phase 2 – Filtering",
            "Already identified:",
            *(lipids_found if lipids_found else ["(none)"]),
            "",
            "- Remove every lipid that appears in the list above.",
            "- Return only the remaining lipid names.",
            "- If none remain, return exactly:",
            "none",
            "Phase 3:"
            "Return ONLY the missing lipid names (or 'none'). No explanations, headers, or comments."
            "Phase 4:"
            "Re-check if all components in the list are lipids, and return one lipid per line.",
        ])

        return {
            "expertise": expertise,
            "initialization": initialization,
            "definitions": {},
            "objective": objective,
            "answer_schema": {"Format": "\n".join(schema_lines)},
            "conclusion": conclusions,
        }
    
class PromptCreationLoadStatus(BaseModel):

    def build_extraction_prompt_json(self, size_entries: list[dict]) -> dict:
        expertise = (
            "You are an expert assistant for determining the loading status of "
            "nanoparticle formulations described in scientific text."
        )
        initialization = (
            "Read the paragraph carefully."
        )
        objective = "Only define each formulation as loaded, unloaded, or unknown using the rules below."

        entries_str = "\n".join(
            f"- size={e.get('value')} {e.get('unit')}, drug={e.get('drug_name') or '(none)'}"
            for e in size_entries
        )

        schema_lines = [
            "Rules:",
            "- Use 'loaded' ONLY if the same sentence (or its immediate clause) explicitly "
            "states the particle is loaded/encapsulated/incorporated with a drug or cargo.",
            "- Use 'unloaded' ONLY if the same sentence (or its immediate clause) explicitly "
            "states the particle is 'blank', 'empty', 'unloaded', or 'control'.",
            "- Do NOT infer loading status from formulation codes "
            "mentioned earlier in the text unless that code or its explicit label is used "
            "in the same sentence as this size value.",
            "- Do NOT assume a value belongs to a different formulation just because a "
            "different preparation method (e.g. a different technique) is mentioned.",
            "- If unsure, answer 'unknown'.",
            "Format: <size_value> | <unit> | <status>",
            "Size values to classify:",
            entries_str,
        ]

        return {
            "expertise": expertise,
            "initialization": initialization,
            "definitions": {},
            "objective": objective,
            "answer_schema": {"Format": "\n".join(schema_lines)},
            "conclusion": "Return ONLY the classification lines. No explanations, headers, or comments.",
        }
    
class PromptCreationLipidRatioUnits(BaseModel):

    def build_extraction_prompt_json(self, ratio_entries: list[dict]) -> dict:
        expertise = (
            "You are an expert assistant for identifying the quantification type "
            "of lipid composition ratios in nanoparticle formulations."
        )

        entries_str = "\n".join(
            f"- {e.get('lipid')}: {e.get('ratio')}"
            for e in ratio_entries
        )

        initialization = (
            "The following lipid:ratio pairs were already identified in the text below:\n"
            + entries_str
        )

        objective = (
            "For the lipids ratio listed above, determine the quantification type "
            "explicitly used in the text (e.g. molar ratio, mol%, weight ratio, w/w, v/v)."
        )

        schema_lines = [
            "Format answer: lipid_composition_ratio_units | <quantification_type>",
            "Rules:",
            "- Use the quantification type EXACTLY as stated in the text near that ratio "
            "(e.g. 'mol%', 'molar ratio', 'w/w', 'weight ratio').",
            "- Do NOT infer or guess a unit; only use what is explicitly written.",
            "- If no quantification type is not explicit just write 'lipid_composition_ratio_units | unknown'."
        ]

        return {
            "expertise": expertise,
            "initialization": initialization,
            "definitions": {},
            "objective": objective,
            "answer_schema": {"Format": "\n".join(schema_lines)},
            "conclusion": "Return ONLY the extraction line using the given format. No explanations, headers, or comments.",
        }
    
class PromptCreationFormulationRegistry(BaseModel):

    def build_extraction_prompt_json(self, candidate_codes: list[str]) -> dict:
        expertise = (
            "You are an expert assistant for identifying what nanoparticle "
            "formulation codes or abbreviations refer to in scientific text."
        )
        initialization = (
            "A first-pass scan found these candidate formulation codes/abbreviations "
            "in the text below:\n" + ", ".join(candidate_codes)
        )
        objective = (
            "For each REAL formulation code (i.e. one that genuinely refers to a "
            "specific nanoparticle/liposome formulation in this text), determine: "
            "(1) the drug or cargo it contains, if any, and "
            "(2) whether it is explicitly described as loaded or unloaded/blank. "
            "Only use information explicitly stated in the text. Do not infer drug "
            "identity or load status from the code's letters/name alone "
            "(e.g. do not assume 'BLK' means blank just because of the abbreviation; "
            "only conclude this if the text itself states it)."
        )
        schema_lines = [
            "Format: <code> | <drug_name_or_none> | <loaded/unloaded/unknown>",
            "Rules:",
            "- If a candidate is NOT actually a formulation identifier (e.g. it's a "
            "method name, unit, or unrelated abbreviation), SKIP it — do not output a line for it.",
            "- Use 'none' for drug_name if no cargo is stated for that code.",
            "- Use 'unknown' for load status only if the text genuinely does not state it.",
            "- One line per valid formulation code."
            "-If no valid formulation codes remain after filtering, answer exactly: 'none' (without quotes).",
        ]
        return {
            "expertise": expertise,
            "initialization": initialization,
            "definitions": {},
            "objective": objective,
            "answer_schema": {"Format": "\n".join(schema_lines)},
            "conclusion": "Return ONLY the extraction lines. Do not add any explanations, headers, or comments.",
        }
    
class PromptCreationCargoCategoryCheck(BaseModel):

    def build_extraction_prompt_json(self, unmatched_cargos: list[str], known_categories: list[str]) -> dict:
        expertise = (
            "You are an expert pharmacology assistant classifying therapeutic "
            "compounds into drug class categories."
        )
        initialization = (
            "The following candidate drug/cargo names were found in the text but "
            "are NOT present in our reference database:\n"
            + ", ".join(unmatched_cargos)
        )
        objective = (
            "For each name, determine if it is a real therapeutic compound, drug, "
            "or biologically active cargo (not a lipid, buffer, or excipient). "
            "If so, assign the closest matching category from the list below, or "
            "'new_category' if none fit well. If it is NOT a real drug/cargo, mark it 'not_a_drug'."
        )
        schema_lines = [
            "Known categories: " + ", ".join(known_categories),
            "Format: <name> | <is_drug: yes/no> | <category_or_new_category_or_not_applicable>",
            "Rules:",
            "- One line per name, in the same order given.",
            "- Do not invent a name not in the input list.",
        ]
        return {
            "expertise": expertise,
            "initialization": initialization,
            "definitions": {},
            "objective": objective,
            "answer_schema": {"Format": "\n".join(schema_lines)},
            "conclusion": "Return ONLY the extraction lines. No explanations, headers, or comments.",
        }


class PromptCreationLipidRatio(BaseModel):

    def build_extraction_prompt_json(self, lipids_found: list[str]) -> dict:
        expertise = (
            "You are an expert assistant for extracting lipid composition ratios "
            "from nanoparticle formulation descriptions in scientific text."
        )

        initialization = (
            "The following lipids were already identified in the text below:\n"
            + ", ".join(lipids_found)
        )

        objective = (
            "For each lipid listed above, extract the numeric ratio or percentage "
            "explicitly stated in the text for that lipid in the formulation composition. "
            "Only extract ratios that are explicitly written — do not calculate or infer."
        )

        schema_lines = [
            "Format: lipid_composition_ratio | <lipid_name> | <ratio_value>",
            "Rules:",
            "- One line per lipid.",
            "- <lipid_name>: exactly as given in the list above.",
            "- <ratio_value>: numeric value only, exactly as written in text "
            "(e.g. 75, 20.5, 5). Do not include units or symbols.",
            "- If no ratio is explicitly stated for a given lipid, write: "
            "lipid_composition_ratio | <lipid_name> | not_extractable",
            "- If NO ratios at all are stated for any lipid in the text, answer exactly: "
            "lipid_composition_ratio | not_extractable",
        ]

        return {
            "expertise": expertise,
            "initialization": initialization,
            "definitions": {},
            "objective": objective,
            "answer_schema": {"Format": "\n".join(schema_lines)},
            "conclusion": "Return ONLY the extraction lines. No explanations, headers, or comments.",
        }

class PromptCreationIsGraphPrompt(BaseModel):

    def build_is_graph_prompt_json(self) -> dict:
        expertise = (
            "You are an expert in identifying scientific graphs and determining whether their primary subject is biodistribution (%ID/g) or tumour volume/size."
        )

        initialization = ""

        objective = (
            "Determine whether the image contains a scientific graph about biodistribution (%ID/g) or tumour volume/size."
        )

        schema = "<YES or NO>"

        rules = [
            "Answer YES only if the image clearly shows a graph about biodistribution (%ID/g) or tumour volume/size."
            "Answer NO if the graph is about any other measurement or scientific outcome.",
            "For biodistribution, look for indicators such as %ID/g, %ID, tissue or organ distribution, tracer accumulation, drug accumulation, or similar measurements.",
            "For tumour volume/size, look for indicators such as tumour volume, tumour size, tumour growth, tumour burden, mm3, cm3, or similar measurements.",
            "Do not infer the subject from the experimental context alone.",
            "Use the axis labels, units, title, legend, and visible graph content to determine the subject.",
            "If the subject cannot be determined confidently, answer NO.",
            "Return exactly one word: YES or NO.",
            "Do not include explanations, punctuation, markdown, or additional text.",
        ]

        return {
            "expertise": expertise,
            "initialization": initialization,
            "definitions": {
    "biodistribution": "A graph showing the distribution or accumulation of a substance, drug, tracer, nanoparticle, or other agent in tissues or organs, typically reported as %ID/g, %ID, or a closely related biodistribution measurement.",
    "tumour_volume_size": "A graph showing tumour volume, tumour size, tumour growth, or changes in tumour dimensions over time or between treatment groups.",
    "target_graph": "A graph whose primary measured outcome is biodistribution (%ID/g) or tumour volume/size."
  },
            "objective": objective,
            "answer_schema": {
                "Format": schema,
                "Rules": "\n".join(f"- {r}" for r in rules),
            },
            "conclusion": (
                "Return ONLY the word YES or NO. "
                "Do not include explanations, markdown, headers, or comments."
            ),
        }


class PromptCreationImageKeys(BaseModel):

    def build_extraction_prompt_json(self) -> dict:
        expertise = (
            "You are an expert assistant for extracting structured information "
            "from scientific graphs."
        )

        initialization = ""

        objective = (
            "Given an image of a scientific graph, extract the x-axis label, "
            "all visible x-axis tick labels, the y-axis label, all visible "
            "y-axis tick labels, and all series names."
        )

        schema = """X_AXIS: <x-axis label>
X_TICKS: [tick1, tick2, ...]

Y_AXIS: <y-axis label>
Y_TICKS: [tick1, tick2, ...]

SERIES:
- name: <series 1 name> | color: <color> | marker: <marker shape, or 'none' for bars> | line: <solid/dashed/dotted/none>
- name: <series 2 name> | color: <color> | marker: <marker shape, or 'none' for bars> | line: <solid/dashed/dotted/none>"""

        rules = [
            "Copy all text exactly as shown.",
            "Preserve capitalization, symbols, and units.",
            "List all visible tick labels in order (left-to-right for x, bottom-to-top for y).",
            "Series names must match the legend or labels exactly.",
            "For each series, identify its visual appearance exactly as shown in the legend: color, marker shape (circle, square, triangle, none, etc.), and line style (solid, dashed, dotted, none).",
            "If a value cannot be read, write UNKNOWN.",
            "Do not infer or invent missing information.",
            "Normalize numbers using the numerical convention shown by the graph.", 
            "If a comma is used as a thousands separator, remove it: 1,100 → 1100, 2,500 → 2500, 10,000 → 10000.", 
            "If a comma is used as a decimal separator, preserve its numerical meaning: 1,5 → 1.5.", 
            "Determine whether a comma represents thousands or decimals from the formatting and progression of the other tick labels on the same axis.", 
            "Do not interpret a thousands separator as a decimal separator.", "Do not interpret a decimal separator as a thousands separator."
        ]

        return {
            "expertise": expertise,
            "initialization": initialization,
            "definitions": {},
            "objective": objective,
            "answer_schema": {
                "Format": schema,
                "Rules": "\n".join(f"- {r}" for r in rules),
            },
            "conclusion": (
                "Return ONLY the requested structure. "
                "Do not include explanations, markdown, headers, or comments."
            ),
        }



class PromptCreationSeriesDataPrompt(BaseModel):

    def build_series_prompt_json(
        self,
        x_axis: str,
        x_ticks: List[str],
        y_axis: str,
        y_ticks: List[str],
        series_name: str,
        series_color: str,
        series_marker: str,
        series_line: str
    ) -> dict:
        expertise = (
            "You are an expert in precisely locating data points on scientific graphs using their axis tick marks as reference points."
        )

        initialization = ""

        objective = objective = (
            f'Extract all visible data points belonging ONLY to the series "{series_name}". '
            f'This series is visually identified in the legend as: color={series_color}, '
            f'marker={series_marker}, line style={series_line}. '
            f'Use this visual identification focus ONLY on this serie.'
            f'The x-axis is "{x_axis}" with visible ticks {x_ticks}. '
            f'The y-axis is "{y_axis}" with visible ticks {y_ticks}.'
        )

        schema = """SERIES: <series name>
                POINTS:
                - (x1, y1)
                - (x2, y2)
                - (x3, y3)"""

        rules = [
            "1. Start by focusing only on the requested series.",
            "2. Read points from left to right along the x-axis.",
            "3. Determine each x-value using the nearest visible x-axis ticks.",
            "4. Determine each y-value using the nearest visible y-axis ticks.",
            "5. When the point lies between ticks, estimate values with maximum PRECISION.",
            "6. Do not infer points that are not visibly present.",
            "7. If a coordinate cannot be determined confidently, use N/A for that coordinate.",
            "8. Ensure every x-value has exactly one corresponding y-value.",
            "9. Return points ONLY in the format shown: one '(x, y)' pair per line, prefixed with '-'.",
        ]

        return {
            "expertise": expertise,
            "initialization": initialization,
            "definitions": {
                "data_point": "A pair consisting of one x-value and its corresponding y-value.",
                "unknown": "Use N/A when a coordinate cannot be determined confidently.",
            },
            "objective": objective,
            "answer_schema": {
                "Format": schema,
                "Rules": "\n".join(f"- {r}" for r in rules),
            },
            "conclusion": (
                "Return ONLY the requested structure. "
                "Do not include reasoning, explanations, markdown, or extra text outside the POINTS list."
            ),
            }


class PromptCreationVerifySeriesPrompt(BaseModel):

    def build_verify_prompt_json(
        self,
        series_name: str,
        series_color: str,
        series_marker: str,
        series_line: str,
        extracted_points: List[Any],
    ) -> dict:
        expertise = (
            "You are a scientific graph data verifier. "
            "The image is the source of truth."
        )

        objective = (
            f'Verify series "{series_name}" '
            f'(color={series_color}, marker={series_marker}, line={series_line}). '
            f"Previous extraction: {extracted_points}. "
            "Independently inspect the image and return the complete corrected point list."
        )

        schema = """POINT_COUNT_CHECK: <matches / does not match — brief note>

        POINTS:
        - (x, y)
        - (x, y)"""

        rules = [
            "Count the series markers directly from the image.",
            "Verify every previous point against its marker and the axis ticks.",
            "Correct inaccurate coordinates.",
            "Add visible missing markers.",
            "Remove points that do not belong to this series.",
            "Do not infer points that are not visibly supported by the graph.",
            "Use N/A when a coordinate cannot be read reliably.",
            "Return only the requested schema; no reasoning or extra text.",
        ]

        return {
            "expertise": expertise,
            "initialization": "",
            "definitions": {},
            "objective": objective,
            "answer_schema": {
                "Format": schema,
                "Rules": "\n".join(f"- {r}" for r in rules),
            },
            "conclusion": (
                "Return ONLY the requested structure. "
                "Do not include explanations, markdown, headers, or extra comments."
            ),
        }