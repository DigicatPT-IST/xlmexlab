from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel


class NanoparticleData(BaseModel):
    """
    Structured data model for one nanoparticle study extracted from a paper.
    Each field maps directly to a column in your parameters table.

    Extraction method legend (see extractor for implementation):
        regex     = simple regex pattern near a keyword
        keyword   = KeywordSearching vocabulary lookup
        parser    = ParametersParser (number + unit extraction)
        ratio     = MolarRatioFinder (lipid composition ratios)
        llm       = small LLM call (only 2-3 features need this)
        derived   = computed from other fields, no extraction needed
    """

    # ── Material properties ──────────────────────────────────────────────────

    # [keyword] organic/inorganic based on particle name keywords
    type: Optional[str] = None

    # [keyword] liposome / SLN / polymeric / dendrimer / micelle / niosome
    subtype: Optional[str] = None

    # [parser] ParametersParser extracts "~150 nm" or "100-200 nm" automatically
    size_nm: Optional[bool] = None

    # [regex] number near "zeta potential" + mV unit
    zeta_potential_mv: Optional[bool] = None

    # [keyword] positive / negative / neutral  (from cationic/anionic/neutral/zwitterionic)
    charge_group: Optional[str] = None

    # [keyword] sphere / rod
    shape: Optional[str] = None

    # [keyword] Unilamellar (SUV/LUV) / Multilamellar (MLV)
    lamellarity: Optional[str] = None

    # [regex] number near "PDI" or "polydispersity index"
    pdi: Optional[bool] = None

    # ── Surface engineering & functionalization ──────────────────────────────

    # [keyword] monolayer / multilayer
    coating_number: Optional[str] = None

    # [llm] low / medium / high  — hard to infer without context
    total_coating_steps_group: Optional[str] = None

    # [keyword] all lipid names found: DPPC, DSPC, DOPC, DOPE, Cholesterol, DSPE-PEG, etc.
    lipid_composition: Optional[List[str]] = None

    # [ratio] MolarRatioFinder result e.g. {"DPPC": "55", "Chol": "40", "DSPE-PEG": "5"}
    lipid_composition_ratio: Optional[Dict[str, Any]] = None


    # [keyword] None / pH-sensitive / Thermosensitive / Redox-sensitive / etc.
    stimulus_responsive: Optional[str] = None

    # [keyword] None / covalent
    bioconjugation_nature: Optional[bool] = None

    # [keyword] PEGylated / non-PEGylated  (from "PEGylated","DSPE-PEG","PEG-modified")
    peg_coat: Optional[str] = None

    # ── Targeting ────────────────────────────────────────────────────────────

    # [keyword] passive / active
    targeting_type: Optional[str] = None

    # ── Therapeutic payload ──────────────────────────────────────────────────

    # [llm] Anthracycline / Taxane / Nucleic acid / etc.
    # LLM maps drug name (doxorubicin, paclitaxel...) to its pharmacological class
    therapeutic_molecule_type: Optional[str] = None

    # [keyword] Passive entrapment / Active loading (pH gradient) / Bilayer intercalation / Surface conj.
    drug_loading_method: Optional[str] = None

    # ── Dosing & administration ──────────────────────────────────────────────

    # [parser] ParametersParser with mg/kg in quantity units config
    dose_group: Optional[bool] = None

    # [keyword] Multiple-dose / Single Dose
    no_days_dosing_grouped: Optional[str] = None

    # [keyword] Systemic / Local
    route: Optional[list[str]] = None

    # [keyword] IV / Intratumoral
    route_subtype: Optional[list[str]] = None

    # ── Therapeutic strategy ─────────────────────────────────────────────────

    # [keyword] first therapy found: Chemotherapy / Gene therapy / PDT / PTT / etc.
    therapy_a: Optional[str] = None

    # [keyword] second therapy found (if combination)
    therapy_b: Optional[str] = None

    # [keyword] third therapy found (if triple combination)
    therapy_c: Optional[str] = None

    # [derived] Monotherapy if only therapy_a, else Combination therapy — no extraction needed
    combined_therapy_grouped: Optional[str] = None

    # [keyword] Diagnosis / Theranostics / Therapy
    study_strategy: Optional[str] = None

    # ── Biological context ───────────────────────────────────────────────────

    # [keyword] Xenograft heterotopic / Xenograft orthotopic / Allograft orthotopic / Allograft heterotopic
    tumor_model: Optional[str] = None

    # [keyword] Immunocompetent / Immunocompromised  (from mouse strain / "nude" / "athymic")
    immune_status: Optional[str] = None

    # [keyword] Breast / Lung / Liver / Brain / etc.  (from cancer type words + cell line names)
    cancer_type: Optional[str] = None

    # [keyword] Triple-Negative / HER2-enriched / Luminal A / etc.  (from TNBC / cell line names)
    breast_cancer_subtype: Optional[str] = None

    # ── Imaging & diagnostics ────────────────────────────────────────────────

    # [keyword] first imaging modality found
    imaging_a: Optional[str] = None

    # [keyword] second imaging modality found
    imaging_b: Optional[str] = None

    # [keyword] third imaging modality found
    imaging_c: Optional[str] = None

    # ── Outcomes: delivery & pharmacokinetics ────────────────────────────────

    # [regex] True if "IC50" or "IC₅₀" found in text, else false
    ic50: Optional[bool] = None

    # [regex] number near "encapsulation efficiency" or "EE%"
    encapsulation_efficiency_pct: Optional[bool] = None

    # [keyword] 1 if delivery efficiency / cellular uptake reported, else 0
    delivery_efficiency: Optional[bool] = None

    # [regex] number near "t½α" or "distribution half-life"
    distribution_half_life_h: Optional[bool] = None

    # [regex] number near "t½β", "blood circulation half-life", "circulation time"
    circulation_half_life_h: Optional[bool] = None

    # ── Biodistribution / off-target effects ─────────────────────────────────

    biodistribution: Optional[bool] = None
    cargos: Optional[List] = None
    categories_cargos:Optional[List] = None
    
    formulations: Optional[List]= None

    # ── Outcome ──────────────────────────────────────────────────────────────

    # [regex] number near "tumor volume reduction" or "tumor growth inhibition" + %
    tumor_reduction: Optional[bool] = None

    tumor_size_or_volume: Optional[bool] = None

