"""
NanoparticleExtractor
=====================
Hybrid extraction pipeline for nanoparticle study parameters.

Extraction strategy per feature type:
  - ~42 features  → regex / KeywordSearching / ParametersParser / MolarRatioFinder  (0 LLM calls)
  -  1 feature    → paragraph_classifier  (select relevant paper sections first)
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from dataclasses import is_dataclass, asdict
import json

from pydantic import BaseModel, PrivateAttr

from xlmexlab.nanoparticle_data import NanoparticleData

from collections import OrderedDict

from pathlib import Path
import pandas as pd



# 1. CANONICAL DATABASE
CARGO_DB = OrderedDict({
    "alcohol_antagonist": [
        "Antabuse",
        "DS",
    ],

    "alkaloid": [
        "Vinblastine",
        "Vincristine",
        "Vinorelbine",
    ],

    "alkylating_agent": [
        "BCNU",
        "Busulfan",
        "Carmustine",
        "Cyclophosphamide",
        "Melphalan",
        "Temozolomide",
        "TMZ",
    ],

    "anthracycline": [
        "Adriamycin",
        "daunorubicin",
        "DOX",
        "Doxil",
        "epirubicin",
        "Hydroxydaunorubicin",
        "idarubicin",
    ],

    "anti_malarial": [
        "Chloroquine",
        "Hydroxychloroquine",
    ],

    "anti_tumourigenic_cytokine": [
        "IFN-γ",
        "IL-2",
    ],

    "anti_tumourigenic_cytokine_nucleic_acid": [
        "IL-12 mRNA",
        "IL-23 mRNA",
    ],

    "antibody": [
        "anti-PD-L1",
        "Atezolizumab",
        "Avelumab",
        "Bavencio",
        "Durvalumab",
        "Herceptin",
        "Imfinzi",
        "Ipilimumab",
        "Keytruda",
        "Nivolumab",
        "Opdivo",
        "Pembrolizumab",
        "Tecentriq",
        "Yervoy",
    ],

    "antibody_anthracycline": [
        "sacituzumab govitecan",
        "Trodelvy",
    ],

    "antifungal": [
        "Amphotericin B",
    ],

    "antigen": [
        "neoantigen mRNA",
        "OVA",
        "OVA peptide",
        "Tumor lysate",
    ],

    "antigen_nucleic_acid": [
        "HER2 antigen mRNA",
    ],

    "antihelmintic_ribosome_inactivating": [
        "Mebendazole",
    ],

    "antimetabolite": [
        "5-Fluorouracil",
        "5-FU",
        "Cytarabine",
        "GEM",
        "MTX",
        "Pemetrexed",
        "Gemcitabine",
    ],

    "antiviral_nucleic_acid": [
        "Acyclovir",
    ],

    "beta_blocker": [
        "Propranolol",
    ],

    "biologic": [
        "DM1",
        "Filgrastim",
        "G-CSF",
        "Granocyte",
        "Lenograstim",
        "Neulasta",
        "Neupogen",
        "Pegfilgrastim",
    ],

    "biphosphonates": [
        "Fosamax",
        "Man-LP@ZOL",
        "Man-NP@ZOL",
        "Mannosylated Liposome ZOL",
        "ZOL",
        "Zoledronate",
        "Zometa",
    ],

    "boron_neutron_capture_therapy": [
        "BPA",
        "BSH",
    ],

    "cdk4_6_inhibitor_autophagy_inhibitor": [
        "Abemaciclib",
        "Ibrance",
        "Kisqali",
        "Palbociclib",
        "Ribociclib",
        "Verzenio",
    ],

    "enzyme": [
        "DNase",
        "glucose oxidase",
        "L-asparaginase",
    ],

    "enzyme_inhibitor": [
        "Lynparza",
        "Talzenna",
    ],

    "estrogen_receptor_modulator": [
        "Fulvestrant",
        "Tamoxifen",
    ],

    "gnrh_agonist": [
        "Goserelin",
        "Leuprolide",
        "Lupron",
        "Zoladex",
    ],

    "hypoxic_cytotoxin": [
        "AQ4N",
        "Tirapazamine",
    ],

    "immunoadjuvant": [
        "cGAMP",
        "CpG ODN",
        "DMXAA",
        "imiquimod",
        "poly I:C",
        "R837",
        "R848",
        "Resiquimod",
        "STING agonist",
        "TLR9 agonist",
        "vadimezan",
    ],

    "kinase_inhibitor": [
        "Acalabrutinib",
        "Afatinib",
        "Alecensa",
        "Alectinib",
        "Aliqopa",
        "Alpelisib",
        "Alunbrig",
        "Bicalutamide",
        "Brigatinib",
        "Brukinsa",
        "Calquence",
        "Casodex",
        "Ceritinib",
        "Copanlisib",
        "Copiktra",
        "Crizotinib",
        "Duvelisib",
        "Erlotinib",
        "Gilotrif",
        "Gleevec",
        "Ibrutinib",
        "Imatinib",
        "Imbruvica",
        "Lapatinib",
        "Nexavar",
        "Osimertinib",
        "Piqray",
        "Rap",
        "Sirolimus",
        "Sorafenib",
        "Sunitinib",
        "Sutent",
        "Tagrisso",
        "Tarceva",
        "Tykerb",
        "Xalkori",
        "Zanubrutinib",
        "Zykadia",
    ],

    "ligand": [
        "C-peptide-SLN-PTX",
        "iRGD",
        "MMP-responsive peptide",
        "RGD peptide",
    ],

    "metal_compound": [
        "Gd-DTPA",
    ],

    "natural_product": [
        "BER",
        "Berberine",
        "clerodol",
        "Curcumin",
        "diferuloylmethane",
        "EGCG",
        "epigallocatechin gallate",
        "fagarasterol",
        "fagarsterol",
        "farganasterol",
        "Ginsenoside",
        "lupenol",
        "Lupeol",
        "monogynol B",
        "Quercetin",
        "Resveratrol",
        "tsl-lup",
        "Betulin"
    ],

    "nitroxide_radical": [
        "4-amino-TEMPO",
        "TEMPO",
    ],

    "non_steroidal_anti_inflammatory": [
        "Aspirin",
        "Celecoxib",
        "Indomethacin",
    ],

    "nucleic_acid": [
        "AKT siRNA",
        "anti-EGFR siRNA",
        "base editor BRCA1",
        "BCL-2 siRNA",
        "Cas12a mRNA",
        "Cas9 + ESR1 sgRNA",
        "Cas9 + HER2 sgRNA",
        "Cas9 + PIK3CA sgRNA",
        "Cas9 mRNA",
        "circRNA",
        "crRNA",
        "EGFR siRNA",
        "EZH2 siRNA",
        "generic siRNA",
        "gp100 mRNA",
        "HER2 saRNA",
        "hTERT circRNA",
        "let-7",
        "miR-10b",
        "miR-145",
        "miR-155",
        "miR-182-3p",
        "miR-200",
        "miR-21",
        "miR-34a",
        "miR-373",
        "miRNA",
        "MMP-9 siRNA",
        "mRNA",
        "mRNA-4157",
        "mRNA-4359",
        "MUC1 mRNA",
        "MUC1 saRNA",
        "NY-ESO-1 saRNA",
        "OX40L mRNA",
        "p53 mRNA",
        "PI3K siRNA",
        "saRNA",
        "sgRNA",
        "shRNA",
        "siRNA",
        "STAT3 siRNA",
        "survivin siRNA",
        "TGF-β trap mRNA",
        "TRP2 mRNA",
        "Twist1 siRNA",
        "VEGF siRNA",
        "WT1 antigen mRNA",
    ],

    "peripheral_vasostimulant": [
        "Nitroglycerin",
        "Sildenafil",
    ],

    "photosensitizer": [
        "BPD",
        "Ce6",
        "ICG",
        "Phthalocyanine",
        "Porphyrin",
        "Verteporfin",
    ],

    "photosensitizer_enzyme": [
        "peroxidase",
    ],

    "platinum_prodrug": [
        "Carboplatin",
        "Cisplatin",
        "Nedaplatin",
        "Oxaliplatin",
    ],

    "proteasome_inhibitor": [
        "Bortezomib",
        "Carfilzomib",
        "Kyprolis",
        "Velcade",
    ],

    "proteinogenic_amino_acid": [
        "L-arginine",
        "L-glutamine",
    ],

    "purine_analog": [
        "Cladribine",
        "Clofarabine",
        "Fludarabine",
    ],

    "radioactive_element": [
        "131I",
        "177Lu",
        "64Cu",
        "89Zr",
        "90Y",
    ],

    "reducing_and_complexing_thiol": [
        "DTT",
        "Glutathione",
        "GSH",
        "N-acetylcysteine",
        "NAC",
    ],

    "ribonuclease": [
        "Onconase",
        "Ranpirnase",
    ],

    "rna_synthesis_inhibitor": [
        "Actinomycin D",
        "eribulin",
        "Halaven",
        "α-amanitin",
    ],

    "taxane": [
        "Abraxane",
        "Cabazitaxel",
        "nab-paclitaxel",
        "PTX",
        "Taxol",
        "Taxotere",
    ],

    "topoisomerase_inhibitor": [
        "Camptothecin",
        "CPT",
        "Etoposide",
        "Irinotecan",
        "Topotecan",
    ],

    "tumour_necrosis_factor": [
        "dulanermin",
        "hTRAIL",
        "TNF-α",
        "TRAIL",
    ],

})

cargo_list = sorted({
    x for vals in CARGO_DB.values() for x in vals
})


# 2. HELPERS
SEP = r"[-_\s/]+"          # flexible separator
RNA = r"(?:rna)?"
WORD = r"\b"

def rx(term):
    return WORD + term + WORD


# 3. LITERATURE-SCALE NOMENCLATURE MAP
cargo_map = OrderedDict({
rx(r"her2[-_\s/]+antigen[-_\s/]+mrna"): "HER2 antigen-encoding mRNA",
    rx(r"betulin"): "Betulin",
    rx(r"BN"): "Betulin",
    rx(r"antabuse"): "Disulfiram",
    rx(r"ds"): "Disulfiram",
    rx(r"vinblastine"): "Vinblastine",
    rx(r"vincristine"): "Vincristine",
    rx(r"vinorelbine"): "Vinorelbine",
    rx(r"bcnu"): "Carmustine",
    rx(r"busulfan"): "Busulfan",
    rx(r"carmustine"): "Carmustine",
    rx(r"cyclophosphamide"): "Cyclophosphamide",
    rx(r"melphalan"): "Melphalan",
    rx(r"temozolomide"): "Temozolomide",
    rx(r"tmz"): "Temozolomide",
    rx(r"doxorubicin"): "Doxorubicin",
    rx(r"adriamycin"): "Doxorubicin",
    rx(r"daunorubicin"): "Daunorubicin",
    rx(r"dox"): "Doxorubicin",
    rx(r"doxil"): "Doxorubicin (liposomal formulation)",
    rx(r"epirubicin"): "Epirubicin",
    rx(r"hydroxydaunorubicin"): "Doxorubicin",
    rx(r"idarubicin"): "Idarubicin",
    rx(r"chloroquine"): "Chloroquine",
    rx(r"hydroxychloroquine"): "Hydroxychloroquine",
    rx(r"ifn[- ]?γ"): "Interferon gamma",
    rx(r"il[- ]?2"): "Interleukin-2",
    rx(r"il[- ]?12[-_\s/]+mrna"): "Interleukin-12 mRNA",
    rx(r"il[- ]?23[-_\s/]+mrna"): "Interleukin-23 mRNA",
    rx(r"anti[- ]?pd[- ]?l1"): "Anti-Programmed Death-Ligand 1 antibody",
    rx(r"atezolizumab"): "Atezolizumab",
    rx(r"avelumab"): "Avelumab",
    rx(r"bavencio"): "Avelumab",
    rx(r"durvalumab"): "Durvalumab",
    rx(r"herceptin"): "Trastuzumab",
    rx(r"imfinzi"): "Durvalumab",
    rx(r"ipilimumab"): "Ipilimumab",
    rx(r"keytruda"): "Pembrolizumab",
    rx(r"nivolumab"): "Nivolumab",
    rx(r"opdivo"): "Nivolumab",
    rx(r"pembrolizumab"): "Pembrolizumab",
    rx(r"tecentriq"): "Atezolizumab",
    rx(r"yervoy"): "Ipilimumab",
    rx(r"sacituzumab[-_\s/]+govitecan"): "Sacituzumab Govitecan",
    rx(r"trodelvy"): "Sacituzumab Govitecan",
    rx(r"amphotericin[-_\s/]+b"): "Amphotericin B",
    rx(r"neoantigen[-_\s/]+mrna"): "Neoantigen-encoding mRNA",
    rx(r"ova"): "Ovalbumin",
    rx(r"ova[-_\s/]+peptide"): "Ovalbumin peptide",
    rx(r"tumor[-_\s/]+lysate"): "Tumour cell lysate",
    rx(r"mebendazole"): "Mebendazole",
    rx(r"5[- ]?fluorouracil"): "Fluorouracil",
    rx(r"5[- ]?fu"): "Fluorouracil",
    rx(r"cytarabine"): "Cytarabine",
    rx(r"gemcitabine"): "Gemcitabine",
    rx(r"gem"): "Gemcitabine",
    rx(r"mtx"): "Methotrexate",
    rx(r"pemetrexed"): "Pemetrexed",
    rx(r"acyclovir"): "Acyclovir",
    rx(r"propranolol"): "Propranolol",
    rx(r"dm1"): "Mertansine (emtansine)",
    rx(r"filgrastim"): "Filgrastim",
    rx(r"g[- ]?csf"): "Granulocyte Colony-Stimulating Factor",
    rx(r"granocyte"): "Lenograstim",
    rx(r"lenograstim"): "Lenograstim",
    rx(r"neulasta"): "Pegfilgrastim",
    rx(r"neupogen"): "Filgrastim",
    rx(r"pegfilgrastim"): "Pegfilgrastim",
    rx(r"fosamax"): "Alendronate",
    rx(r"man[- ]?lp@zol"): "Mannosylated Liposome Zoledronic Acid",
    rx(r"man[- ]?np@zol"): "Mannosylated Nanoparticle Zoledronic Acid",
    rx(r"mannosylated[-_\s/]+liposome[-_\s/]+zol"): "Mannosylated Liposome Zoledronic Acid",
    rx(r"zol"): "Zoledronic Acid",
    rx(r"zoledronate"): "Zoledronic Acid",
    rx(r"zometa"): "Zoledronic Acid",
    rx(r"bpa"): "Boronophenylalanine",
    rx(r"bsh"): "Sodium borocaptate",
    rx(r"abemaciclib"): "Abemaciclib",
    rx(r"ibrance"): "Palbociclib",
    rx(r"kisqali"): "Ribociclib",
    rx(r"palbociclib"): "Palbociclib",
    rx(r"ribociclib"): "Ribociclib",
    rx(r"verzenio"): "Abemaciclib",
    rx(r"dnase"): "Deoxyribonuclease",
    rx(r"glucose[-_\s/]+oxidase"): "Glucose oxidase",
    rx(r"l[- ]?asparaginase"): "L-asparaginase",
    rx(r"lynparza"): "Olaparib",
    rx(r"talzenna"): "Talazoparib",
    rx(r"fulvestrant"): "Fulvestrant",
    rx(r"tamoxifen"): "Tamoxifen",
    rx(r"goserelin"): "Goserelin",
    rx(r"leuprolide"): "Leuprolide",
    rx(r"lupron"): "Leuprolide",
    rx(r"zoladex"): "Goserelin",
    rx(r"aq4n"): "Banoxantrone",
    rx(r"tirapazamine"): "Tirapazamine",
    rx(r"cgamp"): "Cyclic GMP-AMP",
    rx(r"cpg[-_\s/]+odn"): "CpG Oligodeoxynucleotide",
    rx(r"dmxaa"): "Vadimezan (5,6-Dimethylxanthenone-4-acetic acid)",
    rx(r"imiquimod"): "Imiquimod",
    rx(r"poly[-_\s/]+i:c"): "Polyinosinic-polycytidylic acid",
    rx(r"r837"): "Imiquimod",
    rx(r"r848"): "Resiquimod",
    rx(r"resiquimod"): "Resiquimod",
    rx(r"sting[-_\s/]+agonist"): "STING agonist (generic)",
    rx(r"tlr9[-_\s/]+agonist"): "TLR9 agonist (generic)",
    rx(r"vadimezan"): "Vadimezan (DMXAA)",
    rx(r"acalabrutinib"): "Acalabrutinib",
    rx(r"afatinib"): "Afatinib",
    rx(r"alecensa"): "Alectinib",
    rx(r"alectinib"): "Alectinib",
    rx(r"aliqopa"): "Copanlisib",
    rx(r"alpelisib"): "Alpelisib",
    rx(r"alunbrig"): "Brigatinib",
    rx(r"bicalutamide"): "Bicalutamide",
    rx(r"brigatinib"): "Brigatinib",
    rx(r"brukinsa"): "Zanubrutinib",
    rx(r"calquence"): "Acalabrutinib",
    rx(r"casodex"): "Bicalutamide",
    rx(r"ceritinib"): "Ceritinib",
    rx(r"copanlisib"): "Copanlisib",
    rx(r"copiktra"): "Duvelisib",
    rx(r"crizotinib"): "Crizotinib",
    rx(r"duvelisib"): "Duvelisib",
    rx(r"erlotinib"): "Erlotinib",
    rx(r"gilotrif"): "Afatinib",
    rx(r"gleevec"): "Imatinib",
    rx(r"ibrutinib"): "Ibrutinib",
    rx(r"imatinib"): "Imatinib",
    rx(r"imbruvica"): "Ibrutinib",
    rx(r"lapatinib"): "Lapatinib",
    rx(r"nexavar"): "Sorafenib",
    rx(r"osimertinib"): "Osimertinib",
    rx(r"piqray"): "Alpelisib",
    rx(r"rap"): "Rapamycin",
    rx(r"sirolimus"): "Rapamycin",
    rx(r"sorafenib"): "Sorafenib",
    rx(r"sunitinib"): "Sunitinib",
    rx(r"sutent"): "Sunitinib",
    rx(r"tagrisso"): "Osimertinib",
    rx(r"tarceva"): "Erlotinib",
    rx(r"tykerb"): "Lapatinib",
    rx(r"xalkori"): "Crizotinib",
    rx(r"zanubrutinib"): "Zanubrutinib",
    rx(r"zykadia"): "Ceritinib",
    rx(r"c[- ]?peptide[- ]?sln[- ]?ptx"): "C-peptide paclitaxel solid lipid nanoparticle",
    rx(r"irgd"): "Internalizing RGD peptide",
    rx(r"mmp[- ]?responsive[-_\s/]+peptide"): "MMP-responsive peptide",
    rx(r"rgd[-_\s/]+peptide"): "Arg-Gly-Asp peptide",
    rx(r"gd[- ]?dtpa"): "Gadolinium Diethylenetriamine Pentaacetic Acid",
    rx(r"ber"): "Berberine",
    rx(r"berberine"): "Berberine",
    rx(r"curcumin"): "Curcumin",
    rx(r"diferuloylmethane"): "Curcumin",
    rx(r"egcg"): "Epigallocatechin Gallate",
    rx(r"epigallocatechin[-_\s/]+gallate"): "Epigallocatechin Gallate",
    rx(r"ginsenoside"): "Ginsenoside",
    rx(r"quercetin"): "Quercetin",
    rx(r"resveratrol"): "Resveratrol",
    rx(r"4[- ]?amino[- ]?tempo"): "4-Amino-TEMPO",
    rx(r"tempo"): "(2,2,6,6-Tetramethylpiperidin-1-yl)oxyl",
    rx(r"aspirin"): "Acetylsalicylic acid",
    rx(r"celecoxib"): "Celecoxib",
    rx(r"indomethacin"): "Indomethacin",
    rx(r"akt[-_\s/]+sirna"): "AKT-targeting siRNA",
    rx(r"anti[- ]?egfr[-_\s/]+sirna"): "Anti-EGFR siRNA",
    rx(r"base[-_\s/]+editor[-_\s/]+brca1"): "Base editor targeting BRCA1",
    rx(r"bcl[- ]?2[-_\s/]+sirna"): "BCL-2-targeting siRNA",
    rx(r"cas12a[-_\s/]+mrna"): "Cas12a nuclease mRNA",
    rx(r"cas9[-_\s/]+\+[-_\s/]+esr1[-_\s/]+sgrna"): "Cas9 + ESR1-targeting sgRNA",
    rx(r"cas9[-_\s/]+\+[-_\s/]+her2[-_\s/]+sgrna"): "Cas9 + HER2-targeting sgRNA",
    rx(r"cas9[-_\s/]+\+[-_\s/]+pik3ca[-_\s/]+sgrna"): "Cas9 + PIK3CA-targeting sgRNA",
    rx(r"cas9[-_\s/]+mrna"): "Cas9 nuclease mRNA",
    rx(r"circrna"): "Circular RNA",
    rx(r"crrna"): "CRISPR RNA",
    rx(r"egfr[-_\s/]+sirna"): "EGFR-targeting siRNA",
    rx(r"ezh2[-_\s/]+sirna"): "EZH2-targeting siRNA",
    rx(r"generic[-_\s/]+sirna"): "Generic siRNA",
    rx(r"gp100[-_\s/]+mrna"): "Glycoprotein 100 mRNA",
    rx(r"her2[-_\s/]+sarna"): "HER2 self-amplifying RNA",
    rx(r"htert[-_\s/]+circrna"): "hTERT circular RNA",
    rx(r"let[- ]?7"): "Let-7 microRNA",
    rx(r"mir[- ]?10b"): "MicroRNA-10b",
    rx(r"mir[- ]?145"): "MicroRNA-145",
    rx(r"mir[- ]?155"): "MicroRNA-155",
    rx(r"mir[- ]?182[- ]?3p"): "MicroRNA-182-3p",
    rx(r"mir[- ]?200"): "MicroRNA-200",
    rx(r"mir[- ]?21"): "MicroRNA-21",
    rx(r"mir[- ]?34a"): "MicroRNA-34a",
    rx(r"mir[- ]?373"): "MicroRNA-373",
    rx(r"mirna"): "MicroRNA",
    rx(r"mmp[- ]?9[-_\s/]+sirna"): "MMP-9-targeting siRNA",
    rx(r"mrna"): "Messenger RNA",
    rx(r"mrna[- ]?4157"): "Personalized neoantigen mRNA (Moderna)",
    rx(r"mrna[- ]?4359"): "Immune checkpoint mRNA (Moderna)",
    rx(r"muc1[-_\s/]+mrna"): "Mucin-1 mRNA",
    rx(r"muc1[-_\s/]+sarna"): "Mucin-1 self-amplifying RNA",
    rx(r"ny[- ]?eso[- ]?1[-_\s/]+sarna"): "NY-ESO-1 self-amplifying RNA",
    rx(r"ox40l[-_\s/]+mrna"): "OX40 Ligand mRNA",
    rx(r"p53[-_\s/]+mrna"): "Tumour protein p53 mRNA",
    rx(r"pi3k[-_\s/]+sirna"): "PI3K-targeting siRNA",
    rx(r"sarna"): "Self-Amplifying RNA",
    rx(r"sgrna"): "Single Guide RNA",
    rx(r"shrna"): "Short Hairpin RNA",
    rx(r"sirna"): "Small Interfering RNA",
    rx(r"stat3[-_\s/]+sirna"): "STAT3-targeting siRNA",
    rx(r"survivin[-_\s/]+sirna"): "Survivin-targeting siRNA",
    rx(r"tgf[- ]?[βb][-_\s/]+trap[-_\s/]+mrna"): "TGF-β trap-encoding mRNA",
    rx(r"trp2[-_\s/]+mrna"): "Tyrosinase-related protein 2 mRNA",
    rx(r"twist1[-_\s/]+sirna"): "Twist1-targeting siRNA",
    rx(r"vegf[-_\s/]+sirna"): "VEGF-targeting siRNA",
    rx(r"wt1[-_\s/]+antigen[-_\s/]+mrna"): "Wilms Tumour 1 antigen mRNA",
    rx(r"nitroglycerin"): "Nitroglycerin",
    rx(r"sildenafil"): "Sildenafil",
    rx(r"bpd"): "Benzoporphyrin Derivative",
    rx(r"ce6"): "Chlorin e6",
    rx(r"icg"): "Indocyanine Green",
    rx(r"phthalocyanine"): "Phthalocyanine",
    rx(r"porphyrin"): "Porphyrin",
    rx(r"verteporfin"): "Benzoporphyrin Derivative (BPD)",
    rx(r"peroxidase"): "Peroxidase",
    rx(r"carboplatin"): "Carboplatin",
    rx(r"cisplatin"): "Cisplatin",
    rx(r"nedaplatin"): "Nedaplatin",
    rx(r"oxaliplatin"): "Oxaliplatin",
    rx(r"bortezomib"): "Bortezomib",
    rx(r"carfilzomib"): "Carfilzomib",
    rx(r"kyprolis"): "Carfilzomib",
    rx(r"velcade"): "Bortezomib",
    rx(r"l[- ]?arginine"): "L-arginine",
    rx(r"l[- ]?glutamine"): "L-glutamine",
    rx(r"cladribine"): "Cladribine",
    rx(r"clofarabine"): "Clofarabine",
    rx(r"fludarabine"): "Fludarabine",
    rx(r"131i"): "Iodine-131",
    rx(r"177lu"): "Lutetium-177",
    rx(r"64cu"): "Copper-64",
    rx(r"89zr"): "Zirconium-89",
    rx(r"90y"): "Yttrium-90",
    rx(r"dtt"): "Dithiothreitol",
    rx(r"glutathione"): "Glutathione",
    rx(r"gsh"): "Glutathione",
    rx(r"n[- ]?acetylcysteine"): "N-acetylcysteine",
    rx(r"nac"): "N-acetylcysteine",
    rx(r"onconase"): "Onconase",
    rx(r"ranpirnase"): "Ranpirnase",
    rx(r"actinomycin[-_\s/]+d"): "Actinomycin D",
    rx(r"eribulin"): "Eribulin mesylate",
    rx(r"halaven"): "Eribulin",
    rx(r"α[- ]?amanitin"): "Alpha-amanitin",
    rx(r"abraxane"): "Paclitaxel (albumin-bound)",
    rx(r"cabazitaxel"): "Cabazitaxel",
    rx(r"nab[- ]?paclitaxel"): "Paclitaxel (albumin-bound)",
    rx(r"ptx"): "Paclitaxel",
    rx(r"taxol"): "Paclitaxel",
    rx(r"taxotere"): "Docetaxel",
    rx(r"camptothecin"): "Camptothecin",
    rx(r"cpt"): "Camptothecin",
    rx(r"etoposide"): "Etoposide",
    rx(r"irinotecan"): "Irinotecan",
    rx(r"topotecan"): "Topotecan",
    rx(r"dulanermin"): "Recombinant human TRAIL",
    rx(r"tnf[- ]?α"): "Tumour Necrosis Factor alpha",
    rx(r"trail"): "Dulanermin (recombinant human TRAIL)",
    rx(r"htrail"): "Dulanermin (recombinant human TRAIL)",
    rx(r"lupeol"): "Lupeol",
    rx(r"fagarasterol"): "Lupeol",
    rx(r"fagarsterol"): "Lupeol",
    rx(r"monogynol[-_\s/]+b"): "Lupeol",
    rx(r"clerodol"): "Lupeol",
    rx(r"farganasterol"): "Lupeol",
    rx(r"lupenol"): "Lupeol",
    rx(r"tsl[- ]?lup"): "Lupeol",
})

# 4. COMPILE
COMPILED_MAP = [
    (re.compile(pattern, re.I), canonical)
    for pattern, canonical in cargo_map.items()
]

# VOCABULARY REGISTRIES
# These are your keyword lists and regex patterns for direct extraction (no LLM needed)

# type
ORGANIC_NP_KEYWORDS = [
    "lipid-polymer","liposome", "lipid nanoparticle", "solid lipid nanoparticle", "liposomal",
    "polymeric nanoparticle", "micelle", "dendrimer", "niosome",
    "exosome", "polymersome", "nanoemulsion", "lipoplex", "polyplex",
    "nanostructured lipid carrier", "cationic liposome", "ionizable LNP", "small unilamellar vesicles", "multilamellar",
]

ORGANIC_NP_ABBR = ["NP", "LNP", "LNPs", "PLGA", "PLA", "SLN", "NLC", "SUV", "MLV", "MLVs", "LUV", "NLC"]

INORGANIC_NP_KEYWORDS = [
    "gold nanoparticle", "iron oxide", "silica", "quantum dot", "silver nanoparticle",
    "zinc oxide", "titanium dioxide", "carbon nanotube", "graphene",
    "mesoporous silica", "calcium phosphate", "copper sulfide",
    "manganese dioxide", "prussian blue",
]

INORGANIC_NP_ABBR = ["AuNP", "SPION", "MSN", "GdNP", "Fe3O4", "TiO2", "ZnO", "AgNP", "SiO2", "CNT", "QD", "SPION", "MSN"]
 
# subtype
SUBTYPE_MAP = {
    "liposome": {
        "keywords": ["liposome", "liposomal", "liposomes","unilamellar", "multilamellar"],
        "abbr": ["SUV", "MLV", "LUV"]
    },

    "LNP": {
        "keywords": ["lipid nanoparticle", "ionizable lipid nanoparticle"],
        "abbr": ["LNP", "LNPs"]
    },

    "SLN": {
        "keywords": ["solid lipid nanoparticle"],
        "abbr": ["SLN"]
    },

    "NLC": {
        "keywords": ["nanostructured lipid carrier"],
        "abbr": ["NLC"]
    },

    "polymeric": {
        "keywords": [ "polymeric nanoparticle", "polymersome", "polyplex",],
        "abbr": ["PLGA","PLA"]
    },

    "micelle": {
        "keywords": ["micelle", "polymeric micelle"],
        "abbr": []
    },

    "dendrimer": {
        "keywords": ["dendrimer",],
        "abbr": ["PAMAM"]
    },

    "niosome": {
        "keywords": ["niosome"],
        "abbr": ["NISM"]
    },

    "lipoplex": {
        "keywords": ["lipoplex", "cationic lipid-DNA complex"],
        "abbr": []
    }
}
 
# ── charge ────────────────────────────────────────────────────────────────────
POSITIVE_KEYWORDS = [
    "positive zeta",
]
NEGATIVE_KEYWORDS = [
    "negative zeta",
]
NEUTRAL_KEYWORDS = [
    "neutral", "zwitterionic", "near-neutral", "PEGylated neutral",
]
 
# ── shape ─────────────────────────────────────────────────────────────────────
SPHERE_KEYWORDS = ["spherical", "sphere", "spheroid"]
ROD_KEYWORDS    = ["rod", "rod-shaped", "elongated", "cylindrical"]
DISK_KEYWORDS   = ["disc", "disk", "discoidal", "flat nanoparticle"]
OTHERS_KEYWORDS  = ["star", "cube", "plate", "clusters"]
 
# ── lamellarity ───────────────────────────────────────────────────────────────
UNILAMELLAR_KEYWORDS   = [ "unilamellar", "small unilamellar", "large unilamellar",]
UNILAMELLAR_ABBR = ["SUV", "SUVs", "LUV"]

MULTILAMELLAR_KEYWORDS = ["multilamellar", "multi-lamellar"]
MULTILAMELLAR_ABBR = ["MLV", "MLVs"]
 
# ── lipid composition ─────────────────────────────────────────────────────────
LIPID_KEYWORDS = [
    # structural phospholipids
    "MSPC", "monostearoyl phosphatidylcholine", "SPC", "DPPC", "DSPC", "DOPC", "DOPE", "HSPC", "hydrogenated soy phosphatidylcholine", "DPPE", "DMPC", "POPC", "POPE",
    "phosphatidylcholine", "sphingomyelin", "span 60", "PS", "SM", "lyso-PC", "lyso-PE",

    "span", "twen"

    # sterols
    "cholesterol hemisuccinate", "CHEMS", "cholesterol", "chol", "CHO", "CHOL", "choles-terol", "beta-sitosterol", "ergosterol",

    # PEG-lipids
    "DSPE-PEG", "DSPE-PEG2000", "DSPE-MPEG2000", "DSPE- PEG2000", "DSPE-PEG_2000", "DSPE- PEG_2000", "DSPE- MPEG2000", "DSPE- MPEG" , "DSPE- PEG",
    "C14-PEG2000","C14- PEG2000", "PEG-DMG", "PEG- DMG", "PEG2000-Cer16", "PEG-DSPE", "PEG-DMPE", "PED-DPPE", "mPEG2000-DSPE", "mPEG2000 DSPE"

    # ionizable lipids
    "DLin-MC3-DMA", "MC3", "SM-102", "ALC-0315",
    "lipid A9", "C12-200", "OF-02", "5A2-SC8", "CKK-E12", "DLin-KC2-DMA",
    "DODAP", "DOTAP", "DOTMA", "GenVoy-ILM", "Lipid H", "DLODAP", "DODMA", "DSDMA", "L-319",
    
    #Imaging lipid
    "Gd.DOTA.DSA", # Gd.DOTA.DSA for MRI imagining
    "CF750.DSA", # for lipid fluorescence imagining 
]


NORMALIZATION_MAP = {
    "monostearoyl phosphatidylcholine": "MSPC",
    "SPC": "MSPC",
    "choles-terol":"cholesterol",
    "CHO": "cholesterol",
    "chol": "cholesterol",
    "CHOL": "cholesterol",
    "DSPE- PEG2000": "DSPE-PEG2000",
    "DSPE- MPEG2000": "DSPE-MPEG2000",
    "DSPE- PEG_2000": "DSPE-PEG2000",
    "DSPE-PEG_2000": "DSPE-PEG2000",
    "mPEG2000-DSPE": "DSPE-PEG2000",  # escolher forma dominante
    "mPEG2000 DSPE": "DSPE-PEG2000",
    "mPEG_2000-DSPE": "DSPE-PEG2000",  # escolher forma dominante
    "mPEG_2000 DSPE": "DSPE-PEG2000",
    "hydrogenated soy phosphatidylcholine": "HSPC",
    "cholesterol hemisuccinate": "CHEMS",
    "sphingomyelin": "SM",
    "phosphatidylserine": "PS",
    "MC3": "DLin-MC3-DMA",

}

PHOSPHATIDYLCHOLINES = {"SPC","DPPC", "DSPC", "DOPC", "DOPE", "HSPC", "DPPE", "DMPC", "POPC", "POPE"}

GENERIC_TERMS = {
      "phosphatidylcholine": PHOSPHATIDYLCHOLINES,
}
 
# ── stimulus responsive ───────────────────────────────────────────────────────
STIMULUS_MAP = {
    "pH-sensitive":          ["pH-sensitive", "pH-responsive", "acid-responsive", "pH-triggered", "endosomal pH", "tumor acidic pH"],
    "Thermosensitive":       ["thermosensitive", "temperature-responsive", "heat-sensitive", "thermo-responsive", "LTSL"],
    "Redox-sensitive":       ["redox-sensitive", "redox-responsive", "GSH-responsive", "glutathione", "disulfide bond", "ROS-responsive", "reactive oxygen species", "H2O2-responsive"],
    "Enzyme-responsive":     ["enzyme-responsive", "MMP-responsive", "protease-triggered", "cathepsin", "furin-cleavable", "MMP-2", "MMP-9", "hyaluronidase-responsive"],
    "Light-triggered":       ["light-triggered", "photo-responsive", "photosensitive", "NIR-responsive", "azobenzene"],
    "Hypoxia-responsive":    ["hypoxia-responsive", "hypoxia-triggered", "nitroimidazole", "azobenzene hypoxia"],
}
 
# bioconjugation
COVALENT_KEYWORDS     = ["covalent", "conjugated", "conjugation", "crosslinked",
                          "thioether", "amide bond", "maleimide", "click chemistry",
                          "NHS ester", "EDC coupling", "disulfide conjugation", "mannose-modified" ]
ELECTROSTATIC_KEYWORDS  = ["electrostatic",	"electrostatically bound", "electrostatically adsorbed", "ionic interaction", "ionic interactions", "ionic complex"
                           	"ionic complexation", "polyelectrolyte complex", "polyelectrolyte complexation", "ion pairing", "charge interaction", "charge-mediated", 
                            "charge-driven assembly", "self-assembly", "layer-by-layer assembly", "LbL assembly", "electrostatic self-assembly", "non-covalent interaction", 
                            "noncovalent interaction", "physical adsorption", "adsorbed", "surface adsorption", "physisorption"]
 
# PEG 
PEG_KEYWORDS     = ["PEGylated", "PEG-coated", "PEG-modified", "DSPE-PEG",
                     "DSPE-MPEG2000", "PEG2000", "stealth liposome", "DMG-PEG2000",
                     "C14-PEG2000", "long-circulating"  "DMG-PEG2000", "PEG-DMG","PEG2000-Cer16",]

NON_PEG_KEYWORDS = ["non-PEGylated", "without PEG", "PEG-free",
                     "PEG-alternative", "polysarcosine", "polyoxazoline", ]
 
# coating 
MONOLAYER_KEYWORDS  = ["monolayer", "single layer", "single coating"]
MULTILAYER_KEYWORDS = ["multilayer", "multi-layer", "layer-by-layer", "LbL",
                        "double coating", "lipid shell", "hybrid core-shell"]

SURFACE_MODIFIER_KEYWORDS = [
    "hyaluronic acid", "HA coating", "HA-coated", "HA-modified",
    "mannose-modified", "folate-modified", "transferrin-modified",
    "aptamer-functionalized", "antibody-conjugated", "decorated"
]

# targeting ─────────────────────────────────────────────────────────────────
ACTIVE_TARGETING_KEYWORDS = [
    "active targeting", "ligand-targeted", "receptor-targeted",
    "antibody-conjugated", "aptamer", "folate-targeted", "transferrin",
    "RGD", "peptide-decorated", "antibody-functionalized",
    # TNBC-relevant receptors
    "EGFR-targeted", "anti-EGFR", "HER1", "EGF receptor",
    "CD44-targeted", "hyaluronic acid-modified", "HA-coated",
    "TRAIL receptor", "DR5", "αvβ3 integrin",
    "PDL1-targeted", "anti-PD-L1 conjugated",
    "nucleolin-targeted", "AS1411",
]

PASSIVE_TARGETING_KEYWORDS = [
    "passive targeting", "passive tumor targeting", "passive tumour targeting", 
    "passive accumulation", "passive delivery",
    #enhanced permeability and retention
    "EPR", "enhanced permeability and retention", "enhanced permeability retention", 
    "EPR effect", "EPR-mediated accumulation", "EPR-mediated targeting"
]


# drug loading ──────────────────────────────────────────────────────────────
DRUG_LOADING_MAP = {
    "Passive entrapment": {
        "keywords": ["passive", "passive entrapment", "passive loading", "thin film hydration", "solvent injection"],
        "abbr": []
    },

    "Active loading (pH gradient)": {
        "keywords": ["active loading", "pH gradient", "remote loading", "ammonium sulfate", "citrate buffer"],
        "abbr": []
    },

    "Bilayer intercalation": {
        "keywords": ["bilayer intercalation", "membrane intercalation", "lipophilic", "hydrophobic drug"],
        "abbr": []
    },

    "Surface conjugation": {
        "keywords": ["surface conjugation", "surface conj", "surface-conjugated", "surface-adsorbed"],
        "abbr": []
    },

    "Nucleic acid complexation": {
        "keywords": ["complexation", "electrostatic complexation", "electrostatic encapsulation", "siRNA loading", "mRNA encapsulation", "plasmid condensation", "siRNA targeting", "mRNA"
        ],
        "abbr": ["N/P"]
    },

    "Antibody-drug conjugate (ADC)": {
        "keywords": [ "antibody-drug conjugate", "drug-linker", "site-specific conjugation"],
        "abbr": ["ADC", "DAR"]
    },

    "Proliposome method": {
        "keywords": [ "proliposome", "pro-liposome"],
        "abbr": []
    },

    "Solvent injection / Nanoprecipitation": {
        "keywords": ["ethanol injection", "solvent injection", "nanoprecipitation", "solvent displacement", "microfluidic mixing", "rapid mixing"],
        "abbr": []
    },

    "Reverse-phase evaporation": {
        "keywords": ["reverse-phase evaporation", "REV method"],
        "abbr": ["REV"]
    },

    "Hydration-assisted encapsulation": {
        "keywords": ["freeze-thaw", "freeze thaw", "rehydration", "hydration"],
        "abbr": []
    },

    "Size reduction / Homogenization": {
        "keywords": ["high-pressure homogenization", "extrusion", "sonication", "microfluidizer"],
        "abbr": ["HPH"]
    },

    "Co-loading": {
        "keywords": ["co-loaded", "dual-loaded", "co-encapsulation", "simultaneous loading"],
        "abbr": []
    }
}
 
# dosing 
SINGLE_DOSE_KEYWORDS   = ["single dose", "single injection", "one injection",
                           "single administration", "single i.v."]

NUMBER_WORDS = r"(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)"

MULTIPLE_DOSE_PATTERNS = [
    rf"every\s+{NUMBER_WORDS}\s+days?",
    r"every\s+other\s+day",
    r"daily",
    r"weekly",
    r"biweekly",
    r"q\.?d\.?",
    r"b\.?i\.?d\.?",
    r"q\d+d",
    rf"\s+{NUMBER_WORDS}\s+times\s+a\s+week",
    r"repeated",
    r"multiple\s+dose",
    r"cycle",
    r"on days",
    r"days",
]
 
# route 
SYSTEMIC_KEYWORDS = ["intravenous", "intraperitoneal", "systemic", "tail vein","intraperitoneally"]
SYSTEMIC_ABBR = ["i.v.", "IV", "i.p.", "IP"]

LOCAL_KEYWORDS = ["intratumoral", "local", "direct injection",
                          "subcutaneous", "intraductal"]
LOCAL_ABBR = ["i.t.", "IT"]

IV_WORDS = ["intravenous", "intravenously", "tail vein",]
IV_ABBR = ["IV", "i.v."]

IV_EXCLUDED = ["iv breast cancer", "stage iv breast cancer"]


IT_WORDS = ["intratumoral", "intra-tumoral"]
IT_ABBR = ["IT", "i.t."]

#others
IP_WORDS = ["intraperitoneal"]
IP_ABBR = ["IP", "i.p."]

INHALATION_WORDS = ["inhalation"]
INHALATION_ABBR = []

ID_WORDS = ["intradermal", "intradermally"]
ID_ABBR = ["ID", "i.d."]

IA_WORDS = ["intra-arterial", "intraarterial"]
IA_ABBR = ["IA", "i.a."]

PERCUTANEOUS_WORDS = ["percutaneous", "transdermal", "transcutaneous"]
PERCUTANEOUS_ABBR = []

PO_WORDS = ["oral"]
PO_ABBR = ["PO", "p.o."]

INTRATRACHEAL_INSTILLATION_WORDS = ["intratracheal instillation"]
INTRATRACHEAL_INSTILLATION_ABBR = []

INTRAMAMMARY_WORDS = ["intramammary"]
INTRAMAMMARY_ABBR = []

SC_WORDS = ["subcutaneous"]
SC_ABBR = ["SC", "s.c."]


# therapy types ─────────────────────────────────────────────────────────────
THERAPY_MAP = {
    "Chemotherapy":              {"keywords": ["chemotherapy", "chemo", "cytotoxic",
                                  "doxorubicin", "paclitaxel", "docetaxel",
                                  "gemcitabine", "carboplatin", "cisplatin",
                                  "eribulin", "capecitabine", "nab-paclitaxel",
                                  "abraxane", "sacituzumab govitecan"],
                                  "abbr": ["ADC"]},

    "Gene therapy":              {"keywords": ["gene therapy", "siRNA", "mRNA", "plasmid",
                                  "gene silencing", "gene delivery", "miRNA",
                                  "antisense oligonucleotide", "shRNA"], 
                                  "abbr": ["ASO","CRISPR", "cas9"]},   

    "Immunotherapy":             {"keywords": ["immunotherapy", "immune checkpoint",
                                  "anti-PD1", "anti-PD-L1",
                                  "checkpoint inhibitor", "atezolizumab",
                                  "pembrolizumab", "nivolumab", "ipilimumab",
                                  "tumor microenvironment reprogramming",
                                  "macrophage polarization", "M1 polarization",
                                  "innate immune activation", "STING agonist",
                                  "toll-like receptor", "TLR agonist"], 
                                  "abbr": ["CTLA-4", "TIM-3", "LAG-3", "PD-L1"]},

    "Photodynamic therapy":      {"keywords": ["photodynamic",  "photosensitizer",
                                  "ROS generation", "singlet oxygen",
                                  "chlorin e6"], 
                                  "abbr": ["PDT",]}, 

    "Photothermal therapy":      {"keywords": ["photothermal", "NIR irradiation",
                                  "laser irradiation", "indocyanine green",
                                  "gold nanorod", "copper sulfide"],
                                  "abbr": ["PTT", "ICG"]},

    "Radiotherapy":              {"keywords": ["radiotherapy", "radiation therapy",
                                  "radiosensitization", "radiodynamic"], 
                                  "abbr": []},

    "Ultrasound":                {"keywords": ["ultrasound", "sonodynamic", "HIFU",
                                  "focused ultrasound"], 
                                  "abbr": ["HIFU"]},

    "Magnetic Hyperthermia":     {"keywords": ["magnetic hyperthermia", "magnetically induced hyperthermia", "magnetic fluid hyperthermia",
                                "magnetic nanoparticle hyperthermia", "MNP hyperthermia", "magnetic heating", "magnetically induced heating",
                                "magnetothermal therapy", "magnetothermal treatment", "magnetothermal"], 
                                  "abbr": ["MFH"]},    

    "Radiofrequency":                {"keywords": ["radiofrequency", "radio frequency"], 
                                  "abbr": ["RF"]},

    "Others":                {"keywords": ["ultrasound", "sonodynamic", "HIFU",
                                  "focused ultrasound"], 
                                  "abbr": ["HIFU"]},   
}
 
# study strategy ────────────────────────────────────────────────────────────
DIAGNOSIS_KEYWORDS    = ["diagnosis", "diagnostic", "imaging only", 
                         "detection", "contrast agent"]

THERAPY_KEYWORDS    = ["therapy", "treatment", "therapeutic", "therapeutics", "drug delivery",
                    "drug release"]

THERANOSTICS_KEYWORDS = ["theranostic", "theragnosis",
                          "combined imaging and therapy", "dual-function",
                          "image-guided therapy"]
 
# tumor model ───────────────────────────────────────────────────────────────
INVIVO_GENERIC_KEYWORDS = [
    "tumor volume", "tumor growth", "tumor regression",
    "PBS control", "treatment group", "body weight loss",
    "in vivo", "mouse model", "mice treated"
]

INVITRO_GENERIC_KEYWORDS = [
    "in vitro", "cell culture", "tissue culture",
    "primary cells", "cell line", "cultured cells",
    "ex vivo", "laboratory assay", "biological assay", 
    "experimental model",
]


XENOGRAFT_KEYWORDS    = ["xenograft", "human tumor", "human cell line",
                          "human cancer cells", "xenografts", "cell into", "cells in BALB/c nude mice", "in SCID mice" ] # MDA-MB-231-BrM brain metastasis model in SCID mice
ALLOGRAFT_KEYWORDS    = ["allograft", "syngeneic", "syngeneic model",
                          "murine tumor","4t1", "ct26", "b16", "llc", "e0771", "BALB/c mice"] # duvida em BALB/c mice
ORTHOTOPIC_KEYWORDS   = ["orthotopic"]
HETEROTOPIC_KEYWORDS  = ["heterotopic", "subcutaneous", "flank"]

CHEMICALLY_INDUCED_WORDS = [
    "chemically induced", "chemically-induced", "chemical carcinogenesis",
    "carcinogen-induced", "carcinogen induced", "induced tumor model",
    "induced tumour model", "urethane-induced"]

VIRUS_INDUCED_WORDS = [
    "virus induced", "virus-induced", "viral-induced", "viral carcinogenesis",
    "oncogenic virus", "viral tumor model", "viral tumour model"]
 
# immune status ─────────────────────────────────────────────────────────────
IMMUNOCOMPROMISED_KEYWORDS = [
    "nude mice", "athymic", "immunodeficient",
    "immunocompromised", "NCr nude",
]
IMMUNOCOMPROMISED_ABBR = [
     "SCID", "NOD/SCID", "NSG", "RAG",
]

IMMUNOCOMPETENT_KEYWORDS = [
    "BALB/c", "C57BL/6", "immunocompetent", "syngeneic", "intact immune",
    "FVB/N", "immune",
]
 
# cancer type ───────────────────────────────────────────────────────────────
CANCER_TYPE_MAP = {
    "Breast":   ["breast", "4T1", "MCF-7", "MDA-MB", "T47D", "BT-474", "SKBR3",
                  "ZR-75", "SUM149", "SUM159", "HCC1806", "HCC1937", "HCC70",
                  "MDA-MB-231", "MDA-MB-468", "MDA-MB-436", "Hs578T",
                  "BT549", "CAL-51", "TNBC"],}
    #"Lung":     ["lung", "A549", "H460", "LLC", "Lewis lung", "H1299","H1975", "PC9"],
    #"Liver":    ["liver", "hepatocellular", "HCC", "HepG2", "Huh7"],
    #"Brain":    ["brain", "glioma", "glioblastoma", "GBM", "U87", "U251","T98G", "LN229"],
    #"Pancreas": ["pancreas", "pancreatic", "PANC-1", "BxPC-3", "MIA PaCa"],
    #"Ovary":    ["ovary", "ovarian", "SKOV3", "A2780", "ID8", "OVCAR"],
    #"Skin":     ["melanoma", "B16", "A375", "skin", "B16F10"],
    #"Cervix":   ["cervix", "cervical", "HeLa"],
    #"Colon":    ["colon", "colorectal", "HCT116", "SW480", "CT26", "HT-29"],
    #"Prostate": ["prostate", "PC3", "LNCaP", "DU145", "22Rv1"],

 
# breast cancer subtype ─────────────────────────────────────────────────────
BREAST_SUBTYPE_MAP = {
    "Triple-Negative (TNBC)": ["triple-negative", "TNBC", "triple negative",
                                "MDA-MB-231", "MDA-MB-468", "MDA-MB-436",
                                "4T1", "SUM149", "SUM159", "HCC1806",
                                "HCC1937", "HCC70", "BT549", "CAL-51",
                                "Hs578T", "ER-negative PR-negative HER2-negative"],
    "TNBC BL1":               ["BL1", "basal-like 1", "BRCA1-mutated TNBC"],
    "TNBC BL2":               ["BL2", "basal-like 2"],
    "TNBC mesenchymal":       ["mesenchymal TNBC", "MSL", "mesenchymal stem-like"],
    "TNBC immunomodulatory":  ["immunomodulatory TNBC", "IM subtype",
                                "immune-rich TNBC", "TIL-high TNBC"],
    "TNBC LAR":               ["LAR", "luminal androgen receptor", "androgen receptor positive TNBC"],
    "HER2-enriched":          ["HER2-positive", "HER2+", "HER2-enriched",
                                "SKBR3", "BT-474"],
    "Luminal A":              ["Luminal A", "ER-positive", "MCF-7", "T47D"],
    "Luminal B":              ["Luminal B"],
    "metastatic":             ["metastatic", "metastasis", "metastatic TNBC",
                                "brain metastasis", "lung metastasis"],
}
 
# imaging modalities ────────────────────────────────────────────────────────
IMAGING_MAP = {
    "MRI": {
        "abbrs": ["MRI", "T1", "T2", "T1-weighted", "T2-weighted", "Gd.DOTA.DSA"],
        "keywords": ["magnetic resonance imaging", "relaxivity"]
    },

    "Fluorescence": {
        "abbrs": ["NIRF", "DiI", "DiD", "DiR", "FITC", "Cy5", "Cy7", "CF750.DSA"],
        "keywords": ["fluorescence", "fluorescent", "near-infrared fluorescence"]
    },

    "PET": {
        "abbrs": ["PET", "18F", "64Cu", "89Zr"],
        "keywords": ["positron emission"]
    },

    "CT": {
        "abbrs": ["CT", "µCT"],
        "keywords": ["computed tomography", "X-ray CT", "micro-CT"]
    },

    "Ultrasound": {
        "abbrs": [],
        "keywords": ["ultrasound imaging", "echography", "US imaging"]
    },

    "Photoacoustic": {
        "abbrs": ["PAI", "MSOT"],
        "keywords": ["photoacoustic", "optoacoustic"]
    },

    "Thermal": {
        "abbrs": [],
        "keywords": ["thermal imaging", "infrared thermography", "IR imaging"]
    },

    "Luminescence": {
        "abbrs": ["IVIS", "luciferase"],
        "keywords": ["luminescence", "bioluminescence", "chemiluminescence"]
    },

    "Others": {
        "abbrs": ["SPECT", "99mTc", "111In"],
        "keywords": ["single-photon emission"]
    },
    "Others":{
        "abbrs": [],
        "keywords": ["cytometry"]
    },
    "Others":{
        "abbrs": [],
        "keywords": ["single-photon emission", "magnetic particle imaging", "magnetoacoustic tomography",
        "optical imaging", "thermoacoustic", "x-ray imaging", "infrared",
        "near-infrared", "raman", "optoacoustic tomography", "mammography"]
    },

}
 
# off-target organs note: hepatic accumulation = liver acc., 
ORGAN_KEYWORDS = [
    "hepatic", "splenic", "liver", "spleen", "kidney", "lung", "heart", "brain",
    "intestine", "stomach", "muscle", "bone marrow", "thymus", "bladder",
    "lymph node", "skin", "ovary", "thyroid gland",
    # Additional common organs / tissues
    "pancreas", "pancreatic",
    "adrenal", "adrenal gland",
    "prostate", "prostatic",
    "testis", "testes", "testicular",
    "uterus", "uterine",
    "cervix", "cervical",
    "vagina", "vaginal",
    "breast", "mammary",
    "esophagus", "esophageal",
    "colon", "colonic",
    "rectum", "rectal",
    "duodenum", "jejunum", "ileum",
    "gallbladder", "biliary", "bile duct",
    "appendix", "appendiceal",
    "tonsil", "tonsillar",
    "salivary gland", "parotid", "submandibular",
    "pituitary", "pituitary gland",
    "pineal",
    "hypothalamus",
    "spinal cord",
    "nerve", "neural",
    "blood", "vascular", "vessel",
    "aorta",
    "vein", "venous",
    "artery", "arterial",
    "peritoneum", "peritoneal",
    "pleura", "pleural",
    "pericardium", "pericardial",
    "diaphragm",
    "fat", "adipose",
    "soft tissue",
    "cartilage",
    "tendon",
    "ligament",
    "joint", "synovial",
    "eye", "ocular", "retina",
    "ear", "otic",
    "nose", "nasal",
    "sinus",
    "pharynx", "larynx", "trachea", "bronchus", "bronchial",
]

 
TUMOR_PATTERNS = r"""
(?:
    (?:reduc(?:e|ed|tion|es)|decreas(?:e|ed|es)|shrink(?:age|ing|es)?)
        \s+(?:in\s+)?tumou?r\s*(?:volume|size)
  | tumou?r\s*(?:volume|size).*?
        (?:reduc(?:e|ed|tion|es)|decreas(?:e|ed|es)|shrink(?:age|ing|es)?)
  | tumou?r.*?(?:shrinkage|regression)
  | tumou?r\s*growth\s*(?:inhibition|inhibited)
  | (?:inhibition|suppression)\s+of\s+tumou?r\s*growth
  | \bTGI\b
  | tumor\s*size\s*(?:reduction|decrease|shrinkage)
  | tumour\s*size\s*(?:reduction|decrease|shrinkage)
  | tumor\s*volume\s*(?:reduction|decrease|shrinkage)
  | tumour\s*volume\s*(?:reduction|decrease|shrinkage)
)
"""


TUMOR_PATTERNS_SIZE_VOLUME = r"""
(?:
    tumou?r\s*size
  | tumou?r\s*volume
  | tumour\s*burden
  | lesion\s*size
  | lesion\s*volume
  | mass\s*size
  | mass\s*volume
  | tumour\s*diameter
  | tumor\s*diameter
  | longest\s*diameter
  | mean\s*tumou?r\s*volume
  | average\s*tumou?r\s*volume
  | final\s*tumou?r\s*volume
  | initial\s*tumou?r\s*volume
)
"""

EE_PATTERNS = r"""
(?:
    encapsulation\s*efficienc(?:y|ies)
  |efficienc(?:y|ies)\s+of\s+encapsulation
  |\bEE\b\s*%?
  |drug\s*loading\s*efficienc(?:y|ies)
  |loading\s*efficienc(?:y|ies)
  |efficienc(?:y|ies)\s+of\s+drug\s*loading
  |\bDL\b\s*%?
  |encapsulation\s*(?:rate|ratio)
  |(?:encapsulation|loading).{0,40}?(?:efficienc(?:y|ies))
  | (?:encapsulated).{0,40}?(?:efficienc(?:y|ies))
  |entrapment\s*efficienc(?:y|ies)
)
"""

DELIVERY_E_KEYWORDS = r"""
(?:delivery\s+efficiency(?:\W+\w+){0,12}?\W+reached
|delivery\s+efficiency(?:\W+\w+){0,10}?\W+was
|delivery\s+efficiency
|cellular\s+uptake
|uptake
|internalization
|transfection\s+efficiency
|transfection
|released)
"""

PART_SIZE = r"""
(?: 
particl(?:e|es)\s+size
|diamete(?:r|rs)
|size(?:s)?
|Dh
|D_h
|Dₕ
)
"""

FORMULATION_CODE_PATTERN = re.compile(
    r"\b("
    r"[A-Za-z0-9]{2,6}\s*[-*@]\s*[A-Za-z0-9]{1,8}(?:\s*[-*]\s*[A-Za-z0-9]{1,4})?"
    r"|[A-Z][a-z]{1,4}\s*[-_]?\s*\d{1,3}"
    r")\b"
)

# common false-positive tokens to discard outright (units, methods, etc.)
FORMULATION_CODE_STOPLIST = {
    "DLS", "PDI", "REV", "TEM", "SEM", "EE", "PBS", "NaCl", "DMSO",
}

def lookup_cargo_category(cargo_name: str, cargo_db: dict = CARGO_DB) -> str | None:
    """Deterministic lookup: which CARGO_DB category does this drug belong to?"""
    name_norm = cargo_name.strip().lower()
    for category, members in cargo_db.items():
        for m in members:
            if m.strip().lower() == name_norm:
                return category
    return None





# HELPER FUNCTIONS
def _match_abbreviation(text: str, abbrs: List[str]) -> bool:
    for a in abbrs:
        pattern = r"(?<![A-Za-z0-9])" + re.escape(a) + r"(?![A-Za-z0-9])"
        if re.search(pattern, text):
            return True
    return False

def _first_keyword_match(text: str, keywords: List[str]) -> bool:
    """Return True if any keyword from the list is found in text (case-insensitive)."""
    text_lower = text.lower()

    for kw in keywords:
        if kw.lower() in text_lower:
            print(f"Keyword encontrada: {kw}")
            return True

    return False

def _abr_first_keyword_match(text: str, words: List[str], abbrs: List[str]) -> bool:
    return _first_keyword_match(text, words) or _match_abbreviation(text, abbrs)


def _all_keyword_matches(text: str, keywords: List[str]) -> List[str]:
    """Return all keywords from the list found in text (case-insensitive)."""
    text_lower = text.lower()
    return list({kw for kw in keywords if kw.lower() in text_lower})

def _map_exact_keywords(text: str, mapping: Dict[str, List[str]]) -> Optional[str]:
    """
    Given a dict of {label: [keywords]}, return the first label whose
    keywords are found in text. Returns None if no match.
    """
    for label, keywords in mapping.items():
        if _match_abbreviation(text, keywords):
            return label
    return None

def _map_keywords(text: str, mapping: Dict[str, List[str]]) -> Optional[str]:
    """
    Given a dict of {label: [keywords]}, return the first label whose
    keywords are found in text. Returns None if no match.
    """
    for label, keywords in mapping.items():
        if _first_keyword_match(text, keywords):
            return label
    return None

def _is_excluded(text:str, excluded_phrases: List[str]) -> Optional[str]:
    text_lower = text.lower()
    return any(p in text_lower for p in excluded_phrases)


def _all_map_matches(text: str, mapping: Dict[str, List[str]]) -> List[str]:
    """Return all labels whose keywords are found in text."""
    found = []
    for label, data in mapping.items():
        if _abr_first_keyword_match(text, data["keywords"], data["abbr"]):
            found.append(label)
    return found

def _find_matches_with_positions(text:str, rules_dict: Dict[str, Dict[str, List[str]]]) -> List[Tuple[int, str]]:
    """
    rules_dict = {label: {"keywords": [...], "abbr": [...]}}
    returns: list of (position, label)
    """

    results = []
    text_lower = text.lower()

    for label, rules in rules_dict.items():

        # keywords
        for kw in rules["keywords"]:
            pos = text_lower.find(kw.lower())
            if pos != -1:
                results.append((pos, label))

        # abbreviations (regex for word boundary)
        for ab in rules["abbr"]:
            pattern = r"(?<![A-Za-z0-9])" + re.escape(ab) + r"(?![A-Za-z0-9])"
            match = re.search(pattern, text)
            if match:
                results.append((match.start(), label))

    return results

def _extract_float_near_keyword(text: str, keyword_pattern: str) -> Optional[float]:
    """
    Extract the float whose position is closest to the keyword occurrence in the text.
    Closeness is measured by absolute character distance.
    """
    # Find keyword
    keyword_match = re.search(keyword_pattern, text, re.IGNORECASE)
    if not keyword_match:
        return None

    keyword_pos = keyword_match.start()

    # Find all numbers in text
    number_pattern = r"[-+]?\d*\.?\d+"
    candidates = [(m.start(), m.group()) for m in re.finditer(number_pattern, text)]

    if not candidates:
        return None

    # Pick number closest to keyword position
    closest_value = None
    min_distance = float("inf")

    for pos, num_str in candidates:
        distance = abs(pos - keyword_pos)
        if distance < min_distance:
            min_distance = distance
            try:
                closest_value = float(num_str)
            except ValueError:
                continue

    return closest_value


def _extract_value_unit_closest_to_keyword(
    text: str,
    keyword_pattern: str,
    units: list[str] = None,
    max_distance: int =400
) -> Optional[Tuple[float, Optional[str], Optional[float], Optional[str]]]:
    text = text.replace("−", "-")

    unit_pattern = "|".join(re.escape(u) for u in units)

    unit_matches = list(re.finditer(
        rf"(\d+(?:\.\d+)?)\s*({unit_pattern})",
        text,
        re.IGNORECASE
    ))

    if not unit_matches:
        return None

    kw_matches = list(re.finditer(
        keyword_pattern,
        text,
        re.IGNORECASE | re.VERBOSE
    ))
    print(kw_matches)

    if not kw_matches:
        return None

    best = None
    best_dist = float("inf")

    for kw in kw_matches:
        kw_center = (kw.start() + kw.end()) / 2

        for unit in unit_matches:
            unit_center = (unit.start() + unit.end()) / 2
            dist = abs(unit_center - kw_center)

            if dist < best_dist and dist <= max_distance:
                best_dist = dist
                best = unit

    if best:
        value = float(best.group(1))
        unit = best.group(2)
        return value, None, None, unit

    return None
    return None


class NanoparticleExtractor(BaseModel):
    """
    Hybrid extractor: uses keyword search + regex + parsers for ~42 features,
    """
    _errors: List[Dict[str, Any]] = PrivateAttr(default_factory=list)

    # MATERIAL PROPERTIES

    def _extract_type(self, text: str) -> Optional[str]:
        if _first_keyword_match(text, ORGANIC_NP_KEYWORDS):
            return "organic"
        if _first_keyword_match(text, INORGANIC_NP_KEYWORDS):
            return "inorganic"
        return None

    def _extract_subtype(self, text: str) -> Optional[str]:
        for subtype, rules in SUBTYPE_MAP.items():
            if _abr_first_keyword_match(text, rules["keywords"], rules["abbr"]):
                return subtype
            return None

    def _extract_lamellarity(self, text: str) -> Optional[str]:
        if _abr_first_keyword_match(text, MULTILAMELLAR_KEYWORDS, MULTILAMELLAR_ABBR):
            return "Multilamellar (MLV)"
        if _abr_first_keyword_match(text, UNILAMELLAR_KEYWORDS, UNILAMELLAR_ABBR):
            return "Unilamellar (SUV/LUV)"
        return None
    
    def _extract_charge_group(self, text: str) -> Optional[str]:
        
        if _first_keyword_match(text, POSITIVE_KEYWORDS):
            return "positive"
        if _first_keyword_match(text, NEGATIVE_KEYWORDS):
            return "negative"
        if _first_keyword_match(text, NEUTRAL_KEYWORDS):
            return "neutral"
        
        return None
     
    def _extract_zeta_potential(self, text: str) -> Optional[bool]:
        normalized = (text.replace("±", "+/-").replace("−", "-").replace("–", "-"))
        match = re.search(r"(-?\d+(?:\.\d+)?(?:\s*\+/-\s*\d+(?:\.\d+)?)?)\s*mV", normalized, re.IGNORECASE)
        #print (match.group(0).replace("+/-","±") if match else None)
        return True if match else False
        
    def _extract_size(self, text: str) -> Optional[bool]:
        text = text.replace(':nm','nm')
        result = _extract_value_unit_closest_to_keyword(text, PART_SIZE, ["nm"])
        #result = _extract_value_near_keyword(text, PART_SIZE, ["nm"])
        #print (result)
        return True if result else False 

    def _extract_pdi(self, text: str) -> Optional[bool]:
        PDI = (_extract_float_near_keyword(text, r"PDI|polydispersity index|polydispersity"))
        #print (PDI)
        return True if PDI else False

    def _extract_shape(self, text: str) -> Optional[str]:
        if _match_abbreviation(text, ROD_KEYWORDS):
            return "Rod"
        if _match_abbreviation(text, SPHERE_KEYWORDS):
            return "Sphere"
        if _match_abbreviation(text, DISK_KEYWORDS):
            return "Disk"
        if _match_abbreviation(text, OTHERS_KEYWORDS):
            return "Other"
        
        return None
    
    # SURFACE ENGINEERING

    def _extract_coating_number(self, text: str) -> Optional[str]:
        if _first_keyword_match(text, MULTILAYER_KEYWORDS):
            return "multilayer"
        if _first_keyword_match(text, MONOLAYER_KEYWORDS):
            return "monolayer"
        return None
    
    def _extract_stimulus_responsive(self, text: str) -> Optional[str]:
        result = _map_keywords(text, STIMULUS_MAP)
        return result if result else None

    def _extract_bioconjugation(self, text: str) -> Optional[bool]:
        if _first_keyword_match(text, COVALENT_KEYWORDS):
            return True
        if _first_keyword_match(text, ELECTROSTATIC_KEYWORDS):
            return True
        return False
    

    def _extract_peg_coat(self, text: str) -> Optional[str]:
        if _first_keyword_match(text, NON_PEG_KEYWORDS):
            return "non-PEGylated"
        if _first_keyword_match(text, PEG_KEYWORDS):
            return "PEGylated"
        return None

    def _extract_targeting_type(self, text: str) -> Optional[str]:
        if _first_keyword_match(text, ACTIVE_TARGETING_KEYWORDS):
            return "active"
        return None

    def _extract_lipid_composition(self, text: str) -> Optional[List[str]]:
        pattern = r"\b(" + "|".join(map(re.escape, LIPID_KEYWORDS)) + r")\b"
        matches = [m.group(0) for m in re.finditer(pattern, text, re.IGNORECASE)]

        normalized = []
        for m in matches:
            m.lower()
            if m in NORMALIZATION_MAP:
                normalized.append(NORMALIZATION_MAP[m])
            else:
                normalized.append(m)

        seen = dict.fromkeys(normalized)
        normalized = list(seen)


        for generic, specific_set in GENERIC_TERMS.items():
            if generic in normalized and any(spec in normalized for spec in specific_set):
                normalized.remove(generic)

        return normalized if normalized else None
    

    def _extract_lipid_ratio(self, text: str) -> Optional[Dict]:
        match = re.search(
            r'\b\d+(?:\.\d+)?(?:\s*[:/]\s*\d+(?:\.\d+)?)+\b',
            text
        )

        if not match:
            return None

        ratio_text = match.group()

        values = [float(n) for n in re.split(r'\s*[:/]\s*', ratio_text)]

        composition = self._extract_lipid_composition(text)

        if not composition:
            return None

        if len(values) < len(composition):
            return None

        if len(values) > len(composition):
            print("[WARNING]: more ratio values than lipid components. "
                "Check the LIPID_KEYWORDS and NORMALIZATION_MAP for consistency.")

        total = sum(values)

        if total == 0:
            return None

        percentages = [(v / total) * 100 for v in values]

        return {
            "ratios": percentages
        }
    
    def _extract_lipid_ratio_units(self, text: str) -> Optional[bool]:
        if self._extract_lipid_ratio(text) == None or len(self._extract_lipid_ratio(text)["ratios"]) != len(self._extract_lipid_composition(text)):
            return False
        return True 

    # DOSING & THERAPY
    def _extract_drug_loading_method(self, text: str) -> Optional[str]:
        for subtype, rules in DRUG_LOADING_MAP.items():
            if _abr_first_keyword_match(text, rules["keywords"], rules["abbr"]):
                return subtype
            return None
    
    def _extract_therapies(self, text: str):
        """Returns (therapy_a, therapy_b, therapy_c, combined_grouped)."""
        all_therapies = _all_map_matches(text, THERAPY_MAP)
        therapy_a = all_therapies[0] if len(all_therapies) > 0 else None
        therapy_b = all_therapies[1] if len(all_therapies) > 1 else None
        therapy_c = all_therapies[2] if len(all_therapies) > 2 else None
        if therapy_b: # or _first_keyword_match(text, "combined therapy"):
            combined  = "Combination therapy" 
        elif therapy_a == "Combination therapy":
            combined  = "Combination therapy"
        elif therapy_a:
            combined = "Monotherapy" 
        else: 
            combined = None
        return therapy_a, therapy_b, therapy_c, combined

    def _extract_study_strategy(self, text: str) -> Optional[str]:
        if _first_keyword_match(text, THERANOSTICS_KEYWORDS):
            return "Theranostics"
        if _first_keyword_match(text, DIAGNOSIS_KEYWORDS):
            return "Diagnosis"
        if _first_keyword_match(text, THERAPY_KEYWORDS):
            return "Therapy"
        return None
        

    def _extract_route(self, text: str) -> Optional[list[str]]:
        rules = {
            "Local": {
                "keywords": LOCAL_KEYWORDS,
                "abbr": LOCAL_ABBR
            },
            "Systemic": {
                "keywords": SYSTEMIC_KEYWORDS,
                "abbr": SYSTEMIC_ABBR
            }
        }


        matches = _find_matches_with_positions(text, rules)
        matches.sort(key=lambda x: x[0])

        filtered = [
            (pos, label)
            for (pos, label) in matches
            if not (
                label == "Systemic"
                and _is_excluded(text, IV_EXCLUDED)
            )
        ]
        return [label for _, label in filtered]

    def _extract_route_subtype(self, text: str) -> Optional[list[str]]:
        rules = {
            "Intravenous": {
                "keywords": IV_WORDS,
                "abbr": IV_ABBR
            },
            "Intratumoral": {
                "keywords": IT_WORDS,
                "abbr": IT_ABBR
            },
            "Other": {
                "keywords": IP_WORDS,
                "abbr": IP_ABBR
            },
            "Other": {
                "keywords": INHALATION_WORDS,
                "abbr": INHALATION_ABBR
            },
            "Other": {
                "keywords": ID_WORDS,
                "abbr": ID_ABBR
            },
            "Other": {
                "keywords": IA_WORDS,
                "abbr": IA_ABBR
            },
            "Other": {
                "keywords": PERCUTANEOUS_WORDS,
                "abbr": PERCUTANEOUS_ABBR
            },
            "Other": {
                "keywords": PO_WORDS,
                "abbr": PO_ABBR
            },
            "Other": {
                "keywords": INTRATRACHEAL_INSTILLATION_WORDS,
                "abbr": INTRATRACHEAL_INSTILLATION_ABBR
            },
            "Other": {
                "keywords": INTRAMAMMARY_WORDS,
                "abbr": INTRAMAMMARY_ABBR
            },
            "Other": {
                "keywords": SC_WORDS,
                "abbr": SC_ABBR
            }
        }

        matches = _find_matches_with_positions(text, rules)
        matches.sort(key=lambda x: x[0])

        filtered = [
            (pos, label)
            for (pos, label) in matches
            if not (
                label == "Intravenous"
                and _is_excluded(text, IV_EXCLUDED)
            )
        ]

        return [label for _, label in filtered]

    def _extract_dose(self, text: str) -> Optional[bool]:
        """
        ParametersParser extracts mg/kg values if you add 'mg/kg' to
        quantity_units in your synthesis_parsing_parameters.json config.
        Fallback: direct regex.
        """
        # Dose por peso corporal (mg/kg, µg/kg, ...)
        if re.search(
            r"\b\d+(?:\.\d+)?\s*(?:mg|µg|μg|ug|g)\s*/\s*kg\b",
            text,
            re.IGNORECASE,
        ):
            return True

        if re.search(
            r"\b\d+(?:\.\d+)?\s*(?:mgkg|µgkg|μgkg|ugkg|gkg)",
            text,
            re.IGNORECASE,
        ):
            return True
        
        # Massa simples, mas NÃO seguida de mL ou L
        if re.search(
            r"\b\d+(?:\.\d+)?\s*(?:mg|µg|μg|ug|g|kg)\b(?!\s*(?:/|\b(?:mL|L)\b))",
            text,
            re.IGNORECASE,
        ):
            return True

        return False
        

    def _extract_dosing_schedule(self, text: str) -> Optional[str]:
        if self._extract_dose(text)==False:
            return None
        if _first_keyword_match(text, SINGLE_DOSE_KEYWORDS):
            return "Single Dose"
        for pattern in MULTIPLE_DOSE_PATTERNS:
            if re.search(pattern, text):
                return "Multiple-dose"
        return None


    # BIOLOGICAL CONTEXT

    def _extract_tumor_model(self, text: str) -> Optional[str]:
        is_xeno  = _first_keyword_match(text, XENOGRAFT_KEYWORDS)
        is_allo  = _first_keyword_match(text, ALLOGRAFT_KEYWORDS)
        is_ortho = _first_keyword_match(text, ORTHOTOPIC_KEYWORDS)
        is_heter = _first_keyword_match(text, HETEROTOPIC_KEYWORDS)
        base = "Xenograft" if is_xeno else ("Allograft" if is_allo else None)
        loc  = "Orthotopic" if is_ortho else ("Heterotopic" if is_heter else None)
        if base and loc:
            return f"{base} {loc}"
        elif base:
            return f"{base}"
        elif loc:
            return f"{loc}"
        elif _first_keyword_match(text, CHEMICALLY_INDUCED_WORDS):
                return "Other"
        elif _first_keyword_match(text, VIRUS_INDUCED_WORDS):
                return "Other"
        #elif _first_keyword_match(text, INVIVO_GENERIC_KEYWORDS):
         #       return "In vivo generic keyword"
        #else: 
        #    if _first_keyword_match(text, INVITRO_GENERIC_KEYWORDS):
                return "In vitro generic keyword"

    def _extract_immune_status(self, text: str) -> Optional[str]:
        if _abr_first_keyword_match(text, IMMUNOCOMPROMISED_KEYWORDS, IMMUNOCOMPROMISED_ABBR):
            return "Immunocompromised"
        if _first_keyword_match(text, IMMUNOCOMPETENT_KEYWORDS):
            return "Immunocompetent"
        return None

    def _extract_cancer_type(self, text: str) -> Optional[str]:
        return _map_exact_keywords(text, CANCER_TYPE_MAP)

    def _extract_breast_subtype(self, text: str) -> Optional[str]:
        return _map_exact_keywords(text, BREAST_SUBTYPE_MAP)

    def _extract_imaging(self, text: str):
        found = []
        for modality, patterns in IMAGING_MAP.items():
            if _abr_first_keyword_match(text, patterns["keywords"], patterns["abbrs"]):
                found.append(modality)

        a = found[0] if len(found) > 0 else None
        b = found[1] if len(found) > 1 else None
        c = found[2] if len(found) > 2 else None

        return a, b, c


    # OUTCOMES

    def _extract_ic50(self, text: str) -> bool:
        return True if re.search(r"IC\s*50|IC₅₀|IC_50", text, re.IGNORECASE) else False

    def _extract_encapsulation_efficiency(self, text: str) ->  Optional[bool]:
        EE = _extract_value_unit_closest_to_keyword(
            text, EE_PATTERNS, ["%"], 200
        )
        #print (EE)
        return True if EE else False 

    
    def _extract_delivery_efficiency(self, text: str) -> Optional[bool]:
        DE = _extract_value_unit_closest_to_keyword(
            text,
            DELIVERY_E_KEYWORDS,
            ["%"])
        #print (DE)
        return True if DE else False

    def _extract_distribution_half_life(self, text: str) -> Optional[bool]:
        DHL= _extract_float_near_keyword(
            text, r"t½α|t1/2α|distribution half.life|alpha half.life"
        )
        #print (DHL)
        return True if DHL else False

    def _extract_circulation_half_life(self, text: str) -> Optional[float]:
        CHL = _extract_float_near_keyword(
            text, r"t½β|t1/2β|circulation half.life|blood circulation.*half|elimination half.life|t1/2|t½"
        )
        #print (CHL)
        return True if CHL else False
    
    def _extract_biodistribution(self, text: str)-> Optional[bool]:
        """
        Returns list of (organ, pct) tuples sorted by % value descending.
        Looks for patterns like 'liver accumulation was 15%' or '20% in spleen'.
        """
        for organ in ORGAN_KEYWORDS:
            # pattern: organ name near a percentage value
            pattern = (
                rf"(?:{organ}).{{0,80}}?(\d+(?:\.\d+)?)\s*%"
                rf"|"
                rf"(\d+(?:\.\d+)?)\s*%.{{0,80}}?(?:{organ})"
            )

            if re.search(pattern, text, re.IGNORECASE):
                return True

        pattern = r"%\s*ID(?:\s*/\s*g)?"

        if re.search(pattern, text, re.IGNORECASE):
            return True

        return False

    def _extract_tumor_reduction(self, text: str) -> Optional[bool]:
        TVL = _extract_value_unit_closest_to_keyword(
            text,
            TUMOR_PATTERNS,
            ["%"])
        #print (TVL)
        return True if TVL else False
    
    def _extract_tumor_size_volume(self, text: str) -> Optional[bool]:
        if re.search(TUMOR_PATTERNS_SIZE_VOLUME, text, re.IGNORECASE | re.VERBOSE):
            return True
        
        return False
    
    # CARGO

    def extract_cargos(self, text: str)-> Optional[List]:
        """Return all unique cargos in order found."""
        found = []
        for pat, name in COMPILED_MAP:
            if pat.search(text) and name not in found:
                found.append(name)
        return found
    
    def cargo_category(self, name: str) -> Optional[str]:
        for cat, vals in CARGO_DB.items():
            if name in vals:
                return cat
        return None

    def extract_cargo_categories(self, text: str) -> Optional[list]:
        cargos_list = self.extract_cargos(text)

        if cargos_list:
            categories = []

            for item in cargos_list:
                c = self.cargo_category(item)
                categories.append(c)

            return categories

        return None
    
    def harvest_formulation_candidates(self, text: str) -> list[str]:
        """Cheap, high-recall scan for possible formulation codes/abbreviations."""
        found = FORMULATION_CODE_PATTERN.findall(text)

        candidates = []
        seen = set()

        for tok in found:
            # Remove spaces inside the formulation code
            cleaned_tok = tok.replace(" ", "")

            key = cleaned_tok.upper()

            if key in FORMULATION_CODE_STOPLIST:
                continue

            if key in seen:
                continue

            seen.add(key)
            candidates.append(cleaned_tok)

        return candidates


    def extract(self, text: str) -> NanoparticleData:
        """ 
        Extract all nanoparticle features from a paper section text.

        Args:
            text:    Relevant paper section text (after paragraph_classifier filtering)
            use_llm: Set False to skip LLM calls (e.g. for testing / fast runs)

        Returns:
            NanoparticleData with all extractable fields populated
        """ 
        therapy_a, therapy_b, therapy_c, combined = self._extract_therapies(text)
        imaging_a, imaging_b, imaging_c = self._extract_imaging(text)


        data = NanoparticleData(
            # Material properties
            type                    = self._extract_type(text),
            subtype                 = self._extract_subtype(text),
            size_nm                 = self._extract_size(text),
            charge_group            = self._extract_charge_group(text),
            shape                   = self._extract_shape(text),
            lamellarity             = self._extract_lamellarity(text),
            zeta_potential_mv       = self._extract_zeta_potential(text),
            pdi                     = self._extract_pdi(text),

            # Surface engineering
            coating_number          = self._extract_coating_number(text),
            lipid_composition       = self._extract_lipid_composition(text),
            lipid_composition_ratio = self._extract_lipid_ratio(text),
            stimulus_responsive     = self._extract_stimulus_responsive(text),
            bioconjugation_nature   = self._extract_bioconjugation(text),
            peg_coat                = self._extract_peg_coat(text),
            targeting_type          = self._extract_targeting_type(text),

            # Dosing & therapy
            drug_loading_method     = self._extract_drug_loading_method(text),
            dose_group        = self._extract_dose(text),
            no_days_dosing_grouped  = self._extract_dosing_schedule(text),
            route                   = self._extract_route(text),
            route_subtype           = self._extract_route_subtype(text),
            therapy_a               = therapy_a,
            therapy_b               = therapy_b,
            therapy_c               = therapy_c,
            combined_therapy_grouped= combined,
            study_strategy          = self._extract_study_strategy(text),

            # Biological context
            tumor_model             = self._extract_tumor_model(text),
            immune_status           = self._extract_immune_status(text),
            cancer_type             = self._extract_cancer_type(text),
            breast_cancer_subtype   = self._extract_breast_subtype(text),
            imaging_a               = imaging_a,
            imaging_b               = imaging_b,
            imaging_c               = imaging_c,

            # Outcomes
            ic50                    = self._extract_ic50(text),
            encapsulation_efficiency_pct = self._extract_encapsulation_efficiency(text),
            distribution_half_life_h = self._extract_distribution_half_life(text),
            circulation_half_life_h = self._extract_circulation_half_life(text),
            tumor_reduction = self._extract_tumor_reduction(text),
            tumor_size_or_volume = self._extract_tumor_size_volume(text),
            biodistribution = self._extract_biodistribution(text),
            cargos = self.extract_cargos(text),
            categories_cargos = self.extract_cargo_categories(text),
            formulations = self.harvest_formulation_candidates(text)

        )

        return data

    def extract_from_txt(
        self,
        txt_path: str,
        separator: str = "\n\n",
        save_json: str = None,
    ):
        """
        Apply self.extract() to each paragraph in txt
        Return JSON-ready results only.
        Works whether extract() returns dataclass, dict, or object.
        """

        text = Path(txt_path).read_text(encoding="utf-8", errors="ignore")
        paragraphs = [p.strip() for p in text.split(separator) if p.strip()]

        results = []

        for i, paragraph in enumerate(paragraphs, start=1):
            data = self.extract(paragraph)

            # convert result safely
            if is_dataclass(data):
                extracted = asdict(data)
            elif isinstance(data, dict):
                extracted = data
            else:
                extracted = vars(data)

            results.append({
                "paragraph_id": i,
                "paragraph": paragraph,
                "extracted": extracted
            })

        if save_json:
            with open(save_json, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

        return results