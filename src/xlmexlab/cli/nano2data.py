import json
import os
import re
import time
import logging
from dataclasses import asdict, is_dataclass

import click
import torch
import traceback

from xlmexlab.extractor_nanoparticles import NanoparticlesExtractorParagraph
from xlmexlab.nanoparticle_paragraph import NanoparticleExtractor
from xlmexlab.prompt import TEMPLATE_REGISTRY

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

TERMINAL_RX = re.compile(r'[.!?]["”’)\]]*\s*$')           # ends a sentence
CONTINUATION_START_RX = re.compile(r'^\s*(?:\d[\d.,]*\s*)?[a-z]')  # lowercase, or number + lowercase


def _is_unfinished(text: str) -> bool:
    return bool(text.strip()) and not TERMINAL_RX.search(text.strip())


def _is_continuation(text: str) -> bool:
    return bool(CONTINUATION_START_RX.match(text))


def merge_split_paragraphs(blocks: list) -> list:
    """
    Merge paragraphs broken by a column/page break:
      - previous paragraph does not end with . ! ?  (closing quotes/brackets allowed)
      - next paragraph starts with lowercase, or number + lowercase
    The merged block keeps the first block's page/box; the other boxes are stored
    in "continuation_boxes" so the PDF annotator can highlight them too.
    """
    merged = []
    last_par = None  # last paragraph block kept (reset at section headers)

    for block in blocks:
        btype = block.get("type")

        if btype == "section_header":
            last_par = None
            merged.append(block)
            continue

        if btype != "paragraph":
            merged.append(block)
            continue

        content = block.get("content", "")
        if not content.strip():          # existing empty continuation boxes: leave untouched
            merged.append(block)
            continue

        if (last_par is not None
                and _is_unfinished(last_par["content"])
                and _is_continuation(content)):
            prev = last_par["content"].rstrip()
            nxt = content.lstrip()
            # "ANNV-" + "450 ..." -> "ANNV-450 ..." ; otherwise join with a space
            last_par["content"] = prev + nxt if prev.endswith("-") else prev + " " + nxt
            last_par.setdefault("continuation_boxes", []).append(
                {"page": block.get("page"), "box": block.get("box")}
            )
            print(f"MERGED split paragraph (p{last_par.get('page')} + p{block.get('page')}): "
                  f"...{prev[-40:]} | {nxt[:40]}...")
            continue

        block = dict(block)              # don't mutate the input
        merged.append(block)
        last_par = block

    return merged

def load_blocks(json_path: str) -> list:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f).get("blocks", [])


def to_dict(result) -> dict:
    if is_dataclass(result):
        return asdict(result)
    return dict(result)


def has_relevant_findings(flags: dict) -> bool:
    """Returns True if any flag is exactly True (boolean)."""
    found = any(v is True for v in flags.values())
    return found

def remove_sections(blocks):
    sections_to_remove = {
        "introduction",
        "background",
        "acknowledgments",
        "authors contribution",
        "citation",
        "competing interests"
    }

    filtered_blocks = []
    inside_removed_section = False

    for block in blocks:
        block_type = block.get("type")
        content = block.get("content", "").strip()
        content_lower = content.lower()

        if block_type == "section_header":

            # Enter a section to remove
            if any(section in content_lower for section in sections_to_remove):
                print(f"ENTERING REMOVED SECTION: {content}")
                inside_removed_section = True
                continue  # don't keep the header

            # Any new section header ends the removed section
            if inside_removed_section:
                print(f"LEAVING REMOVED SECTION -> {content}")
                inside_removed_section = False

            filtered_blocks.append(block)
            continue

        # Skip everything inside the removed section
        if inside_removed_section:
            print(f"SKIPPING: {content[:80]}...")
            continue

        filtered_blocks.append(block)

    return filtered_blocks


def process_blocks(blocks, regex_extractor, llm_extractor, min_text_length, skip_llm):
    results = []
    errors = 0
    paragraph_index = 0

    for block_index, block in enumerate(blocks):
        block_type = block.get("type")
        content = block.get("content", "").strip()

        #  Headers 
        if block_type == "section_header":
            print(f"\n[BLOCK {block_index}] HEADER: '{content[:80]}...'")
            results.append({
                "type": "header",
                "content": content,
                "page": block.get("page"),
            })
            continue

        if block_type != "paragraph":
            print(f"\n[BLOCK {block_index}] SKIPPING block type='{block_type}'")
            continue

        if len(content) < min_text_length:
            print(f"\n[BLOCK {block_index}] SKIPPING paragraph too short ({len(content)} chars < {min_text_length})")
            continue

        paragraph_index += 1
        print(f"\n{'='*60}")
        print(f"[BLOCK {block_index}] PARAGRAPH #{paragraph_index} | page={block.get('page')} | {len(content)} chars")
        print(f"  TEXT PREVIEW: '{content[:120]}...'")

        try:
            # Step 1: Regex extraction
            print(f"\n  [STEP 1] Running REGEX extractor...")
            regex_flags = to_dict(regex_extractor.extract(content))

            if not llm_extractor:
                print(f"  [STEP 2] LLM extractor not loaded, skipping.")
            elif skip_llm:
                print(f"  [STEP 2] --skip_llm flag is set, skipping LLM.")
            
            else: 
                
                print(regex_flags.get("lipid_composition"))  
                if regex_flags.get("lipid_composition"):
                    print("\n  [STEP LIPID COMPOSITION] Running lipid composition confirmation...")
                    
                    try:
                        updated_lipids = llm_extractor.confirm_lipid_composition_info(text=content, data_response=regex_flags)
                        regex_flags["lipid_composition"] = updated_lipids
                        print(f"  [STEP LIPID COMPOSITION] Lipid composition returned: {updated_lipids}")
                    except Exception as e:
                        print(f"  [STEP LIPID COMPOSITION] !! LIPID ERROR: {type(e).__name__}: {e}")
                        traceback.print_exc()
                
                print(regex_flags.get("lipid_composition_ratio"))
                lipids = regex_flags.get("lipid_composition") or []

                ratio_info = regex_flags.get("lipid_composition_ratio") or {}
                ratios = ratio_info.get("ratios") or []

                if len(lipids) != len(ratios):
                    regex_flags.pop("lipid_composition_ratio", None)
                    

                if (regex_flags.get("lipid_composition") and not regex_flags.get("lipid_composition_ratio")):
                    print("\n  [STEP 5b] Lipid composition found but no ratios — running ratio extractor...")
                    try:
                        extracted_ratios = llm_extractor.extract_lipid_ratio_info(text=content, data_response=regex_flags)
                        if extracted_ratios:
                            regex_flags["lipid_composition_ratio"] = extracted_ratios
                            print(f"  [STEP 5b] Ratios extracted: {extracted_ratios}")
                        else:
                            print("  [STEP 5b] No ratios found in text.")
                    except Exception as e:
                        print(f"  [STEP 5b] !! RATIO EXTRACTION ERROR: {type(e).__name__}: {e}")
                        traceback.print_exc()

                if regex_flags.get("lipid_composition_ratio"):
                    print("\n  [STEP LIPID COMPOSITION RATIO] Running lipid ratio units classification...")
                    try:
                        updated_ratios = llm_extractor.extract_lipid_ratio_units_info(text=content, data_response=regex_flags)
                        regex_flags["lipid_composition_ratio"] = updated_ratios
                    except Exception as e:
                        print(f"  [STEP LIPID COMPOSITION RATIO] !! RATIO UNITS ERROR: {type(e).__name__}: {e}")
                        traceback.print_exc()



                print(regex_flags.get("formulations"))
                if regex_flags.get("formulations"):
                    print("\n  [STEP FORMULATION] Running formulations detection...")
                    try:
                        updated_formulations = llm_extractor.extract_formulation_registry(text=content, data_response=regex_flags)
                        regex_flags["formulations"] = updated_formulations
                    except Exception as e:
                        print(f"  [STEP FORMULATION] !! ERROR FORMULATIONS: {type(e).__name__}: {e}")
                        traceback.print_exc()

                if regex_flags.get("formulations"):
                    print("\n  [STEP CARGOS] Running formulations detection...")
                    try:
                        add_cargos_by_llm = llm_extractor.check_cargo(data_response=regex_flags)
                        regex_flags["cargos LLM"] = add_cargos_by_llm
                    except Exception as e:
                        print(f"  [STEP FORMULATION] !! ERROR FORMULATIONS: {type(e).__name__}: {e}")
                        traceback.print_exc()
            

            true_flags = {k: v for k, v in regex_flags.items() if v is True}
            print(f"  [STEP 1] Done. TRUE flags: {true_flags if true_flags else 'NONE'}")

            # Step 2: Decide if LLM should run
            has_findings = has_relevant_findings(regex_flags)
            print(f"\n  [STEP 2] has_findings={has_findings} | skip_llm={skip_llm} | llm_extractor={'LOADED' if llm_extractor else 'NOT LOADED'}")

            llm_values = None

            if not llm_extractor:
                print(f"  [STEP 2] LLM extractor not loaded, skipping.")
            elif skip_llm:
                print(f"  [STEP 2] --skip_llm flag is set, skipping LLM.")
            elif not has_findings:
                print(f"  [STEP 2] No True flags found, skipping LLM.")
            else:
                print(f"\n  [STEP 3] Running LLM extractor...")
                print(f"  [STEP 3] Setting _extracted_flags: {true_flags}")
                llm_extractor._extracted_flags = regex_flags

                try:
                    # First extraction 
                    llm_values = llm_extractor.extract_text_info(content)

                    print(f"  [STEP 3] LLM returned: {llm_values}")

                    # Second extraction: schedule info 
                    schedule_values = None

                    if llm_values and llm_values.get("dose_group"):

                        print("\n  [STEP 4] Running schedule extractor...")

                        try:
                            schedule_values = llm_extractor.extract_schedule_info(
                                text=content,
                                data_response=llm_values
                            )

                            print(f"  [STEP 4] Schedule returned: {schedule_values}")

                        except Exception as e:
                            print(f"  [STEP 4] !! SCHEDULE ERROR: {type(e).__name__}: {e}")
                            traceback.print_exc()

                    if llm_values and llm_values.get("size_nm"):
                        print("\n  [STEP 5] Running load-status confirmation...")
                        try:
                            updated_sizes = llm_extractor.extract_load_status_info(text=content, data_response=llm_values)
                            llm_values["size_nm"] = updated_sizes
                        except Exception as e:
                            print(f"  [STEP 5] !! LOAD STATUS ERROR: {type(e).__name__}: {e}")
                            traceback.print_exc()


                except Exception as e:
                    print(f"  [STEP 3] !! LLM ERROR: {type(e).__name__}: {e}")
                    traceback.print_exc()
                    llm_values = None

            # Step 3: Merge 
            extraction = {**regex_flags, **(llm_values or {})}
            print(f"\n  [MERGE] Final extraction keys with non-null values: "
                  f"{[k for k, v in extraction.items() if v is not None and v is not False and v != []]}")

            results.append({
                "type": "paragraph",
                "page": block.get("page"),
                "text": content,
                "continuation_boxes": block.get("continuation_boxes", []),
                "extraction": extraction,
            })

        except Exception as e:
            errors += 1
            print(f"  !! OUTER ERROR on paragraph #{paragraph_index}: {type(e).__name__}: {e}")
            traceback.print_exc()
            logger.error(f"Failed to process paragraph: {e}")

    print(f"\n{'='*60}")
    print(f"PROCESSING COMPLETE: {paragraph_index} paragraphs, {errors} errors")
    return results, errors


@click.command()
@click.argument("paragraph_json", type=click.Path(exists=True))
@click.argument("output_file_path", type=click.Path())
@click.option("--prompt_template_path", default=None)
@click.option("--prompt_schema_path", default=None)
@click.option("--llm_model_name", default=None)
@click.option("--llm_model_parameters_path", default=None)
@click.option("--skip_llm", is_flag=True, default=False)
@click.option("--min_text_length", type=int, default=200, show_default=True)
def nanoparticles2data(
    paragraph_json,
    output_file_path,
    prompt_template_path,
    prompt_schema_path,
    llm_model_name,
    llm_model_parameters_path,
    skip_llm,
    min_text_length,
):
    torch.cuda.empty_cache()
    start = time.time()
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    print("STARTING nanoparticles2data")
    print(f"  Input:            {paragraph_json}")
    print(f"  Output:           {output_file_path}")
    print(f"  skip_llm:         {skip_llm}")
    print(f"  min_text_length:  {min_text_length}")
    print(f"  llm_model_name:   {llm_model_name}")

    print("\nLOADING FILE...")
    blocks = load_blocks(paragraph_json)
    print(f"  Loaded {len(blocks)} blocks total.")

    # Regex extractor 
    print("\nLOADING REGEX EXTRACTOR...")
    regex_extractor = NanoparticleExtractor()
    print("  Regex extractor ready.")

    # LLM extractor 
    llm_extractor = None
    if skip_llm:
        print("\nLLM EXTRACTOR: skipped (--skip_llm flag)")
    else:
        print("\nLOADING LLM EXTRACTOR...")
        try:
            name = llm_model_name.split("/")[-1]
            prompt_template_path = TEMPLATE_REGISTRY[name]

            print(f'template used: {prompt_template_path}')           

            llm_extractor = NanoparticlesExtractorParagraph(
                prompt_template_path=prompt_template_path,
                prompt_schema_path=prompt_schema_path,
                llm_model_name=llm_model_name,
                llm_model_parameters_path=llm_model_parameters_path,
            )
            print("  LLM extractor ready.")
        except Exception as e:
            print(f"  !! LLM extractor FAILED to load: {type(e).__name__}: {e}")
            traceback.print_exc()
            logger.warning(f"LLM disabled: {e}")

    # Process 
    print("\nSTARTING BLOCK PROCESSING...")
    results, error_count = process_blocks(
        blocks=merge_split_paragraphs(remove_sections(blocks)),
        blocks=remove_sections(blocks),
        regex_extractor=regex_extractor,
        llm_extractor=llm_extractor,
        min_text_length=min_text_length,
        skip_llm=skip_llm,
    )

    # Save
    output = {
        "source": {
            "file": os.path.abspath(paragraph_json),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_paragraphs": sum(1 for r in results if r["type"] == "paragraph"),
        },
        "results": results,
    }

    print(f"\nSAVING OUTPUT to {output_file_path}...")
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nDONE")
    print(f"  Paragraphs processed: {output['source']['total_paragraphs']}")
    print(f"  Errors:               {error_count}")
    print(f"  Time (min):           {(time.time() - start) / 60:.2f}")


def main():
    nanoparticles2data()


if __name__ == "__main__":
    main()