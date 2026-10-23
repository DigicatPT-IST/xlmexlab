import os
import json
import time
import click
from typing import Optional, List, Dict, Any
from importlib import resources as importlib_resources
from xlmexlab.extractor import TableExtractor, List2Headers, Table2Blocks
from xlmexlab.prompt import TEMPLATE_REGISTRY


def parse_header_rows_from_response(response: str, convert_to_index: bool = True) -> List[int]:
    """
    Parse the VLM response to extract header row numbers and convert to indices.
    
    Args:
        response: VLM response containing row numbers
        convert_to_index: If True, converts row numbers (1-based) to indices (0-based)
    
    Examples (with convert_to_index=True):
    - "rows 1, 2, 3" → [0, 1, 2]
    - "The headers are in rows 1 and 2" → [0, 1]
    - '{"header_rows": [1, 2]}' → [0, 1]
    """
    import re
    
    header_rows = []
    
    # Try JSON parsing first
    try:
        data = json.loads(response)
        if "header_rows" in data:
            header_rows = data["header_rows"]
        elif "headers" in data:
            header_rows = data["headers"]
        elif "rows" in data:
            header_rows = data["rows"]
    except:
        pass
    
    # If JSON parsing didn't work, extract all numbers from the response
    if not header_rows:
        numbers = re.findall(r'\b\d+\b', response)
        header_rows = [int(n) for n in numbers]
    
    # Convert row numbers (1-based) to indices (0-based)
    if convert_to_index and header_rows:
        header_rows = [row - 1 for row in header_rows]
        # Remove negative indices (in case VLM returned 0)
        header_rows = [idx for idx in header_rows if idx >= 0]
    
    return header_rows


@click.command()
@click.argument("image_folder", type=str)
@click.argument("output_file_path", type=str)
@click.option("--table_type", default="All", help="Type of table to process")
@click.option("--prompt_template_path", default=None, help="Path to prompt template for stage 1")
@click.option("--prompt_schema_path", default=None, help="Path to prompt schema for stage 1")
@click.option("--vlm_model_name", default=None, help="Name of VLM model")
@click.option("--vlm_model_parameters_path", default=None, help="Path to VLM model parameters")
@click.option("--enable_header_refinement", is_flag=True, default=False, 
              help="Enable stage 2: use VLM to refine header detection")
@click.option("--header_prompt_template_path", default=None, 
              help="Path to prompt template for header refinement (stage 2)")
@click.option("--header_prompt_schema_path", default=None,
              help="Path to prompt schema for header refinement (stage 2)")
def extract_tables_chain(
    image_folder: str,
    output_file_path: str,
    table_type: str,
    prompt_template_path: Optional[str],
    prompt_schema_path: Optional[str],
    vlm_model_name: Optional[str],
    vlm_model_parameters_path: Optional[str],
    enable_header_refinement: bool,
    header_prompt_template_path: Optional[str],
    header_prompt_schema_path: Optional[str],
):
    """
    Two-stage table extraction pipeline (per image):
    
    For each image:
      Stage 1: Extract table data
      Stage 2 (optional): Refine header detection using VLM
    """
    start_time = time.time()
    
    print("\n" + "="*60)
    if enable_header_refinement:
        print("MODE: Two-stage extraction (Table + Header refinement)")
    else:
        print("MODE: Single-stage extraction (Table only)")
    print("="*60 + "\n")
    
    # Initialize the table extractor
    if prompt_template_path is None and vlm_model_name:
        try:
            name = vlm_model_name.split("/")[-1]
            prompt_template_path = TEMPLATE_REGISTRY[name]
        except KeyError:
            pass
    
    extractor = TableExtractor(
        table_type=table_type,
        prompt_template_path=prompt_template_path,
        prompt_schema_path=prompt_schema_path,
        vlm_model_name=vlm_model_name,
        vlm_model_parameters_path=vlm_model_parameters_path,
    )
    
    # If header refinement is enabled, prepare the header extractor with SHARED model
    header_extractor = None
    if enable_header_refinement:
        print("[INFO] Header refinement enabled - preparing header detector...")
        
        # Create List2Headers that will reuse the same VLM model
        # Temporarily disable model_post_init to prevent loading model twice
        original_post_init = List2Headers.model_post_init
        List2Headers.model_post_init = lambda self, context: None
        
        try:
            header_extractor = List2Headers(
                table_type=table_type,
                prompt_template_path=prompt_template_path,
                prompt_schema_path=header_prompt_schema_path,
                vlm_model_name=vlm_model_name,
                vlm_model_parameters_path=vlm_model_parameters_path
            )
            
            # Manually initialize the prompt
            if header_prompt_schema_path is None:
                schema_path = str(
                    importlib_resources.files("xlmexlab")
                    / "resources/schemas"
                    / "table_extraction_schema.json"
                )
            else:
                schema_path = header_prompt_schema_path

            with open(schema_path, "r", encoding="utf-8") as f:
                prompt_dict = json.load(f)

            from xlmexlab.prompt import PromptFormatter
            header_extractor._prompt = PromptFormatter(**prompt_dict)
            header_extractor._prompt.model_post_init(header_prompt_template_path)
            
            # Share the VLM model from the table extractor
            header_extractor._vlm_model = extractor._vlm_model
            header_extractor._condition_parser = None
            
            print("[INFO] Header detector ready (sharing VLM model - no extra memory)")
            
        finally:
            # Restore original model_post_init
            List2Headers.model_post_init = original_post_init
    
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    
    all_results = []
    file_list = sorted(os.listdir(image_folder))
    
    # Process each image
    for file in file_list:
        extension = file.split(".")[-1].lower()
        
        if extension not in {"png", "jpg", "jpeg", "tif", "tiff"}:
            continue
            
        file_path = os.path.join(image_folder, file)
        print(f"\n{'='*60}")
        print(f"Processing: {file}")
        print(f"{'='*60}")
        
        try:
            # ===== STAGE 1: Extract table =====
            print(f"[Stage 1] Extracting table structure...")
            image_file, parsed_output = extractor.extract_table_info(file_path)
            
            # parsed_output is the result from LaTeXTableParser (list of lists)
            table = Table2Blocks(
                page=0,
                name=image_file,
                block=parsed_output
            )
            
            # Initial header/index detection (heuristic)
            table.find_collumn_headers()
            table.find_row_indexes()
            
            result = {
                'image': image_file,
                'page': table.page,
                'name': table.name,
                'block': table.block,
                'type': table.type,
                'collumn_headers': table.collumn_headers,
                'row_indexes': table.row_indexes,
                'number': table.number,
                'legend': table.legend,
                'box': table.box
            }
            
            print(f"[Stage 1] ✓ Extracted {len(parsed_output)} rows")
            print(f"[Stage 1]   Heuristic headers: {table.collumn_headers}")
            
            # STAGE 2: Refine headers (if enabled)
            if enable_header_refinement and header_extractor and parsed_output:
                print(f"[Stage 2] Refining header detection with VLM...")

                try:
                    _, vlm_response = header_extractor.extract_table_info(
                        file_path,
                        extracted_data=parsed_output
                    )

                    print(f"[Stage 2]   VLM response: {vlm_response}")

                    refined_headers = parse_header_rows_from_response(vlm_response)

                    result['collumn_headers'] = refined_headers
                    result['vlm_header_response'] = vlm_response
                    result['heuristic_headers'] = table.collumn_headers

                    print(f"[Stage 2] ✓ Refined headers: {refined_headers}")
                
                except Exception as e:
                    print(f"[Stage 2] ✗ Header refinement failed: {e}")
                    result['header_refinement_error'] = str(e)

            all_results.append(result)

        except Exception as e:
            print(f"[Stage 2] ✗ Header refinement failed: {e}")
                    
            all_results.append({
                "image": file,
                "error": str(e),
                "block": [],
                "collumn_headers": [],
                "row_indexes": []
            })
    
    # Save results
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)
    
    elapsed_time = time.time() - start_time
    print("\n" + "="*60)
    print(f"COMPLETE: Processed {len([f for f in file_list if f.split('.')[-1].lower() in {'png', 'jpg', 'jpeg', 'tif', 'tiff'}])} images in {elapsed_time:.2f} seconds")
    print(f"Results saved to: {output_file_path}")
    print("="*60)


def main():
    extract_tables_chain()


if __name__ == "__main__":
    main()