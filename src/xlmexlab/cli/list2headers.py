import os
import time
from typing import List, Optional
import json
import click

from xlmexlab.extractor import List2Headers
from xlmexlab.prompt import TEMPLATE_REGISTRY


@click.command()
@click.argument("image_folder", type=str)
@click.argument("image_json_path", type=str)  # JSON mapping: image_name -> extracted_data
@click.argument("output_file_path", type=str)
@click.option("--table_type", default="All", help="Type of table to process")
@click.option("--prompt_template_path", default=None, help="Path to prompt template")
@click.option("--prompt_schema_path", default=None, help="Path to prompt schema")
@click.option("--vlm_model_name", default=None, help="Name of VLM model")
@click.option("--vlm_model_parameters_path", default=None, help="Path to VLM model parameters")
@click.option("--table_schema_path", default="table_extraction_schema.json", help="Base schema JSON path")
def list2headers(
    image_folder: str,
    image_json_path: str,
    output_file_path: str,
    table_type: str,
    prompt_template_path: Optional[str],
    prompt_schema_path: Optional[str],
    vlm_model_name: Optional[str],
    vlm_model_parameters_path: Optional[str],
    table_schema_path: Optional[str],
):
    start_time = time.time()

    # Load JSON list
    with open(image_json_path, "r", encoding="utf-8") as f:
        all_image_data = json.load(f)

    # Initialize List2Headets extractor
    extractor = List2Headers(
        table_type=table_type,
        prompt_template_path=prompt_template_path,
        prompt_schema_path=prompt_schema_path,
        vlm_model_name=vlm_model_name,
        vlm_model_parameters_path=vlm_model_parameters_path
    )
    extractor.model_post_init()

    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    all_results = []

    # Loop over images
    for file in sorted(os.listdir(image_folder)):
        extension = file.split(".")[-1].lower()
        if extension not in {"png", "jpg", "jpeg", "tif", "tiff"}:
            continue

        file_path = os.path.join(image_folder, file)
        print(f"[INFO] Processing {file}")

        # Get 'block' from JSON for this image
        extracted_block = get_block_for_image(all_image_data, file_path)

        try:
            # Extract headers using List2Headets
            image_file, headers = extractor.extract_table_info(
                file_path, extracted_data=extracted_block
            )

            all_results.append({
                "image": file,
                "headers": headers
            })

        except Exception as e:
            print(f"[ERROR] Failed on {file}: {e}")
            all_results.append({
                "image": file,
                "headers": [],
                "error": str(e)
            })

    # Save all results to JSON
    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)

    elapsed_time = time.time() - start_time
    print(f"\n[INFO] Processed {len(all_results)} images in {elapsed_time:.2f} seconds")
    print(f"[INFO] Results saved to: {output_file_path}")

def main():
    list2headers()


if __name__ == "__main__":
    main()
