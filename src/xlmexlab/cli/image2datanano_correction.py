import json
import os
import shutil
import time
from typing import List, Optional, Dict, Any

import click

from xlmexlab.extractor_nanoparticles import SeriesVerifier
from xlmexlab.prompt import TEMPLATE_REGISTRY

@click.command()
@click.argument("input_json_path", type=str)
@click.argument("images_folder", type=str)
@click.argument("output_json_path", type=str)
@click.option("--vlm_model_name", default="Qwen/Qwen3.5-9B", help="VLM used for verification.")
@click.option("--vlm_model_parameters_path", default=None, help="Model parameters path (vllm inference).")
@click.option("--prompt_template_path", default=None, help="Prompt template path.")
@click.option("--scale", default=1.0, type=float, help="Scale factor for image resolution.")
def verify_series_cli(
    input_json_path: str,
    images_folder: str,
    output_json_path: str,
    vlm_model_name: str,
    vlm_model_parameters_path: Optional[str],
    prompt_template_path: Optional[str],
    scale: float,
):
    start_time = time.time()

    if prompt_template_path is None:
        try:
            name = vlm_model_name.split("/")[-1]
            prompt_template_path = TEMPLATE_REGISTRY[name]
        except KeyError:
            pass

    with open(input_json_path, "r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)

    verifier = SeriesVerifier(
        vlm_model_name=vlm_model_name,
        vlm_model_parameters_path=vlm_model_parameters_path,
        prompt_template_path=prompt_template_path,
    )

    for image_name, image_data in data.items():
        image_path = os.path.join(images_folder, image_name)
        if not os.path.exists(image_path):
            print(f"Skipping {image_name}: image not found in {images_folder}")
            continue

        print(f"\nVerifying series in: {image_name}")

        reserved_keys = {"x_axis", "y_axis", "flagged_for_review", "flag_reason"}
        series_names = [k for k in image_data.keys() if k not in reserved_keys]

        for series_name in series_names:
            extracted_points = image_data[series_name]
            try:
                corrected = verifier.verify_series(
                    image_path=image_path,
                    series_name=series_name,
                    extracted_points=extracted_points,
                    scale=scale,
                )
                image_data[series_name] = corrected
            except Exception as e:
                print(f"Error verifying series '{series_name}' in {image_name}: {e}")

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved verified results to {output_json_path}")

    elapsed_time = (time.time() - start_time) / 60
    print(f"Verification completed in {elapsed_time} minutes")


def main():
    verify_series_cli()


if __name__ == "__main__":
    main()
