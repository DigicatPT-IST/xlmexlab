import json
import os
import shutil
import time
from typing import List, Optional

import click

from xlmexlab.extractor_nanoparticles import ImageExtractor
from xlmexlab.prompt import TEMPLATE_REGISTRY


@click.command()
@click.argument("image_folder", type=str)
@click.argument("images_chosen_folder", type=str)
@click.argument("output_file_path", type=str)
@click.option(
    "--prompt_template_path",
    default=None,
    help="Path to the file containing the structure of the prompt",
)
@click.option(
    "--prompt_schema_path",
    default=None,
    help="Path to the file containing the schema of the prompt",
)
@click.option(
    "--vlm_model_name",
    default=None,
    help="Name of the VLM used to process the figures",
)
@click.option(
    "--vlm_model_parameters_path",
    default=None,
    help="Parameters of the VLM used to process the figures, (only in case of vllm inference).",
)
@click.option(
    "--scale",
    default=1.0,
    type=float,
    help="Scale factor to reduce image resolution (e.g., 0.5 for 50%).",
)
def image2datanano(
    image_folder: str,
    images_chosen_folder: str,
    output_file_path: str,
    prompt_template_path: Optional[str],
    prompt_schema_path: Optional[str],
    vlm_model_name: str,
    vlm_model_parameters_path: Optional[str],
    scale: float,
):
    start_time = time.time()

    if prompt_template_path is None:
        try:
            name = vlm_model_name.split("/")[-1]
            prompt_template_path = TEMPLATE_REGISTRY[name]
        except KeyError:
            pass

    extractor = ImageExtractor(
        prompt_template_path=prompt_template_path,
        prompt_schema_path=prompt_schema_path,
        vlm_model_name=vlm_model_name,
        vlm_model_parameters_path=vlm_model_parameters_path,
    )
    os.makedirs(images_chosen_folder, exist_ok=True)
    file_list = os.listdir(image_folder)
    aggregated_data = {}

    for file in file_list:
        print(f"Processing file: {file}")
        extension = file.split(".")[-1]
        print(f"File extension: {extension}")
        if extension in {"png", "jpeg", "tiff"}:
            file_path = f"{image_folder}/{file}"
            print(f"Processing image file: {file_path}")

            try:
                if not extractor.is_graph(file_path, scale=scale):
                    print(f"Skipping {file}: not a graph")
                    continue
                shutil.copy(file_path, os.path.join(images_chosen_folder, file))
                print(f"Copying {file} to chosen folder")

                extracted_data = extractor.extract_series_data(file_path, scale=scale)
                aggregated_data[file] = extracted_data

            except Exception as e:
                print(f"Error processing file {file}: {e}")

    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(aggregated_data, f, indent=2, ensure_ascii=False)
    print(f"Saved results to {output_file_path}")
    elapsed_time = (time.time() - start_time) / 60
    print(f"Process completed in {elapsed_time} minutes")


def main():
    image2datanano()


if __name__ == "__main__":
    main()
