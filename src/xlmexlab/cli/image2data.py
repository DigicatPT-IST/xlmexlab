import json
import os
import time
from typing import List, Optional

import click

from xlmexlab.extractor import ImageExtractor
from xlmexlab.prompt import TEMPLATE_REGISTRY


@click.command()
@click.argument("image_folder", type=str)
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
def image2data(
    image_folder: str,
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
                # Extract image info with the image name as a key in the parsed data
                parsed_data = extractor.extract_image_info(file_path)
                print(f"Parsed data for {parsed_data}")

                # Update aggregated_data using a nested dictionary merge logic
                for key, subdict in parsed_data.items():
                    if key in aggregated_data:
                        aggregated_data[key].update(subdict)
                    else:
                        aggregated_data[key] = subdict

            except Exception as e:
                print(f"Error processing file {file}: {e}")

    print("Writing aggregated data to output file...")
    with open(output_file_path, "w") as output_file:
        json.dump(aggregated_data, output_file, indent=4)
    print(f"Aggregated data written to {output_file_path}")

    elapsed_time = (time.time() - start_time) / 60
    print(f"Process completed in {elapsed_time} minutes")


def main():
    image2data()


if __name__ == "__main__":
    main()
