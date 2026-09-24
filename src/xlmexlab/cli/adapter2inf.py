import click
import json
import os
from PIL import Image
from xlmexlab.inference_w_adapter import ModelWithAdapter

from typing import Any, Dict, List, Optional

@click.command()
@click.argument("image_folder", type=str)
@click.argument("output_file", type=str)
@click.option("--base_model_path", required=True, help="Path or HF name of base model")
@click.option("--imports_config_path", required=True, help="Path to imports.json")
@click.option("--adapter_path", default=None, help="Path to LoRA adapter (optional)")

def run_inference(
    image_folder: str,
    output_file : str,
    base_model_path: str,
    imports_config_path: str ,
    adapter_path: Optional[str],
):
    model = ModelWithAdapter(
        base_model_path=base_model_path,
        imports_config_path=imports_config_path,
        adapter_path=adapter_path,
    )

    aggregated_data = {}

    for file in os.listdir(image_folder):
        if file.lower().endswith((".png", ".jpg", ".jpeg", ".tiff")):
            path = os.path.join(image_folder, file)
            image = Image.open(path).convert("RGB")
            print(image)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": "Return the data in the following format:\n{\n  \"<serie_name>\": {\n    \"<x_axis_label>\": [x1, x2, ...],\n    \"<y_axis_label>\": [y1, y2, ...]\n  },\n  ...\n}\nReplace '<...>' with the actual text and values from the image. And please do not add any additional information. \n\n**If any data point is unreadable or missing, use 'N/A' in place of that value.**\n\n**Important:** Copy series names and the axis names *exactly* as shown in the graph, and *convert only superscripts/subscripts to plain text*."},
                    ],
                }
            ]

            parsed_data = model.generate(messages)
            print (parsed_data)
            for key, subdict in parsed_data.items():
                if key in aggregated_data:
                    aggregated_data[key].update(subdict)
                else:
                    aggregated_data[key] = subdict


    with open(output_file, "w") as f:
        json.dump(aggregated_data, f, indent=4)

    print(f"Results saved to {output_file}")

def main():
    run_inference()

if __name__ == "__main__":
    main()