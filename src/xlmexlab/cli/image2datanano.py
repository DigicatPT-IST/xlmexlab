import json
import os
import shutil
import time
from typing import List, Optional

import click

from xlmexlab.extractor_nanoparticles import ImageExtractor
from xlmexlab.prompt import TEMPLATE_REGISTRY

IMAGE_EXTS = {"png", "jpg", "jpeg", "tif", "tiff"}
 
 
def _list_images(folder: str):
    """Image files of a folder, sorted (same order on every run)."""
    return sorted(f for f in os.listdir(folder)
                  if os.path.isfile(os.path.join(folder, f)) and f.rsplit(".", 1)[-1].lower() in IMAGE_EXTS)
 
 
def _save(data: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

@click.command()
@click.argument("image_folder", type=str)
@click.argument("images_chosen_folder", type=str)
@click.argument("output_file_path", type=str)
@click.option(
    "--mode",
    type=click.Choice(["all", "select", "extract"]),
    default="all",
    show_default=True,
    help="select = only choose the graphs (copy them to IMAGES_CHOSEN_FOLDER); "
         "extract = only extract data from the images already in IMAGES_CHOSEN_FOLDER; "
         "all = choose + extract.",
)
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
@click.option("--debug_dir", default=None, help="Where to save grid/overlay images")
def image2datanano(
    image_folder: str,
    images_chosen_folder: str,
    output_file_path: str,
    mode: str,
    prompt_template_path: Optional[str],
    prompt_schema_path: Optional[str],
    vlm_model_name: str,
    vlm_model_parameters_path: Optional[str],
    scale: float,
    debug_dir: Optional[str],
):
    """
    \b
    all     : IMAGE_FOLDER -> choose graphs -> IMAGES_CHOSEN_FOLDER -> extract -> OUTPUT_FILE_PATH
    select  : IMAGE_FOLDER -> choose graphs -> IMAGES_CHOSEN_FOLDER   (no extraction, no output file)
    extract : images already in IMAGES_CHOSEN_FOLDER -> extract -> OUTPUT_FILE_PATH
              (IMAGE_FOLDER is not read; is_graph is not called)
    """
    start_time = time.time()

    do_select = mode in ("all", "select")
    do_extract = mode in ("all", "extract")

    if do_extract and not output_file_path:
        raise click.UsageError(f"--mode {mode} needs OUTPUT_FILE_PATH (where the extracted data is saved).")
    if mode == "extract" and not os.path.isdir(images_chosen_folder):
        raise click.UsageError(f"--mode extract: folder '{images_chosen_folder}' does not exist "
                               f"(run --mode select first, or put your chosen images there).")
 
    if prompt_template_path is None:
        try:
            name = vlm_model_name.split("/")[-1]
            prompt_template_path = TEMPLATE_REGISTRY[name]
        except (KeyError, AttributeError):
            pass
    debug_dir = debug_dir or os.path.join(images_chosen_folder, "grounded")
 
    extractor = ImageExtractor(
        prompt_template_path=prompt_template_path,
        prompt_schema_path=prompt_schema_path,
        vlm_model_name=vlm_model_name,
        vlm_model_parameters_path=vlm_model_parameters_path,
    )
 
    #  step 1: choose the graphs
    if do_select:
        os.makedirs(images_chosen_folder, exist_ok=True)
        chosen, skipped, failed = [], [], {}
        for file in _list_images(image_folder):
            file_path = os.path.join(image_folder, file)
            print(f"Checking image file: {file_path}")
            try:
                if not extractor.is_graph(file_path, scale=scale):
                    print(f"Skipping {file}: not a graph")
                    skipped.append(file)
                    continue
                shutil.copy(file_path, os.path.join(images_chosen_folder, file))
                print(f"Copying {file} to chosen folder")
                chosen.append(file)
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"Error checking file {file}: {e}")
                failed[file] = f"{type(e).__name__}: {e}"
        sel_path = os.path.join(images_chosen_folder, "selection.json")
        _save({"chosen": chosen, "not_graph": skipped, "errors": failed}, sel_path)
        print(f"Selection: {len(chosen)} graphs, {len(skipped)} not graphs, {len(failed)} errors -> {sel_path}")
        to_extract = chosen
    else:
        to_extract = _list_images(images_chosen_folder)     # every image you left in the folder
        print(f"Extract only: {len(to_extract)} images in {images_chosen_folder}")
 
    # step 2: extract the data
    if do_extract:
        aggregated_data = {}
        for file in to_extract:
            file_path = os.path.join(images_chosen_folder, file)
            print(f"Extracting data from: {file_path}")
            try:
                aggregated_data[file] = extractor.extract_series_data_grid(file_path, scale=scale,
                                                                          debug_dir=debug_dir)
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"Error processing file {file}: {e}")
            _save(aggregated_data, output_file_path)        # saved after every image: a crash loses nothing
        print(f"Saved results to {output_file_path}")
 
    elapsed_time = (time.time() - start_time) / 60
    print(f"Process completed in {elapsed_time:.1f} minutes")
 
 
def main():
    image2datanano()
 
 
if __name__ == "__main__":
    main()
 