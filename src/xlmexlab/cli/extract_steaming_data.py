import os
import time
from typing import Any, Dict, List, Optional

import click
import torch

from xlmexlab.extractor import SamplesExtractorFromText, SteamingDataExtractor
from xlmexlab.prompt import TEMPLATE_REGISTRY
from xlmexlab.randomization import seed_everything


@click.command()
@click.argument("text_file_path", type=str)
@click.argument("output_file_path", type=str)
@click.option(
    "--llm_model_name",
    default=None,
    help="Name of the LLM used to get the actions",
)
@click.option(
    "--llm_model_parameters_path",
    default=None,
    help="Parameters of the LLM used to get the actions",
)
@click.option(
    "--prompt_template_path",
    default=None,
    help="Path to the file containing the structure of the action prompt",
)
def steaming_extraction(
    text_file_path: str,
    output_file_path: str,
    llm_model_name: str,
    llm_model_parameters_path: Optional[str],
    prompt_template_path: Optional[str],
):
    torch.cuda.empty_cache()
    start_time = time.time()
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    if prompt_template_path is None:
        try:
            name = llm_model_name.split("/")[-1]
            prompt_template_path = TEMPLATE_REGISTRY[name]
        except KeyError:
            pass
    steaming_extractor: SteamingDataExtractor = SteamingDataExtractor(
        llm_model_name=llm_model_name,
        llm_model_parameters_path=llm_model_parameters_path,
        prompt_template_path=prompt_template_path,
    )
    sample_extractor: SamplesExtractorFromText = SamplesExtractorFromText()
    with open(text_file_path, "r") as f:
        text_lines: List[str] = f.readlines()
    size = len(text_lines)
    count = 1
    if os.path.isfile(output_file_path):
        os.remove(output_file_path)
    for text in text_lines:
        print(f"text processed: {count}/{size}")
        sample_list: List[Dict[str, Any]] = sample_extractor.retrieve_samples_from_text(
            text
        )
        results = []
        for sample in sample_list:
            steaming_data: Dict[str, Any] = steaming_extractor.extract(
                sample["procedure"]
            )
            results.append(steaming_data)
        with open(output_file_path, "a") as f:
            f.write(str(results) + "\n")
        count = count + 1
    print(f"{(time.time() - start_time) / 60} minutes")


def main():
    steaming_extraction()


if __name__ == "__main__":
    main()
