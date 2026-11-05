import os
import time
from typing import List, Optional

import click
import torch

from xlmexlab.extractor import SamplesExtractorFromText
from xlmexlab.prompt import TEMPLATE_REGISTRY


@click.command()
@click.argument("text_file_path", type=str)
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
    "--llm_model_name",
    default="meta-llama/Meta-Llama-3-8B-Instruct",
    help="Name of the LLM used to get the actions",
)
@click.option(
    "--llm_model_parameters_path",
    default=None,
    help="Parameters of the LLM used to get the actions",
)
def text2samples(
    text_file_path: str,
    output_file_path: str,
    prompt_template_path: Optional[str],
    prompt_schema_path: Optional[str],
    llm_model_name: str,
    llm_model_parameters_path: Optional[str],
):
    torch.cuda.empty_cache()
    start_time = time.time()
    if prompt_template_path is None:
        try:
            name = llm_model_name.split("/")[-1]
            prompt_template_path = TEMPLATE_REGISTRY[name]
        except KeyError:
            pass
    extractor: SamplesExtractorFromText = SamplesExtractorFromText(
        prompt_template_path=prompt_template_path,
        prompt_schema_path=prompt_schema_path,
        llm_model_name=llm_model_name,
        llm_model_parameters_path=llm_model_parameters_path,
    )
    extractor.model_post_init(None)
    with open(text_file_path, "r") as f:
        text_lines: List[str] = f.readlines()
    if os.path.isfile(output_file_path):
        os.remove(output_file_path)
    size = len(text_lines)
    count = 1
    for text in text_lines:
        print(f"text processed: {count}/{size}")
        results: str = str(extractor.retrieve_samples_from_text(text))
        with open(output_file_path, "a") as f:
            f.write(str(results) + "\n")
        count = count + 1
    print(f"{(time.time() - start_time) / 60} minutes")


def main():
    text2samples()


if __name__ == "__main__":
    main()
