import os
import time
from typing import List, Optional

import click
import importlib_resources
import torch

from xlmexlab.extractor import ParagraphClassifier
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
    help="Path to the file containing the schema of the action prompt",
)
@click.option(
    "--type",
    default="n2_physisorption",
    help="Type of paragraphs you want to find: Options: n2_physorption, ftir_pyridine, desilication_dealumination",
)
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
def paragraph_classifier(
    text_file_path: str,
    output_file_path: str,
    prompt_template_path: Optional[str],
    prompt_schema_path: Optional[str],
    type: str,
    llm_model_name: str,
    llm_model_parameters_path: Optional[str],
):
    torch.cuda.empty_cache()
    start_time = time.time()
    if prompt_schema_path is not None:
        pass
    elif type == "n2_physisorption":
        prompt_schema_path = str(
            importlib_resources.files("xlmexlab")
            / "resources/schemas"
            / "classify_n2_physisorption_schema.json"
        )
    elif type == "ftir_pyridine":
        prompt_schema_path = str(
            importlib_resources.files("xlmexlab")
            / "resources/schemas"
            / "classify_ftiv_pyridine_schema.json"
        )
    elif type == "multi_sample":
        prompt_schema_path = str(
            importlib_resources.files("xlmexlab")
            / "resources/schemas"
            / "classify_multi_sample_schema.json"
        )
    elif type == "desilication_dealumination":
        prompt_schema_path = str(
            importlib_resources.files("xlmexlab")
            / "resources/schemas"
            / "classify_desilication_dealumination_schema.json"
        )
    elif type == "desilication":
        prompt_schema_path = str(
            importlib_resources.files("xlmexlab")
            / "resources/schemas"
            / "classify_desilication_schema.json"
        )
    elif type == "steaming":
        prompt_schema_path = str(
            importlib_resources.files("xlmexlab")
            / "resources/schemas"
            / "classify_steaming_schema.json"
        )
    else:
        raise AttributeError(
            "You need to pass a valid --type or a --prompt_schema_path"
        )
    if prompt_template_path is None:
        try:
            name = llm_model_name.split("/")[-1]
            print(name)
            prompt_template_path = TEMPLATE_REGISTRY[name]
        except KeyError:
            pass
    extractor: ParagraphClassifier = ParagraphClassifier(
        prompt_template_path=prompt_template_path,
        prompt_schema_path=prompt_schema_path,
        llm_model_name=llm_model_name,
        llm_model_parameters_path=llm_model_parameters_path,
    )
    extractor.model_post_init(None)
    with open(text_file_path, "r") as f:
        text_lines: List[str] = f.readlines()
    size = len(text_lines)
    count = 1
    if os.path.isfile(output_file_path):
        os.remove(output_file_path)
    for text in text_lines:
        print(f"text processed: {count}/{size}")
        results: str = str(extractor.classify_paragraph(text))
        with open(output_file_path, "a") as f:
            f.write(str(results) + "\n")
        count = count + 1
    print(f"{(time.time() - start_time) / 60} minutes")


def main():
    paragraph_classifier()


if __name__ == "__main__":
    main()
