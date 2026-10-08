import json
from typing import Any, Dict

import click

from xlmexlab.evaluator import Evaluator
from xlmexlab.parser import KeywordSearching


@click.command()
@click.argument("reference_dataset_path", type=str)
@click.argument("evaluated_dataset_path", type=str)
@click.argument("output_file_path", type=str)
def eval_molar_ratio(
    reference_dataset_path: str,
    evaluated_dataset_path: str,
    output_file_path: str,
) -> None:

    evaluator = Evaluator(reference_dataset_path=reference_dataset_path)
    evaluator.model_post_init(None)
    results: Dict[str, Any] = evaluator.evaluate_molar_ratio(evaluated_dataset_path)
    results_json = json.dumps(results, indent=4)
    with open(output_file_path, "w") as f:
        f.write(results_json)


def main():
    eval_molar_ratio()


if __name__ == "__main__":
    main()
