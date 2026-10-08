import json
from typing import Any, Dict

import click
import pandas as pd

from xlmexlab.evaluator import Evaluator
from xlmexlab.parser import KeywordSearching


@click.command()
@click.argument("reference_dataset_path", type=str)
@click.argument("evaluated_dataset_path", type=str)
@click.argument("output_file_path", type=str)
@click.option(
    "--threshold",
    default=0.77,
    help="Minimum threshold value to consider two string similar",
)
def eval_classifier(
    reference_dataset_path: str,
    evaluated_dataset_path: str,
    output_file_path: str,
    threshold: float,
) -> None:

    evaluator = Evaluator(reference_dataset_path=reference_dataset_path)
    evaluation: Dict[str, Any] = evaluator.evaluate_table_extractor(
        evaluated_dataset_path, threshold=threshold
    )
    results: Dict[str, Any] = {}
    for key in evaluation.keys():
        for new_key in evaluation[key].keys():
            result_key = f"{key}_{new_key}"
            results[result_key] = evaluation[key][new_key]
    df = pd.DataFrame(results, index=[0])
    df.to_excel(
        output_file_path,
        index=False,
    )


def main():
    eval_classifier()


if __name__ == "__main__":
    main()
