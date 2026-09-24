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
def eval_classifier(
    reference_dataset_path: str,
    evaluated_dataset_path: str,
    output_file_path: str,
) -> None:

    evaluator = Evaluator(reference_dataset_path=reference_dataset_path)
    results: Dict[str, Any] = evaluator.evaluate_classifier(evaluated_dataset_path)
    df = pd.DataFrame(results, index=[0])
    df.to_excel(
        output_file_path,
        index=False,
    )


def main():
    eval_classifier()


if __name__ == "__main__":
    main()
