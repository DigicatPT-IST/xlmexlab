import time
from typing import Any, Dict, List, Optional

import click
import torch

from xlmexlab.extractor import MolarRatioExtractorFromText


@click.command()
@click.argument("text_file_path", type=str)
@click.argument("output_file_path", type=str)
@click.option(
    "--valid_chemicals_path",
    default=None,
    help="Path to the file containing the chemicals to consider",
)
def text2molar_ratio(
    text_file_path: str,
    output_file_path: str,
    valid_chemicals_path: Optional[str],
):
    torch.cuda.empty_cache()
    start_time = time.time()
    extractor: MolarRatioExtractorFromText = MolarRatioExtractorFromText(
        chemicals_path=valid_chemicals_path
    )
    extractor.model_post_init(None)
    with open(text_file_path, "r") as f:
        text_lines: List[str] = f.readlines()
    size = len(text_lines)
    count = 1
    for text in text_lines:
        print(f"text processed: {count}/{size}")
        molar_ratio: Dict[Any, Any] = extractor.extract_molar_ratio(text)
        with open(output_file_path, "a") as f:
            f.write(str(molar_ratio) + "\n")
        count = count + 1
    print(f"{(time.time() - start_time) / 60} minutes")


def main():
    text2molar_ratio()


if __name__ == "__main__":
    main()
