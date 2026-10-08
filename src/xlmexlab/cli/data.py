import json
from pathlib import Path

import click


@click.command()
@click.argument("file_path", type=click.Path(path_type=Path, exists=True))
@click.option("--number", default=0, help="Number of the entry in the data file")
def visualize_paragraph(file_path: Path, number: int) -> None:
    with open(file_path, "r") as paragraphs:
        paragraphs_list = paragraphs.readlines()
    paragraph = json.loads(paragraphs_list[number])
    click.echo(paragraph["text"])
    senteces_list = paragraph["sentences"]
    for sentence in senteces_list:
        action_list = sentence["actions"]
        for action in action_list:
            click.echo(action)


def main():
    visualize_paragraph()


if __name__ == "__main__":
    main()
