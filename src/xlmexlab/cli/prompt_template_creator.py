from pathlib import Path
from typing import Any, Dict, List

import click
from langchain.prompts import PromptTemplate


@click.command()
@click.argument("file_path", type=click.Path(path_type=Path, exists=True))
@click.argument("template", type=str)
@click.argument("input_variables", type=List[str])
@click.option(
    "--dictionary_type",
    default={},
    help="Dictionary containing the type of the input_variables",
)
def create_prompt_template(
    file_path: Path,
    template: str,
    input_variables: List[str],
    dictionary_type: Dict[str, Any],
) -> None:
    """Create a langchain prompt json template file

    Args:
        file_path: path to save the file
        template: langchain prompt template
        input_variables: variables to be considered in the prompt
        dictionary_type: dicitonary containign the type of each variable
    """
    prompt: Any = PromptTemplate(
        input_variables=input_variables, template=template, input_types=dictionary_type
    )
    prompt.save(file_path)


def main():
    create_prompt_template()


if __name__ == "__main__":
    main()
