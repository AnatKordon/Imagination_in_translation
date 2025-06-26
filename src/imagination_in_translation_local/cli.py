"""Console script for imagination_in_translation_local."""
import imagination_in_translation_local

import typer
from rich.console import Console

app = typer.Typer()
console = Console()


@app.command()
def main():
    """Console script for imagination_in_translation_local."""
    console.print("Replace this message by putting your code into "
               "imagination_in_translation_local.cli.main")
    console.print("See Typer documentation at https://typer.tiangolo.com/")
    


if __name__ == "__main__":
    app()
