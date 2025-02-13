import click

from pipelines.training_pipeline import training_pipeline


# @click.command()
# @click.option(
#     "--type",
#     default="tracking",
#     help="The type of MLflow example to run.",
#     type=click.Choice(["tracking", "registry", "deployment"]),
# )


def main():
    trained = training_pipeline()


if __name__ == "__main__":

    main()
