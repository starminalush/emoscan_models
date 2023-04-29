import click


click.command()
click.argument('dataset-path', type=click.Path())