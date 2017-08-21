from __future__ import division
import click
from . import app


@app.command()
@click.argument('text')
def echo(text):
    click.echo(text)
