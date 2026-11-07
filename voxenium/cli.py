"""
Voxenium Command-Line Interface.
"""

from typing import Annotated
import typer
import logging
from pocket_tts.cli import cli_app as base_app
from voxenium import __version__

logger = logging.getLogger("voxenium")

cli_app = typer.Typer(
    help=f"VOXENIUM v{__version__} - Next-Gen Lightweight Real-Time TTS Engine for CPU",
    pretty_exceptions_show_locals=False,
)

# Inherit existing core commands
for cmd in base_app.registered_commands:
    cli_app.registered_commands.append(cmd)

@cli_app.command(name="version")
def version():
    """Print the Voxenium Engine version."""
    print(f"VOXENIUM Engine v{__version__} (CPU Optimized, Sub-100ms Latency)")

def main():
    cli_app()

if __name__ == "__main__":
    main()
