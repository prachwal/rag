#!/usr/bin/env python3
"""
RAG Application CLI

Main CLI module that registers all commands.
"""

import sys
import click
from pathlib import Path

# Add Common to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    # Import command modules
    from .commands import help, config, youtube, neon, huggingface
except ImportError as e:
    click.echo(f"Error importing command modules: {e}", err=True)
    click.echo("Make sure you're running from the project root directory.", err=True)
    sys.exit(1)


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """RAG Application CLI - Command-line interface for RAG operations."""
    pass


def main():
    """Main entry point for the CLI application."""
    try:
        # Register all commands
        help.register_help_command(cli)
        config.register_config_command(cli)
        youtube.register_youtube_commands(cli)
        cli.add_command(neon.neon)
        cli.add_command(huggingface.huggingface)

        # Run the CLI
        cli()
    except KeyboardInterrupt:
        click.echo("\nOperation cancelled by user.", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"An unexpected error occurred: {e}", err=True)
        sys.exit(1)