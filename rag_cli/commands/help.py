"""
Help command for RAG CLI.
"""

import click


def register_help_command(cli):
    """Register the help command with the CLI group."""

    @cli.command()
    def help():
        """Show help information."""
        click.echo("""
RAG Application CLI v1.0.0

Available commands:
  help        Show this help message
  config      Display current configuration
  youtube test Test YouTube API connectivity

Examples:
  rag help
  rag config
  rag youtube test

For more information about each command, use: rag <command> --help
""")

    return help