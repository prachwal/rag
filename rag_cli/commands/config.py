"""
Config command for RAG CLI.
"""

import sys
import click

from Common.services.config_service import config_service


def register_config_command(cli):
    """Register the config command with the CLI group."""

    @cli.command()
    def config():
        """Display current application configuration."""
        try:
            settings = config_service.settings

            click.echo("RAG Application Configuration")
            click.echo("=" * 40)

            click.echo(f"Application Name: {settings.app_name}")
            click.echo(f"Version: {settings.app_version}")
            click.echo(f"Debug Mode: {settings.debug}")

            click.echo(f"\nServer Settings:")
            click.echo(f"  Host: {settings.host}")
            click.echo(f"  Port: {settings.port}")

            click.echo(f"\nDatabase:")
            click.echo(f"  URL: {settings.database_url or 'Not configured'}")

            click.echo(f"\nAPI Settings:")
            click.echo(f"  API Key: {'Configured' if settings.api_key else 'Not configured'}")
            click.echo(f"  Timeout: {settings.api_timeout}s")

            click.echo(f"\nYouTube API:")
            click.echo(f"  API Key: {'Configured' if settings.youtube_api_key else 'Not configured'}")
            click.echo(f"  Timeout: {settings.youtube_api_timeout}s")

            click.echo(f"\nLogging:")
            click.echo(f"  Level: {settings.log_level}")
            click.echo(f"  File: {settings.log_file or 'Not configured'}")

            click.echo(f"\nSecurity:")
            click.echo(f"  Secret Key: {'Configured' if settings.secret_key else 'Not configured'}")

        except Exception as e:
            click.echo(f"Error loading configuration: {e}", err=True)
            sys.exit(1)

    return config