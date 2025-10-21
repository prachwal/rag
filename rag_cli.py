#!/usr/bin/env python3
"""
RAG Application CLI

Command-line interface for the RAG (Retrieval-Augmented Generation) application.
Provides commands for configuration management and YouTube API testing.
"""

import os
import sys
import click
from pathlib import Path
from typing import Optional

# Add Common to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from Common.services.config_service import config_service
    from Common.services.youtube_service import search_youtube_videos
except ImportError as e:
    click.echo(f"Error importing services: {e}", err=True)
    click.echo("Make sure you're running from the project root directory.", err=True)
    sys.exit(1)


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """RAG Application CLI - Command-line interface for RAG operations."""
    pass


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


@cli.group()
def youtube():
    """YouTube API related commands."""
    pass


@youtube.command()
@click.option('--query', default='python programming', help='Search query for YouTube videos')
@click.option('--max-results', default=5, type=int, help='Maximum number of results to return')
def test(query: str, max_results: int):
    """Test YouTube API connectivity by searching for videos."""
    try:
        # Check if YouTube API key is configured
        youtube_config = config_service.get_youtube_config()
        if not youtube_config.get('api_key'):
            click.echo("Error: YouTube API key not configured.", err=True)
            click.echo("Please set YOUTUBE_API_KEY in your environment variables or .env file.", err=True)
            click.echo("You can use the .env.youtube template file as a reference.", err=True)
            sys.exit(1)

        click.echo(f"Testing YouTube API with query: '{query}'")
        click.echo(f"Searching for up to {max_results} results...")
        click.echo()

        # Perform search
        results = search_youtube_videos(query, max_results)

        if not results:
            click.echo("No videos found for the given query.")
            return

        click.echo(f"Found {len(results)} video(s):")
        click.echo("-" * 60)

        for i, video in enumerate(results, 1):
            click.echo(f"{i}. {video['title']}")
            click.echo(f"   Channel: {video['channel_title']}")
            click.echo(f"   Video ID: {video['video_id']}")
            click.echo(f"   Published: {video['published_at']}")
            if video.get('thumbnail_url'):
                click.echo(f"   Thumbnail: {video['thumbnail_url']}")
            click.echo()

        click.echo("✅ YouTube API test successful!")

    except Exception as e:
        click.echo(f"❌ YouTube API test failed: {e}", err=True)
        click.echo()
        click.echo("Troubleshooting tips:")
        click.echo("1. Check your internet connection")
        click.echo("2. Verify YOUTUBE_API_KEY is correct")
        click.echo("3. Ensure you have enabled YouTube Data API v3 in Google Cloud Console")
        click.echo("4. Check your API quota hasn't been exceeded")
        sys.exit(1)


def main():
    """Main entry point for the CLI application."""
    try:
        cli()
    except KeyboardInterrupt:
        click.echo("\nOperation cancelled by user.", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"An unexpected error occurred: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    main()