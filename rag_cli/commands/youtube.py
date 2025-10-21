"""
YouTube commands for RAG CLI.
"""

import sys
import click

from Common.services.config_service import config_service
from Common.services.youtube_service import search_youtube_videos


def register_youtube_commands(cli):
    """Register YouTube commands with the CLI group."""

    youtube = click.Group('youtube', help='YouTube API related commands.')
    cli.add_command(youtube)

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

    return youtube