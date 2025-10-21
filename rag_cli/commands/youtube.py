"""
YouTube commands for RAG CLI.
"""

import sys
import json as json_module
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
    @click.option('--json', is_flag=True, help='Output results as JSON')
    def test(query: str, max_results: int, json: bool):
        """Test YouTube API connectivity by searching for videos."""
        try:
            # Check if YouTube API key is configured
            youtube_config = config_service.get_youtube_config()
            if not youtube_config.get('api_key'):
                click.echo("Error: YouTube API key not configured.", err=True)
                click.echo("Please set YOUTUBE_API_KEY in your environment variables or .env file.", err=True)
                click.echo("You can use the .env.youtube template file as a reference.", err=True)
                sys.exit(1)

            # Perform search
            results = search_youtube_videos(query, max_results)

            if json:
                click.echo(json_module.dumps(results, indent=2, ensure_ascii=False))
            else:
                click.echo(f"Testing YouTube API with query: '{query}'")
                click.echo(f"Searching for up to {max_results} results...")
                click.echo()

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

    @youtube.command()
    @click.argument('video_input', required=True)
    @click.option('--json', is_flag=True, help='Output video information as JSON')
    def info(video_input: str, json: bool):
        """Get detailed information about a YouTube video."""
        try:
            # Check if YouTube API key is configured
            youtube_config = config_service.get_youtube_config()
            if not youtube_config.get('api_key'):
                click.echo("Error: YouTube API key not configured.", err=True)
                click.echo("Please set YOUTUBE_API_KEY in your environment variables or .env file.", err=True)
                click.echo("You can use the .env.youtube template file as a reference.", err=True)
                sys.exit(1)

            # Get video info
            from Common.services.youtube_service import get_youtube_video_info
            video_info = get_youtube_video_info(video_input)

            if json:
                click.echo(json_module.dumps(video_info, indent=2, ensure_ascii=False))
            else:
                click.echo("Video Information:")
                click.echo("=" * 50)
                click.echo(f"Title: {video_info['title']}")
                click.echo(f"Channel: {video_info['channel_title']}")
                click.echo(f"Video ID: {video_info['video_id']}")
                click.echo(f"Published: {video_info['published_at']}")
                click.echo(f"Duration: {video_info['duration']}")
                click.echo(f"Views: {video_info['view_count']:,}")
                click.echo(f"Likes: {video_info['like_count']:,}")
                click.echo(f"Comments: {video_info['comment_count']:,}")
                if video_info.get('thumbnail_url'):
                    click.echo(f"Thumbnail: {video_info['thumbnail_url']}")
                if video_info.get('tags'):
                    click.echo(f"Tags: {', '.join(video_info['tags'][:5])}" + ("..." if len(video_info['tags']) > 5 else ""))
                click.echo()
                click.echo(f"Description: {video_info['description'][:200]}" + ("..." if len(video_info['description']) > 200 else ""))

                click.echo("✅ Video information retrieved successfully!")

        except Exception as e:
            click.echo(f"❌ Failed to get video information: {e}", err=True)
            click.echo()
            click.echo("Troubleshooting tips:")
            click.echo("1. Check your internet connection")
            click.echo("2. Verify the video ID or URL is correct")
            click.echo("3. Verify YOUTUBE_API_KEY is correct")
            click.echo("4. Ensure you have enabled YouTube Data API v3 in Google Cloud Console")
            click.echo("5. Check your API quota hasn't been exceeded")
            sys.exit(1)

    @youtube.command()
    @click.argument('channel_input', required=True)
    @click.option('--json', is_flag=True, help='Output channel information as JSON')
    def channel(channel_input: str, json: bool):
        """Get detailed information about a YouTube channel."""
        try:
            # Check if YouTube API key is configured
            youtube_config = config_service.get_youtube_config()
            if not youtube_config.get('api_key'):
                click.echo("Error: YouTube API key not configured.", err=True)
                click.echo("Please set YOUTUBE_API_KEY in your environment variables or .env file.", err=True)
                click.echo("You can use the .env.youtube template file as a reference.", err=True)
                sys.exit(1)

            # Get channel info
            from Common.services.youtube_service import get_youtube_channel_info
            channel_info = get_youtube_channel_info(channel_input)

            if json:
                click.echo(json_module.dumps(channel_info, indent=2, ensure_ascii=False))
            else:
                click.echo("Channel Information:")
                click.echo("=" * 50)
                click.echo(f"Title: {channel_info['title']}")
                click.echo(f"Channel ID: {channel_info['channel_id']}")
                click.echo(f"Subscriber Count: {channel_info['subscriber_count']:,}")
                click.echo(f"Video Count: {channel_info['video_count']:,}")
                click.echo(f"Total Views: {channel_info['view_count']:,}")
                if channel_info.get('thumbnail_url'):
                    click.echo(f"Thumbnail: {channel_info['thumbnail_url']}")
                click.echo()
                click.echo(f"Description: {channel_info['description'][:300]}" + ("..." if len(channel_info['description']) > 300 else ""))

                click.echo("✅ Channel information retrieved successfully!")

        except Exception as e:
            click.echo(f"❌ Failed to get channel information: {e}", err=True)
            click.echo()
            click.echo("Troubleshooting tips:")
            click.echo("1. Check your internet connection")
            click.echo("2. Verify the channel ID or URL is correct")
            click.echo("3. Verify YOUTUBE_API_KEY is correct")
            click.echo("4. Ensure you have enabled YouTube Data API v3 in Google Cloud Console")
            click.echo("5. Check your API quota hasn't been exceeded")
            sys.exit(1)

    return youtube