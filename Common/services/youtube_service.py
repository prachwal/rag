"""
YouTube API service for accessing YouTube Data API v3.

This module provides functionality to interact with YouTube API,
including searching for videos, getting video details, and managing API requests.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config_service import config_service


logger = logging.getLogger(__name__)


class YouTubeAPIService:
    """Service for interacting with YouTube Data API v3."""

    BASE_URL = "https://www.googleapis.com/youtube/v3"

    def __init__(self):
        """Initialize YouTube API service with configuration."""
        self.api_key = config_service.settings.youtube_api_key
        self.timeout = config_service.settings.youtube_api_timeout

        if not self.api_key:
            raise ValueError("YouTube API key not configured. Please set YOUTUBE_API_KEY in environment variables.")

        # Setup session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            backoff_factor=1
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make authenticated request to YouTube API."""
        url = f"{self.BASE_URL}/{endpoint}"
        params['key'] = self.api_key

        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()

            # Check for API quota exceeded
            if response.status_code == 403:
                error_data = response.json()
                if 'quotaExceeded' in str(error_data):
                    raise Exception("YouTube API quota exceeded")

            return response.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"YouTube API request failed: {e}")
            raise Exception(f"YouTube API request failed: {e}")

    def search_videos(self, query: str, max_results: int = 10,
                     order: str = "relevance", published_after: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for videos on YouTube.

        Args:
            query: Search query string
            max_results: Maximum number of results (1-50)
            order: Sort order (relevance, date, rating, title, videoCount, viewCount)
            published_after: ISO 8601 date string for filtering videos published after this date

        Returns:
            List of video search results
        """
        if not 1 <= max_results <= 50:
            raise ValueError("max_results must be between 1 and 50")

        params = {
            'part': 'snippet',
            'q': query,
            'type': 'video',
            'maxResults': max_results,
            'order': order
        }

        if published_after:
            params['publishedAfter'] = published_after

        response = self._make_request('search', params)

        videos = []
        for item in response.get('items', []):
            video = {
                'video_id': item['id']['videoId'],
                'title': item['snippet']['title'],
                'description': item['snippet']['description'],
                'channel_title': item['snippet']['channelTitle'],
                'published_at': item['snippet']['publishedAt'],
                'thumbnail_url': item['snippet']['thumbnails'].get('default', {}).get('url'),
                'channel_id': item['snippet']['channelId']
            }
            videos.append(video)

        return videos

    def get_video_details(self, video_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Get detailed information about specific videos.

        Args:
            video_ids: List of YouTube video IDs

        Returns:
            List of video details
        """
        if not video_ids:
            return []

        if len(video_ids) > 50:
            raise ValueError("Cannot request details for more than 50 videos at once")

        params = {
            'part': 'snippet,statistics,contentDetails',
            'id': ','.join(video_ids)
        }

        response = self._make_request('videos', params)

        videos = []
        for item in response.get('items', []):
            video = {
                'video_id': item['id'],
                'title': item['snippet']['title'],
                'description': item['snippet']['description'],
                'channel_title': item['snippet']['channelTitle'],
                'published_at': item['snippet']['publishedAt'],
                'duration': item['contentDetails']['duration'],
                'view_count': int(item['statistics'].get('viewCount', 0)),
                'like_count': int(item['statistics'].get('likeCount', 0)),
                'comment_count': int(item['statistics'].get('commentCount', 0)),
                'thumbnail_url': item['snippet']['thumbnails'].get('default', {}).get('url'),
                'tags': item['snippet'].get('tags', [])
            }
            videos.append(video)

        return videos

    def get_channel_info(self, channel_id: str) -> Dict[str, Any]:
        """
        Get information about a YouTube channel.

        Args:
            channel_id: YouTube channel ID

        Returns:
            Channel information
        """
        params = {
            'part': 'snippet,statistics',
            'id': channel_id
        }

        response = self._make_request('channels', params)

        if not response.get('items'):
            raise ValueError(f"Channel with ID {channel_id} not found")

        channel = response['items'][0]
        return {
            'channel_id': channel['id'],
            'title': channel['snippet']['title'],
            'description': channel['snippet']['description'],
            'subscriber_count': int(channel['statistics'].get('subscriberCount', 0)),
            'video_count': int(channel['statistics'].get('videoCount', 0)),
            'view_count': int(channel['statistics'].get('viewCount', 0)),
            'thumbnail_url': channel['snippet']['thumbnails'].get('default', {}).get('url')
        }

    def search_recent_videos(self, query: str, days_back: int = 7, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search for videos published within the last N days.

        Args:
            query: Search query string
            days_back: Number of days to look back
            max_results: Maximum number of results

        Returns:
            List of recent video search results
        """
        published_after = (datetime.utcnow() - timedelta(days=days_back)).isoformat() + 'Z'
        return self.search_videos(query, max_results, order='date', published_after=published_after)

    def get_playlist_videos(self, playlist_id: str, max_results: int = 50) -> List[Dict[str, Any]]:
        """
        Get videos from a YouTube playlist.

        Args:
            playlist_id: YouTube playlist ID
            max_results: Maximum number of results (1-50)

        Returns:
            List of playlist video items
        """
        if not 1 <= max_results <= 50:
            raise ValueError("max_results must be between 1 and 50")

        params = {
            'part': 'snippet',
            'playlistId': playlist_id,
            'maxResults': max_results
        }

        response = self._make_request('playlistItems', params)

        videos = []
        for item in response.get('items', []):
            video = {
                'video_id': item['snippet']['resourceId']['videoId'],
                'title': item['snippet']['title'],
                'description': item['snippet']['description'],
                'channel_title': item['snippet']['channelTitle'],
                'published_at': item['snippet']['publishedAt'],
                'thumbnail_url': item['snippet']['thumbnails'].get('default', {}).get('url'),
                'position': item['snippet']['position']
            }
            videos.append(video)

        return videos


# Global YouTube service instance - lazy initialization
class _YouTubeServiceLazy:
    """Lazy initialization wrapper for YouTubeAPIService."""

    def __init__(self):
        self._instance = None
        self._instance_error = None

    def __call__(self):
        if self._instance_error:
            raise self._instance_error
        if self._instance is None:
            try:
                self._instance = YouTubeAPIService()
            except Exception as e:
                self._instance_error = e
                raise
        return self._instance

    def __getattr__(self, name):
        return getattr(self(), name)

# Global YouTube service instance
youtube_service = _YouTubeServiceLazy()


def search_youtube_videos(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """
    Convenience function to search YouTube videos.

    Args:
        query: Search query string
        max_results: Maximum number of results

    Returns:
        List of video search results
    """
    return youtube_service.search_videos(query, max_results)


def get_youtube_video_details(video_ids: List[str]) -> List[Dict[str, Any]]:
    """
    Convenience function to get YouTube video details.

    Args:
        video_ids: List of YouTube video IDs

    Returns:
        List of video details
    """
    return youtube_service.get_video_details(video_ids)