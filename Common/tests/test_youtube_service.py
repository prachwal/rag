"""
Tests for youtube_service.py module.

This module contains comprehensive tests for the YouTube API service,
including API calls, error handling, and configuration integration.
"""

import pytest
from unittest.mock import patch, MagicMock
from requests.exceptions import RequestException, Timeout

# Import after patching to avoid initialization issues


class TestYouTubeAPIService:
    """Test cases for YouTubeAPIService class."""

    def setup_method(self):
        """Setup before each test."""
        self.api_key = "test_api_key_12345"
        self.timeout = 30

    @patch('Common.services.youtube_service.config_service')
    def test_initialization_success(self, mock_config):
        """Test successful initialization with API key."""
        mock_config.settings.youtube_api_key = self.api_key
        mock_config.settings.youtube_api_timeout = self.timeout

        from Common.services.youtube_service import YouTubeAPIService
        service = YouTubeAPIService()
        assert service.api_key == self.api_key
        assert service.timeout == self.timeout

    @patch('Common.services.youtube_service.config_service')
    def test_initialization_no_api_key(self, mock_config):
        """Test initialization failure without API key."""
        mock_config.settings.youtube_api_key = None

        from Common.services.youtube_service import YouTubeAPIService
        with pytest.raises(ValueError, match="YouTube API key not configured"):
            YouTubeAPIService()

    @patch('Common.services.youtube_service.requests.Session')
    @patch('Common.services.youtube_service.config_service')
    def test_search_videos_success(self, mock_config, mock_session_class):
        """Test successful video search."""
        from Common.services.youtube_service import YouTubeAPIService

        # Mock response data
        mock_response_data = {
            'items': [
                {
                    'id': {'videoId': 'test_video_1'},
                    'snippet': {
                        'title': 'Test Video 1',
                        'description': 'Test description',
                        'channelTitle': 'Test Channel',
                        'publishedAt': '2023-01-01T00:00:00Z',
                        'thumbnails': {'default': {'url': 'http://example.com/thumb.jpg'}},
                        'channelId': 'test_channel_1'
                    }
                }
            ]
        }

        # Setup mocks
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = mock_response_data
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session

        mock_config.settings.youtube_api_key = self.api_key
        mock_config.settings.youtube_api_timeout = self.timeout

        service = YouTubeAPIService()
        results = service.search_videos("test query", max_results=5)

        assert len(results) == 1
        assert results[0]['video_id'] == 'test_video_1'
        assert results[0]['title'] == 'Test Video 1'
        assert results[0]['channel_title'] == 'Test Channel'

        # Verify API call
        mock_session.get.assert_called_once()
        call_args = mock_session.get.call_args
        assert 'q=test+query' in call_args[1]['params']['q']
        assert call_args[1]['params']['maxResults'] == 5

    @patch('Common.services.youtube_service.requests.Session')
    def test_search_videos_invalid_max_results(self, mock_session_class):
        """Test validation of max_results parameter."""
        with patch('Common.services.config_service.config_service') as mock_config:
            mock_config.settings.youtube_api_key = self.api_key
            mock_config.settings.youtube_api_timeout = self.timeout

            service = YouTubeAPIService()

            with pytest.raises(ValueError, match="max_results must be between 1 and 50"):
                service.search_videos("test", max_results=0)

            with pytest.raises(ValueError, match="max_results must be between 1 and 50"):
                service.search_videos("test", max_results=51)

    @patch('Common.services.youtube_service.requests.Session')
    def test_get_video_details_success(self, mock_session_class):
        """Test successful video details retrieval."""
        mock_response_data = {
            'items': [
                {
                    'id': 'test_video_1',
                    'snippet': {
                        'title': 'Test Video',
                        'description': 'Test description',
                        'channelTitle': 'Test Channel',
                        'publishedAt': '2023-01-01T00:00:00Z',
                        'thumbnails': {'default': {'url': 'http://example.com/thumb.jpg'}},
                        'tags': ['tag1', 'tag2']
                    },
                    'statistics': {
                        'viewCount': '1000',
                        'likeCount': '100',
                        'commentCount': '10'
                    },
                    'contentDetails': {
                        'duration': 'PT10M30S'
                    }
                }
            ]
        }

        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = mock_response_data
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session

        with patch('Common.services.config_service.config_service') as mock_config:
            mock_config.settings.youtube_api_key = self.api_key
            mock_config.settings.youtube_api_timeout = self.timeout

            service = YouTubeAPIService()
            results = service.get_video_details(['test_video_1'])

            assert len(results) == 1
            video = results[0]
            assert video['video_id'] == 'test_video_1'
            assert video['title'] == 'Test Video'
            assert video['view_count'] == 1000
            assert video['like_count'] == 100
            assert video['duration'] == 'PT10M30S'
            assert video['tags'] == ['tag1', 'tag2']

    @patch('Common.services.youtube_service.requests.Session')
    def test_get_video_details_too_many_ids(self, mock_session_class):
        """Test validation of video IDs count."""
        with patch('Common.services.config_service.config_service') as mock_config:
            mock_config.settings.youtube_api_key = self.api_key
            mock_config.settings.youtube_api_timeout = self.timeout

            service = YouTubeAPIService()

            with pytest.raises(ValueError, match="Cannot request details for more than 50 videos"):
                service.get_video_details(['video_id'] * 51)

    @patch('Common.services.youtube_service.requests.Session')
    def test_api_request_failure(self, mock_session_class):
        """Test handling of API request failures."""
        mock_session = MagicMock()
        mock_session.get.side_effect = RequestException("Network error")
        mock_session_class.return_value = mock_session

        with patch('Common.services.config_service.config_service') as mock_config:
            mock_config.settings.youtube_api_key = self.api_key
            mock_config.settings.youtube_api_timeout = self.timeout

            service = YouTubeAPIService()

            with pytest.raises(Exception, match="YouTube API request failed"):
                service.search_videos("test query")

    @patch('Common.services.youtube_service.requests.Session')
    def test_api_quota_exceeded(self, mock_session_class):
        """Test handling of quota exceeded errors."""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = None
        mock_response.status_code = 403
        mock_response.json.return_value = {'error': {'errors': [{'reason': 'quotaExceeded'}]}}

        mock_session = MagicMock()
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session

        with patch('Common.services.config_service.config_service') as mock_config:
            mock_config.settings.youtube_api_key = self.api_key
            mock_config.settings.youtube_api_timeout = self.timeout

            service = YouTubeAPIService()

            with pytest.raises(Exception, match="YouTube API quota exceeded"):
                service.search_videos("test query")

    @patch('Common.services.youtube_service.requests.Session')
    def test_get_channel_info_success(self, mock_session_class):
        """Test successful channel info retrieval."""
        mock_response_data = {
            'items': [
                {
                    'id': 'test_channel_1',
                    'snippet': {
                        'title': 'Test Channel',
                        'description': 'Test channel description',
                        'thumbnails': {'default': {'url': 'http://example.com/channel_thumb.jpg'}}
                    },
                    'statistics': {
                        'subscriberCount': '10000',
                        'videoCount': '500',
                        'viewCount': '1000000'
                    }
                }
            ]
        }

        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = mock_response_data
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session

        with patch('Common.services.config_service.config_service') as mock_config:
            mock_config.settings.youtube_api_key = self.api_key
            mock_config.settings.youtube_api_timeout = self.timeout

            service = YouTubeAPIService()
            channel_info = service.get_channel_info('test_channel_1')

            assert channel_info['channel_id'] == 'test_channel_1'
            assert channel_info['title'] == 'Test Channel'
            assert channel_info['subscriber_count'] == 10000
            assert channel_info['video_count'] == 500

    @patch('Common.services.youtube_service.requests.Session')
    def test_get_channel_info_not_found(self, mock_session_class):
        """Test channel not found error."""
        mock_response_data = {'items': []}

        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = mock_response_data
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session

        with patch('Common.services.config_service.config_service') as mock_config:
            mock_config.settings.youtube_api_key = self.api_key
            mock_config.settings.youtube_api_timeout = self.timeout

            service = YouTubeAPIService()

            with pytest.raises(ValueError, match="Channel with ID test_channel_1 not found"):
                service.get_channel_info('test_channel_1')

    @patch('Common.services.youtube_service.requests.Session')
    def test_search_recent_videos(self, mock_session_class):
        """Test searching for recent videos."""
        mock_response_data = {'items': []}

        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = mock_response_data
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session

        with patch('Common.services.config_service.config_service') as mock_config:
            mock_config.settings.youtube_api_key = self.api_key
            mock_config.settings.youtube_api_timeout = self.timeout

            service = YouTubeAPIService()
            results = service.search_recent_videos("test query", days_back=3)

            # Verify publishedAfter parameter is set
            call_args = mock_session.get.call_args
            assert 'publishedAfter' in call_args[1]['params']
            assert call_args[1]['params']['order'] == 'date'


class TestConvenienceFunctions:
    """Test cases for convenience functions."""

    def test_search_youtube_videos(self):
        """Test search_youtube_videos convenience function."""
        with patch('Common.services.youtube_service.youtube_service') as mock_service:
            mock_service.search_videos.return_value = [{'video_id': 'test'}]

            results = search_youtube_videos("test query", max_results=5)

            mock_service.search_videos.assert_called_once_with("test query", 5)
            assert results == [{'video_id': 'test'}]

    def test_get_youtube_video_details(self):
        """Test get_youtube_video_details convenience function."""
        with patch('Common.services.youtube_service.youtube_service') as mock_service:
            mock_service.get_video_details.return_value = [{'video_id': 'test'}]

            results = get_youtube_video_details(['video_id'])

            mock_service.get_video_details.assert_called_once_with(['video_id'])
            assert results == [{'video_id': 'test'}]


class TestIntegration:
    """Integration tests for YouTube service."""

    @patch('Common.services.youtube_service.requests.Session')
    def test_full_search_workflow(self, mock_session_class):
        """Test complete search workflow."""
        mock_response_data = {
            'items': [
                {
                    'id': {'videoId': 'video_1'},
                    'snippet': {
                        'title': 'Integration Test Video',
                        'description': 'Test video for integration',
                        'channelTitle': 'Integration Channel',
                        'publishedAt': '2023-12-01T10:00:00Z',
                        'thumbnails': {'default': {'url': 'http://example.com/thumb.jpg'}},
                        'channelId': 'channel_1'
                    }
                }
            ]
        }

        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = mock_response_data
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session

        with patch('Common.services.config_service.config_service') as mock_config:
            mock_config.settings.youtube_api_key = "integration_api_key"
            mock_config.settings.youtube_api_timeout = 60

            # Test through convenience function
            results = search_youtube_videos("integration test", max_results=1)

            assert len(results) == 1
            assert results[0]['title'] == 'Integration Test Video'
            assert results[0]['channel_title'] == 'Integration Channel'