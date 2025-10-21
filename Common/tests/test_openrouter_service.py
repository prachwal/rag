"""
Tests for OpenRouter service.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import requests

from Common.services.openrouter_service import OpenRouterService, openrouter_service
from Common.services.config_service import ConfigService


class TestOpenRouterService:
    """Test suite for OpenRouterService."""

    def setup_method(self):
        """Reset service instance before each test."""
        OpenRouterService._reset_instance()
        ConfigService._reset_instance()

    def teardown_method(self):
        """Clean up after each test."""
        OpenRouterService._reset_instance()
        ConfigService._reset_instance()

    def test_singleton_pattern(self):
        """Test that OpenRouterService follows singleton pattern."""
        service1 = OpenRouterService()
        service2 = OpenRouterService()
        assert service1 is service2

    def test_is_available_with_api_key(self):
        """Test is_available returns True when API key is configured."""
        with patch.dict('os.environ', {
            'OPENROUTER_API_KEY': 'test_key',
            'SECRET_KEY': 'test_secret_key_minimum_32_characters_long_12345'
        }):
            service = OpenRouterService()
            assert service.is_available() is True

    @patch('requests.Session.get')
    def test_connection_success(self, mock_get):
        """Test successful API connection."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"id": "openai/gpt-3.5-turbo", "name": "GPT-3.5 Turbo"},
                {"id": "anthropic/claude-2", "name": "Claude 2"}
            ]
        }
        mock_get.return_value = mock_response

        with patch.dict('os.environ', {
            'OPENROUTER_API_KEY': 'test_key',
            'OPENROUTER_API_TIMEOUT': '30',
            'SECRET_KEY': 'test_secret_key_minimum_32_characters_long_12345'
        }):
            service = OpenRouterService()
            result = service.test_connection()

            assert result["status"] == "success"
            assert result["authenticated"] is True
            assert result["model_count"] == 2
            assert "response_time" in result

    @patch('requests.Session.get')
    def test_connection_unauthorized(self, mock_get):
        """Test connection with invalid API key."""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response)
        mock_get.return_value = mock_response

        with patch.dict('os.environ', {
            'OPENROUTER_API_KEY': 'invalid_key',
            'SECRET_KEY': 'test_secret_key_minimum_32_characters_long_12345'
        }):
            service = OpenRouterService()
            result = service.test_connection()

            assert result["status"] == "error"
            assert "Invalid API key" in result["message"]
            assert result["status_code"] == 401

    @patch('requests.Session.get')
    def test_get_available_models_success(self, mock_get):
        """Test getting available models."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {
                    "id": "openai/gpt-3.5-turbo",
                    "name": "GPT-3.5 Turbo",
                    "description": "Fast and efficient",
                    "context_length": 4096,
                    "pricing": {"prompt": "0.001", "completion": "0.002"},
                    "top_provider": {"name": "OpenAI"}
                },
                {
                    "id": "anthropic/claude-2",
                    "name": "Claude 2",
                    "description": "Large context window",
                    "context_length": 100000,
                    "pricing": {"prompt": "0.01", "completion": "0.03"}
                }
            ]
        }
        mock_get.return_value = mock_response

        with patch.dict('os.environ', {
            'OPENROUTER_API_KEY': 'test_key',
            'SECRET_KEY': 'test_secret_key_minimum_32_characters_long_12345'
        }):
            service = OpenRouterService()
            result = service.get_available_models()

            assert result["status"] == "success"
            assert result["count"] == 2
            assert len(result["models"]) == 2
            assert result["models"][0]["id"] == "openai/gpt-3.5-turbo"
            assert result["models"][0]["context_length"] == 4096

    @patch('requests.Session.post')
    def test_generate_text_success(self, mock_post):
        """Test successful text generation."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "gen-123",
            "model": "openai/gpt-3.5-turbo",
            "choices": [
                {
                    "message": {"content": "This is a generated response."},
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 15,
                "total_tokens": 25
            }
        }
        mock_post.return_value = mock_response

        with patch.dict('os.environ', {
            'OPENROUTER_API_KEY': 'test_key',
            'SECRET_KEY': 'test_secret_key_minimum_32_characters_long_12345'
        }):
            service = OpenRouterService()
            result = service.generate_text(
                prompt="Hello, world!",
                model="openai/gpt-3.5-turbo"
            )

            assert result["status"] == "success"
            assert result["generated_text"] == "This is a generated response."
            assert result["model"] == "openai/gpt-3.5-turbo"
            assert result["backend"] == "openrouter"
            assert result["usage"]["total_tokens"] == 25

    @patch('requests.Session.post')
    def test_chat_success(self, mock_post):
        """Test successful chat completion."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "chat-123",
            "model": "openai/gpt-3.5-turbo",
            "choices": [
                {
                    "message": {"content": "Hello! How can I help you today?"},
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 10,
                "total_tokens": 30
            }
        }
        mock_post.return_value = mock_response

        with patch.dict('os.environ', {
            'OPENROUTER_API_KEY': 'test_key',
            'SECRET_KEY': 'test_secret_key_minimum_32_characters_long_12345'
        }):
            service = OpenRouterService()
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"}
            ]
            result = service.chat(
                messages=messages,
                model="openai/gpt-3.5-turbo"
            )

            assert result["status"] == "success"
            assert result["response"] == "Hello! How can I help you today?"
            assert result["message_count"] == 2
            assert result["backend"] == "openrouter"
            assert result["usage"]["total_tokens"] == 30

    @patch('requests.Session.post')
    def test_chat_http_error(self, mock_post):
        """Test chat with HTTP error."""
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.json.return_value = {
            "error": {"message": "Rate limit exceeded"}
        }
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response)
        mock_post.return_value = mock_response

        with patch.dict('os.environ', {
            'OPENROUTER_API_KEY': 'test_key',
            'SECRET_KEY': 'test_secret_key_minimum_32_characters_long_12345'
        }):
            service = OpenRouterService()
            messages = [{"role": "user", "content": "Test"}]
            result = service.chat(messages=messages, model="openai/gpt-3.5-turbo")

            assert result["status"] == "error"
            assert "Rate limit exceeded" in result["message"]
            assert result["status_code"] == 429

    @patch('requests.Session.post')
    def test_generate_text_http_error(self, mock_post):
        """Test text generation with HTTP error."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.json.return_value = {
            "error": {"message": "Internal server error"}
        }
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response)
        mock_post.return_value = mock_response

        with patch.dict('os.environ', {
            'OPENROUTER_API_KEY': 'test_key',
            'SECRET_KEY': 'test_secret_key_minimum_32_characters_long_12345'
        }):
            service = OpenRouterService()
            result = service.generate_text(
                prompt="Test",
                model="openai/gpt-3.5-turbo"
            )

            assert result["status"] == "error"
            assert "Internal server error" in result["message"]
            assert result["status_code"] == 500

    def test_module_level_service(self):
        """Test that module-level service instance works."""
        with patch.dict('os.environ', {
            'OPENROUTER_API_KEY': 'test_key',
            'SECRET_KEY': 'test_secret_key_minimum_32_characters_long_12345'
        }):
            # Access via module-level instance
            assert openrouter_service.is_available() is True
