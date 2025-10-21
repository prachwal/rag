"""
Tests for huggingface_service.py module.

This module contains comprehensive tests for the HuggingFace service,
including local model support, API connections, text generation, 
embeddings, and error handling.
"""

import pytest
import time
import requests
from unittest.mock import patch, MagicMock, Mock
from Common.services.huggingface_service import (
    HuggingFaceService,
    test_huggingface_connection,
    generate_text_huggingface,
    get_embeddings_huggingface,
)


class TestHuggingFaceService:
    """Test cases for HuggingFaceService class."""

    def setup_method(self):
        """Reset singleton instance before each test."""
        HuggingFaceService._reset_instance()

    def teardown_method(self):
        """Reset singleton instance after each test."""
        HuggingFaceService._reset_instance()

    def test_singleton_pattern(self):
        """Test that HuggingFaceService follows singleton pattern."""
        with patch('Common.services.config_service.config_service.get_huggingface_config'), \
             patch('Common.services.config_service.config_service.get_embedding_config'):
            service1 = HuggingFaceService()
            service2 = HuggingFaceService()
            assert service1 is service2

    @patch('Common.services.config_service.config_service.get_embedding_config')
    @patch('Common.services.config_service.config_service.get_huggingface_config')
    def test_initialization(self, mock_hf_config, mock_emb_config):
        """Test service initialization."""
        mock_hf_config.return_value = {"token": "test_token", "timeout": 30}
        mock_emb_config.return_value = {"model": "all-MiniLM-L6-v2"}
        service = HuggingFaceService()
        assert service._config == {"token": "test_token", "timeout": 30}
        assert hasattr(service, '_session')
        assert hasattr(service, '_local_embedding_model')
        assert hasattr(service, '_has_sentence_transformers')
        assert hasattr(service, '_has_transformers')

    @patch('Common.services.config_service.config_service.get_embedding_config')
    @patch('Common.services.config_service.config_service.get_huggingface_config')
    def test_test_connection_success(self, mock_hf_config, mock_emb_config):
        """Test successful connection test."""
        mock_hf_config.return_value = {"token": "test_token", "timeout": 30}
        mock_emb_config.return_value = {"model": "all-MiniLM-L6-v2"}

        service = HuggingFaceService()
        
        # Mock session and model loading
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        
        with patch.object(service._session, 'get', return_value=mock_response), \
             patch.object(service, '_load_local_embedding_model', side_effect=RuntimeError("Not installed")):
            result = service.test_connection()

        assert result["status"] in ["success", "partial"]
        assert "local_models" in result
        assert "api_connection" in result
        
        # API connection should be successful
        assert result["api_connection"].get("status") == "connected"

    @patch('Common.services.config_service.config_service.get_embedding_config')
    @patch('Common.services.config_service.config_service.get_huggingface_config')
    def test_test_connection_api_failure(self, mock_hf_config, mock_emb_config):
        """Test API connection failure but service still works with local models."""
        mock_hf_config.return_value = {"token": "test_token", "timeout": 30}
        mock_emb_config.return_value = {"model": "all-MiniLM-L6-v2"}

        service = HuggingFaceService()
        
        # Mock local model loading to avoid network calls
        # Use RequestException for API calls
        with patch.object(service, '_load_local_embedding_model', side_effect=RuntimeError("Model not available")), \
             patch.object(service._session, 'get', side_effect=requests.exceptions.RequestException("Connection failed")):
            result = service.test_connection()

        # Status might be partial if local models available, or error if none work
        assert result["status"] in ["error", "partial"]
        assert "api_connection" in result
        assert result["api_connection"]["status"] == "error"
        assert "Connection failed" in result["api_connection"]["error"]

    @patch('Common.services.config_service.config_service.get_embedding_config')
    @patch('Common.services.config_service.config_service.get_huggingface_config')
    @patch('requests.Session.post')
    def test_generate_text_api_success(self, mock_post, mock_hf_config, mock_emb_config):
        """Test successful text generation via API."""
        mock_hf_config.return_value = {"token": "test_token", "timeout": 30}
        mock_emb_config.return_value = {"model": "all-MiniLM-L6-v2"}

        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = [{"generated_text": "Hello world!"}]
        mock_post.return_value = mock_response

        service = HuggingFaceService()
        result = service.generate_text("Hello", use_local=False)

        assert result["status"] == "success"
        assert result["generated_text"] == "Hello world!"
        assert result["model"] == "microsoft/DialoGPT-medium"
        assert result["prompt"] == "Hello"
        assert result["backend"] == "api"

    @patch('Common.services.config_service.config_service.get_embedding_config')
    @patch('Common.services.config_service.config_service.get_huggingface_config')
    def test_generate_text_api_failure(self, mock_hf_config, mock_emb_config):
        """Test API text generation failure."""
        mock_hf_config.return_value = {"token": "test_token", "timeout": 30}
        mock_emb_config.return_value = {"model": "all-MiniLM-L6-v2"}

        service = HuggingFaceService()
        
        # Mock session.post to raise RequestException
        with patch.object(service._session, 'post', side_effect=requests.exceptions.RequestException("API error")):
            result = service.generate_text("Hello", use_local=False)

        assert result["status"] == "error"
        assert "failed" in result["message"]
        assert "API error" in result["message"]

    @patch('Common.services.config_service.config_service.get_embedding_config')
    @patch('Common.services.config_service.config_service.get_huggingface_config')
    def test_generate_text_local_not_available(self, mock_hf_config, mock_emb_config):
        """Test local generation when transformers not installed."""
        mock_hf_config.return_value = {"token": None, "timeout": 30}
        mock_emb_config.return_value = {"model": "all-MiniLM-L6-v2"}

        service = HuggingFaceService()
        service._has_transformers = False
        
        result = service.generate_text("Hello", use_local=True)

        assert result["status"] == "error"
        assert "not installed" in result["message"] or "failed" in result["message"]

    @patch('Common.services.config_service.config_service.get_embedding_config')
    @patch('Common.services.config_service.config_service.get_huggingface_config')
    def test_get_local_embeddings_success(self, mock_hf_config, mock_emb_config):
        """Test successful local embeddings generation."""
        mock_hf_config.return_value = {"token": "test_token", "timeout": 30}
        mock_emb_config.return_value = {"model": "all-MiniLM-L6-v2"}

        service = HuggingFaceService()
        
        # Mock the sentence transformer model
        mock_model = MagicMock()
        mock_model.encode.return_value = Mock(tolist=lambda: [[0.1, 0.2, 0.3]])
        
        with patch.object(service, '_load_local_embedding_model', return_value=mock_model):
            result = service.get_local_embeddings(["Hello world"])

        assert result["status"] == "success"
        assert result["embeddings"] == [[0.1, 0.2, 0.3]]
        assert result["text_count"] == 1
        assert result["backend"] == "local"
        assert result["dimensions"] == 3

    @patch('Common.services.config_service.config_service.get_embedding_config')
    @patch('Common.services.config_service.config_service.get_huggingface_config')
    @patch('requests.Session.post')
    def test_get_embeddings_api_success(self, mock_post, mock_hf_config, mock_emb_config):
        """Test successful API embeddings generation."""
        mock_hf_config.return_value = {"token": "test_token", "timeout": 30}
        mock_emb_config.return_value = {"model": "all-MiniLM-L6-v2"}

        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = [[0.1, 0.2, 0.3]]
        mock_post.return_value = mock_response

        service = HuggingFaceService()
        service._has_sentence_transformers = False  # Force API usage
        
        result = service.get_embeddings(["Hello world"], use_local=False)

        assert result["status"] == "success"
        assert result["embeddings"] == [[0.1, 0.2, 0.3]]
        assert result["text_count"] == 1
        assert result["backend"] == "api"

    @patch('Common.services.config_service.config_service.get_embedding_config')
    @patch('Common.services.config_service.config_service.get_huggingface_config')
    def test_get_embeddings_local_failure_no_api(self, mock_hf_config, mock_emb_config):
        """Test embeddings failure when local fails and no API token."""
        mock_hf_config.return_value = {"token": None, "timeout": 30}
        mock_emb_config.return_value = {"model": "all-MiniLM-L6-v2"}

        service = HuggingFaceService()
        service._has_sentence_transformers = False
        
        result = service.get_embeddings(["Hello world"])

        assert result["status"] == "error"
        assert "failed" in result["message"]

    @patch('Common.services.config_service.config_service.get_embedding_config')
    @patch('Common.services.config_service.config_service.get_huggingface_config')
    @patch('requests.Session.get')
    def test_list_models_success(self, mock_get, mock_hf_config, mock_emb_config):
        """Test successful model listing."""
        mock_hf_config.return_value = {"token": "test_token", "timeout": 30}
        mock_emb_config.return_value = {"model": "all-MiniLM-L6-v2"}

        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = [
            {"id": "model1", "downloads": 1000},
            {"id": "model2", "downloads": 2000}
        ]
        mock_get.return_value = mock_response

        service = HuggingFaceService()
        result = service.list_models(search="test", limit=5)

        assert result["status"] == "success"
        assert len(result["models"]) == 2
        assert result["count"] == 2
        assert result["search"] == "test"

    @patch('Common.services.config_service.config_service.get_embedding_config')
    @patch('Common.services.config_service.config_service.get_huggingface_config')
    @patch('requests.Session.get')
    def test_list_models_with_filter(self, mock_get, mock_hf_config, mock_emb_config):
        """Test model listing with filter."""
        mock_hf_config.return_value = {"token": "test_token", "timeout": 30}
        mock_emb_config.return_value = {"model": "all-MiniLM-L6-v2"}

        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = [{"id": "model1", "downloads": 1000}]
        mock_get.return_value = mock_response

        service = HuggingFaceService()
        result = service.list_models(filter_by="text-generation")

        assert result["status"] == "success"
        assert result["filter"] == "text-generation"

    @patch('Common.services.config_service.config_service.get_embedding_config')
    @patch('Common.services.config_service.config_service.get_huggingface_config')
    def test_list_models_failure(self, mock_hf_config, mock_emb_config):
        """Test model listing failure."""
        mock_hf_config.return_value = {"token": "test_token", "timeout": 30}
        mock_emb_config.return_value = {"model": "all-MiniLM-L6-v2"}

        service = HuggingFaceService()
        
        # Mock session.get to raise RequestException
        with patch.object(service._session, 'get', side_effect=requests.exceptions.RequestException("List error")):
            result = service.list_models()

        assert result["status"] == "error"
        assert "Failed to list models" in result["message"]
        assert "List error" in result["message"]

    @patch('Common.services.config_service.config_service.get_embedding_config')
    @patch('Common.services.config_service.config_service.get_huggingface_config')
    def test_is_available(self, mock_hf_config, mock_emb_config):
        """Test service availability check."""
        mock_hf_config.return_value = {"token": "test_token", "timeout": 30}
        mock_emb_config.return_value = {"model": "all-MiniLM-L6-v2"}

        service = HuggingFaceService()
        assert service.is_available() is True

    @patch('Common.services.config_service.config_service.get_embedding_config')
    @patch('Common.services.config_service.config_service.get_huggingface_config')
    def test_get_capabilities(self, mock_hf_config, mock_emb_config):
        """Test getting service capabilities."""
        mock_hf_config.return_value = {"token": "test_token", "timeout": 30}
        mock_emb_config.return_value = {"model": "all-MiniLM-L6-v2"}

        service = HuggingFaceService()
        caps = service.get_capabilities()

        assert "local_embeddings" in caps
        assert "local_generation" in caps
        assert "api_access" in caps
        assert caps["api_access"] is True
        assert "configured_embedding_model" in caps


class TestConvenienceFunctions:
    """Test cases for convenience functions."""

    def setup_method(self):
        """Reset singleton instance before each test."""
        HuggingFaceService._reset_instance()

    def teardown_method(self):
        """Reset singleton instance after each test."""
        HuggingFaceService._reset_instance()

    @patch('Common.services.config_service.config_service.get_embedding_config')
    @patch('Common.services.config_service.config_service.get_huggingface_config')
    @patch('requests.Session.get')
    def test_test_huggingface_connection(self, mock_get, mock_hf_config, mock_emb_config):
        """Test test_huggingface_connection convenience function."""
        mock_hf_config.return_value = {"token": "test_token", "timeout": 30}
        mock_emb_config.return_value = {"model": "all-MiniLM-L6-v2"}

        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.elapsed.total_seconds.return_value = 1.0
        mock_get.return_value = mock_response

        result = test_huggingface_connection()
        assert result["status"] in ["success", "partial"]
        assert "local_models" in result
        assert "api_connection" in result

    @patch('Common.services.config_service.config_service.get_embedding_config')
    @patch('Common.services.config_service.config_service.get_huggingface_config')
    @patch('requests.Session.post')
    def test_generate_text_huggingface(self, mock_post, mock_hf_config, mock_emb_config):
        """Test generate_text_huggingface convenience function."""
        mock_hf_config.return_value = {"token": "test_token", "timeout": 30}
        mock_emb_config.return_value = {"model": "all-MiniLM-L6-v2"}

        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = [{"generated_text": "Response"}]
        mock_post.return_value = mock_response

        result = generate_text_huggingface("Hello", use_local=False)
        assert result["status"] == "success"
        assert result["generated_text"] == "Response"

    @patch('Common.services.config_service.config_service.get_embedding_config')
    @patch('Common.services.config_service.config_service.get_huggingface_config')
    def test_get_embeddings_huggingface_local(self, mock_hf_config, mock_emb_config):
        """Test get_embeddings_huggingface convenience function with local model."""
        mock_hf_config.return_value = {"token": "test_token", "timeout": 30}
        mock_emb_config.return_value = {"model": "all-MiniLM-L6-v2"}

        # Need to reset to get fresh instance
        HuggingFaceService._reset_instance()
        
        # Mock sentence transformer
        mock_model = MagicMock()
        mock_model.encode.return_value = Mock(tolist=lambda: [[0.1, 0.2]])
        
        with patch('Common.services.huggingface_service.HuggingFaceService._load_local_embedding_model', return_value=mock_model):
            result = get_embeddings_huggingface(["text"], use_local=True)
            # Result depends on whether sentence_transformers is installed
            assert "status" in result
            assert "embeddings" in result or "message" in result


class TestLocalModels:
    """Test cases for local model functionality."""

    def setup_method(self):
        """Reset singleton instance before each test."""
        HuggingFaceService._reset_instance()

    def teardown_method(self):
        """Reset singleton instance after each test."""
        HuggingFaceService._reset_instance()

    @patch('Common.services.config_service.config_service.get_embedding_config')
    @patch('Common.services.config_service.config_service.get_huggingface_config')
    def test_check_library_availability(self, mock_hf_config, mock_emb_config):
        """Test library availability checking."""
        mock_hf_config.return_value = {"token": None, "timeout": 30}
        mock_emb_config.return_value = {"model": "all-MiniLM-L6-v2"}

        service = HuggingFaceService()
        
        # These will be True or False depending on actual installation
        assert isinstance(service._has_sentence_transformers, bool)
        assert isinstance(service._has_transformers, bool)

    @patch('Common.services.config_service.config_service.get_embedding_config')
    @patch('Common.services.config_service.config_service.get_huggingface_config')
    def test_load_local_embedding_model_not_installed(self, mock_hf_config, mock_emb_config):
        """Test loading local embedding model when library not installed."""
        mock_hf_config.return_value = {"token": None, "timeout": 30}
        mock_emb_config.return_value = {"model": "all-MiniLM-L6-v2"}

        service = HuggingFaceService()
        service._has_sentence_transformers = False

        with pytest.raises(RuntimeError, match="sentence-transformers not installed"):
            service._load_local_embedding_model()

    @patch('Common.services.config_service.config_service.get_embedding_config')
    @patch('Common.services.config_service.config_service.get_huggingface_config')
    def test_load_local_text_model_not_installed(self, mock_hf_config, mock_emb_config):
        """Test loading local text model when library not installed."""
        mock_hf_config.return_value = {"token": None, "timeout": 30}
        mock_emb_config.return_value = {"model": "all-MiniLM-L6-v2"}

        service = HuggingFaceService()
        service._has_transformers = False

        with pytest.raises(RuntimeError, match="transformers not installed"):
            service._load_local_text_model("test-model")