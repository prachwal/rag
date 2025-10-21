"""
Base interface for LLM (Large Language Model) providers.

This module defines abstract base classes that all LLM provider services
should implement to ensure consistent API across different providers.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


class LLMProviderBase(ABC):
    """
    Abstract base class for LLM providers.
    
    All LLM provider services (HuggingFace, OpenRouter, etc.) should
    inherit from this class and implement its methods.
    """

    @abstractmethod
    def test_connection(self) -> Dict[str, Any]:
        """
        Test connection to the LLM provider.
        
        Returns:
            Dict containing:
                - status: 'success', 'error', or 'partial'
                - message: Human-readable status message
                - Additional provider-specific metadata
        """
        pass

    @abstractmethod
    def generate_text(
        self,
        prompt: str,
        model: str,
        max_tokens: int = 250,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input text prompt
            model: Model name/identifier
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 to 2.0)
            **kwargs: Additional provider-specific parameters
        
        Returns:
            Dict containing:
                - status: 'success' or 'error'
                - generated_text: The generated text (if successful)
                - model: Model used
                - prompt: Original prompt
                - error: Error message (if failed)
                - Additional provider-specific metadata
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the provider is available (configured and accessible).
        
        Returns:
            True if provider can be used, False otherwise
        """
        pass

    @abstractmethod
    def get_available_models(self) -> Dict[str, Any]:
        """
        Get list of available models from the provider.
        
        Returns:
            Dict containing:
                - status: 'success' or 'error'
                - models: List of available models
                - Additional metadata
        """
        pass


class EmbeddingProviderBase(ABC):
    """
    Abstract base class for embedding providers.
    
    Providers that support text embeddings should implement this interface.
    """

    @abstractmethod
    def get_embeddings(
        self,
        texts: List[str],
        model: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Get embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            model: Model name (None = use default)
            **kwargs: Additional provider-specific parameters
        
        Returns:
            Dict containing:
                - status: 'success' or 'error'
                - embeddings: List of embedding vectors
                - model: Model used
                - text_count: Number of texts processed
                - dimensions: Embedding dimensions
                - error: Error message (if failed)
        """
        pass


class ChatProviderBase(ABC):
    """
    Abstract base class for chat/conversation providers.
    
    Providers that support chat-style interactions should implement this interface.
    """

    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        max_tokens: int = 250,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a chat response from a conversation history.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
                Example: [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello!"},
                    {"role": "assistant", "content": "Hi! How can I help?"},
                    {"role": "user", "content": "What's the weather?"}
                ]
            model: Model name/identifier
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 to 2.0)
            **kwargs: Additional provider-specific parameters
        
        Returns:
            Dict containing:
                - status: 'success' or 'error'
                - response: The assistant's response (if successful)
                - model: Model used
                - message_count: Number of messages in conversation
                - error: Error message (if failed)
                - Additional provider-specific metadata
        """
        pass
