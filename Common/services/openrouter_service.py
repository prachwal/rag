"""
OpenRouter service for interacting with OpenRouter API.

OpenRouter provides unified access to multiple LLM providers through a single API.
Supports various models from OpenAI, Anthropic, Google, Meta, and more.

Documentation: https://openrouter.ai/docs
"""

import logging
import time
from typing import Dict, Any, List, Optional

import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from Common.services.config_service import config_service
from Common.services.llm_provider_base import LLMProviderBase, ChatProviderBase


class OpenRouterService(LLMProviderBase, ChatProviderBase):
    """
    Service for interacting with OpenRouter API.
    
    OpenRouter provides a unified interface to multiple LLM providers.
    Supports text generation, chat completions, and various models.
    
    Features:
    - Unified API for multiple LLM providers
    - Automatic model routing
    - Streaming support
    - Usage tracking
    """

    _instance: Optional['OpenRouterService'] = None

    def __new__(cls) -> 'OpenRouterService':
        """Singleton pattern to ensure single service instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the OpenRouter service."""
        if not hasattr(self, '_initialized'):
            self._config = config_service.get_openrouter_config()
            self._session = requests.Session()
            self._base_url = "https://openrouter.ai/api/v1"
            self._initialized = True

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for API requests."""
        headers = {
            "Content-Type": "application/json",
        }
        if self._config.get("api_key"):
            headers["Authorization"] = f"Bearer {self._config['api_key']}"
        
        # Optional: Add HTTP-Referer and X-Title for rankings
        # headers["HTTP-Referer"] = "https://your-app.com"
        # headers["X-Title"] = "Your App Name"
        
        return headers

    def is_available(self) -> bool:
        """Check if OpenRouter service is available (API key configured)."""
        return bool(self._config.get("api_key"))

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        reraise=True
    )
    def test_connection(self) -> Dict[str, Any]:
        """
        Test OpenRouter API connection.
        
        Returns:
            Dict with status, response_time, and available info
        """
        if not self.is_available():
            return {
                "status": "error",
                "message": "OpenRouter API key not configured. Set OPENROUTER_API_KEY in .env file.",
                "authenticated": False
            }

        try:
            start_time = time.time()
            
            # Test with a simple models list request
            response = self._session.get(
                f"{self._base_url}/models",
                headers=self._get_headers(),
                timeout=self._config["timeout"]
            )
            response.raise_for_status()
            
            response_time = time.time() - start_time
            models_data = response.json()
            
            # Count available models
            model_count = len(models_data.get("data", []))
            
            return {
                "status": "success",
                "message": f"Connected to OpenRouter API ({model_count} models available)",
                "response_time": round(response_time, 2),
                "authenticated": True,
                "model_count": model_count,
                "api_version": "v1"
            }

        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP {e.response.status_code}"
            if e.response.status_code == 401:
                error_msg = "Invalid API key"
            elif e.response.status_code == 429:
                error_msg = "Rate limit exceeded"
            
            return {
                "status": "error",
                "message": f"OpenRouter API error: {error_msg}",
                "error_type": type(e).__name__,
                "status_code": e.response.status_code
            }
        
        except requests.exceptions.RequestException as e:
            return {
                "status": "error",
                "message": f"Connection failed: {str(e)}",
                "error_type": type(e).__name__
            }

    def get_available_models(self) -> Dict[str, Any]:
        """
        Get list of available models from OpenRouter.
        
        Returns:
            Dict with status and list of models with their details
        """
        if not self.is_available():
            return {
                "status": "error",
                "message": "OpenRouter API key not configured",
                "models": []
            }

        try:
            response = self._session.get(
                f"{self._base_url}/models",
                headers=self._get_headers(),
                timeout=self._config["timeout"]
            )
            response.raise_for_status()
            
            data = response.json()
            models = data.get("data", [])
            
            # Format model information
            formatted_models = []
            for model in models:
                formatted_models.append({
                    "id": model.get("id"),
                    "name": model.get("name", model.get("id")),
                    "description": model.get("description", ""),
                    "context_length": model.get("context_length"),
                    "pricing": model.get("pricing", {}),
                    "top_provider": model.get("top_provider", {})
                })
            
            return {
                "status": "success",
                "models": formatted_models,
                "count": len(formatted_models)
            }

        except requests.exceptions.RequestException as e:
            return {
                "status": "error",
                "message": f"Failed to get models: {str(e)}",
                "error_type": type(e).__name__,
                "models": []
            }

    def generate_text(
        self,
        prompt: str,
        model: str = "openai/gpt-3.5-turbo",
        max_tokens: int = 250,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate text from a prompt using OpenRouter.
        
        Args:
            prompt: Input text prompt
            model: Model identifier (e.g., "openai/gpt-3.5-turbo", "anthropic/claude-2")
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 to 2.0)
            **kwargs: Additional parameters (top_p, frequency_penalty, etc.)
        
        Returns:
            Dict with status, generated_text, and metadata
        """
        if not self.is_available():
            return {
                "status": "error",
                "message": "OpenRouter API key not configured",
                "model": model
            }

        try:
            payload = {
                "model": model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
                **kwargs
            }

            response = self._session.post(
                f"{self._base_url}/chat/completions",
                headers=self._get_headers(),
                json=payload,
                timeout=self._config["timeout"]
            )
            response.raise_for_status()

            result = response.json()
            
            # Extract generated text
            generated_text = ""
            if "choices" in result and len(result["choices"]) > 0:
                generated_text = result["choices"][0]["message"]["content"]

            # Extract usage info
            usage = result.get("usage", {})

            return {
                "status": "success",
                "generated_text": generated_text,
                "model": result.get("model", model),
                "prompt": prompt,
                "backend": "openrouter",
                "usage": {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0)
                },
                "finish_reason": result["choices"][0].get("finish_reason") if result.get("choices") else None
            }

        except requests.exceptions.HTTPError as e:
            error_detail = "Unknown error"
            try:
                error_data = e.response.json()
                error_detail = error_data.get("error", {}).get("message", str(e))
            except:
                error_detail = str(e)

            return {
                "status": "error",
                "message": f"Text generation failed: {error_detail}",
                "model": model,
                "error_type": type(e).__name__,
                "status_code": e.response.status_code
            }

        except requests.exceptions.RequestException as e:
            return {
                "status": "error",
                "message": f"Request failed: {str(e)}",
                "model": model,
                "error_type": type(e).__name__
            }

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str = "openai/gpt-3.5-turbo",
        max_tokens: int = 250,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a chat response from conversation history.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            model: Model identifier
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 to 2.0)
            **kwargs: Additional parameters
        
        Returns:
            Dict with status, response, and metadata
        """
        if not self.is_available():
            return {
                "status": "error",
                "message": "OpenRouter API key not configured",
                "model": model
            }

        try:
            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                **kwargs
            }

            response = self._session.post(
                f"{self._base_url}/chat/completions",
                headers=self._get_headers(),
                json=payload,
                timeout=self._config["timeout"]
            )
            response.raise_for_status()

            result = response.json()
            
            # Extract response
            assistant_response = ""
            if "choices" in result and len(result["choices"]) > 0:
                assistant_response = result["choices"][0]["message"]["content"]

            # Extract usage info
            usage = result.get("usage", {})

            return {
                "status": "success",
                "response": assistant_response,
                "model": result.get("model", model),
                "message_count": len(messages),
                "backend": "openrouter",
                "usage": {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0)
                },
                "finish_reason": result["choices"][0].get("finish_reason") if result.get("choices") else None
            }

        except requests.exceptions.HTTPError as e:
            error_detail = "Unknown error"
            try:
                error_data = e.response.json()
                error_detail = error_data.get("error", {}).get("message", str(e))
            except:
                error_detail = str(e)

            return {
                "status": "error",
                "message": f"Chat failed: {error_detail}",
                "model": model,
                "error_type": type(e).__name__,
                "status_code": e.response.status_code
            }

        except requests.exceptions.RequestException as e:
            return {
                "status": "error",
                "message": f"Request failed: {str(e)}",
                "model": model,
                "error_type": type(e).__name__
            }

    @classmethod
    def _reset_instance(cls) -> None:
        """Reset the singleton instance. Used for testing."""
        if cls._instance is not None:
            # Remove _initialized flag to allow re-initialization
            if hasattr(cls._instance, '_initialized'):
                delattr(cls._instance, '_initialized')
        cls._instance = None


# Lazy initialization wrapper
class _OpenRouterServiceLazy:
    """Lazy initialization wrapper for OpenRouterService."""

    def __init__(self):
        self._instance = None
        self._instance_error = None

    def __call__(self):
        if self._instance_error:
            raise self._instance_error
        if self._instance is None:
            try:
                self._instance = OpenRouterService()
            except Exception as e:
                self._instance_error = e
                raise
        return self._instance

    def __getattr__(self, name):
        return getattr(self(), name)


# Global service instance
openrouter_service = _OpenRouterServiceLazy()
