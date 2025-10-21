"""
HuggingFace service for AI model interactions.

This module provides integration with both local models (via transformers/sentence-transformers)
and HuggingFace API for text generation, embeddings, and other AI model capabilities.

Features:
- Local model support with lazy loading
- HuggingFace API integration
- Automatic fallback mechanism
- Configurable model selection
- Memory monitoring and batch processing
- Retry logic for model downloads
"""

import requests
import time
import warnings
from typing import Optional, Dict, Any, List, Union, Callable
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from Common.services.config_service import config_service

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class HuggingFaceService:
    """
    Service for interacting with HuggingFace models (local and API).
    
    Supports:
    - Local embeddings via sentence-transformers
    - Local text generation via transformers
    - Remote API calls for both embeddings and generation
    - Automatic model caching and lazy loading
    """

    _instance: Optional['HuggingFaceService'] = None

    def __new__(cls) -> 'HuggingFaceService':
        """Singleton pattern to ensure single service instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the HuggingFace service."""
        if not hasattr(self, '_initialized'):
            self._config = config_service.get_huggingface_config()
            self._embedding_config = config_service.get_embedding_config()
            self._session = requests.Session()
            
            # Lazy-loaded local models
            self._local_embedding_model = None
            self._local_text_model = None
            self._local_tokenizer = None
            
            # Check availability of local libraries
            self._has_sentence_transformers = self._check_library('sentence_transformers')
            self._has_transformers = self._check_library('transformers')
            
            self._initialized = True

    def _check_library(self, library_name: str) -> bool:
        """Check if a library is available."""
        try:
            __import__(library_name)
            return True
        except ImportError:
            return False

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for API requests."""
        headers = {
            "Content-Type": "application/json",
        }
        if self._config.get("token"):
            headers["Authorization"] = f"Bearer {self._config['token']}"
        return headers

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        reraise=True
    )
    def _load_local_embedding_model(
        self,
        model_name: Optional[str] = None,
        progress_callback: Optional[Callable[[str], None]] = None
    ):
        """
        Lazy load local embedding model with retry logic and progress feedback.
        
        Args:
            model_name: Optional model name, uses configured default if None
            progress_callback: Optional callback function for progress updates
            
        Returns:
            Loaded SentenceTransformer model
            
        Raises:
            RuntimeError: If sentence-transformers not installed
            ConnectionError: If model download fails after retries
        """
        if not self._has_sentence_transformers:
            raise RuntimeError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
        
        if self._local_embedding_model is None:
            from sentence_transformers import SentenceTransformer
            
            model_name = model_name or self._embedding_config.get("model", "all-MiniLM-L6-v2")
            
            if progress_callback:
                progress_callback(f"Loading model: {model_name}")
            
            try:
                self._local_embedding_model = SentenceTransformer(model_name)
                
                if progress_callback:
                    progress_callback("Model loaded successfully")
                    
            except Exception as e:
                # Log error and raise for retry
                import logging
                logging.error(f"Failed to load embedding model {model_name}: {e}")
                
                if progress_callback:
                    progress_callback(f"Error loading model: {e}")
                    
                raise
        
        return self._local_embedding_model

    def _validate_text_length(
        self, 
        texts: List[str], 
        model_max_length: int = 512
    ) -> Dict[str, Any]:
        """
        Validate and potentially warn about text lengths.
        
        Args:
            texts: List of texts to validate
            model_max_length: Maximum token length for the model
            
        Returns:
            Dict with validated texts and warnings
        """
        warnings_list = []
        
        for i, text in enumerate(texts):
            # Simple tokenization estimate (words * 1.3 ≈ tokens)
            estimated_tokens = len(text.split()) * 1.3
            
            if estimated_tokens > model_max_length:
                warnings_list.append({
                    "index": i,
                    "estimated_tokens": int(estimated_tokens),
                    "max_tokens": model_max_length,
                    "text_preview": text[:100] + "..." if len(text) > 100 else text,
                    "action": "will_be_truncated_by_model"
                })
        
        return {
            "texts": texts,
            "warnings": warnings_list,
            "truncated_count": len(warnings_list)
        }

    def _check_memory_availability(self, required_gb: float = 2.0) -> None:
        """
        Check if sufficient memory is available.
        
        Args:
            required_gb: Minimum required memory in GB
            
        Raises:
            RuntimeError: If insufficient memory available
        """
        try:
            import psutil
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            max_memory = config_service.settings.max_model_memory_gb
            
            if available_memory_gb < required_gb:
                raise RuntimeError(
                    f"Insufficient memory: {available_memory_gb:.1f}GB available, "
                    f"{required_gb:.1f}GB required. "
                    f"Close other applications or increase MAX_MODEL_MEMORY_GB in .env"
                )
            
            if available_memory_gb < max_memory:
                import warnings
                warnings.warn(
                    f"Low memory: {available_memory_gb:.1f}GB available, "
                    f"recommended minimum is {max_memory}GB"
                )
        except ImportError:
            # psutil not installed, skip check
            pass

    def _load_local_text_model(self, model_name: str):
        """Lazy load local text generation model with memory checks."""
        if not self._has_transformers:
            raise RuntimeError(
                "transformers not installed. "
                "Install with: pip install transformers torch"
            )
        
        # Check memory before loading
        self._check_memory_availability(required_gb=2.0)
        
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        # Load model and tokenizer
        if self._local_text_model is None or self._local_tokenizer is None:
            self._local_tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Determine load parameters
            load_in_8bit = (
                config_service.settings.enable_model_quantization and
                torch.cuda.is_available()
            )
            
            if load_in_8bit:
                self._local_text_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    load_in_8bit=True,
                    device_map="auto",
                    low_cpu_mem_usage=True
                )
            else:
                self._local_text_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None,
                    low_cpu_mem_usage=True
                )
        
        return self._local_text_model, self._local_tokenizer

    def test_connection(self) -> Dict[str, Any]:
        """
        Test both local models and HuggingFace API connection.
        
        Returns detailed status for each component.
        """
        results = {
            "status": "success",
            "local_models": {},
            "api_connection": {},
            "message": ""
        }
        
        # Test local sentence-transformers
        if self._has_sentence_transformers:
            try:
                model = self._load_local_embedding_model()
                test_embedding = model.encode(["test"])
                results["local_models"]["sentence_transformers"] = {
                    "status": "available",
                    "model": self._embedding_config.get("model", "all-MiniLM-L6-v2"),
                    "embedding_dim": len(test_embedding[0])
                }
            except Exception as e:
                results["local_models"]["sentence_transformers"] = {
                    "status": "error",
                    "error": str(e)
                }
        else:
            results["local_models"]["sentence_transformers"] = {
                "status": "not_installed",
                "message": "Install with: pip install sentence-transformers"
            }
        
        # Test local transformers
        if self._has_transformers:
            results["local_models"]["transformers"] = {
                "status": "available",
                "message": "Ready for local text generation"
            }
        else:
            results["local_models"]["transformers"] = {
                "status": "not_installed",
                "message": "Install with: pip install transformers torch"
            }
        
        # Test HuggingFace API
        try:
            start_time = time.time()
            response = self._session.get(
                "https://huggingface.co/api/models",
                headers=self._get_headers(),
                timeout=self._config["timeout"],
                params={"limit": 1}
            )
            response.raise_for_status()
            response_time = time.time() - start_time
            
            results["api_connection"] = {
                "status": "connected",
                "response_time": round(response_time, 2),
                "authenticated": bool(self._config.get("token"))
            }
        except requests.exceptions.RequestException as e:
            results["api_connection"] = {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__
            }
            results["status"] = "partial"
        
        # Set overall message
        local_ok = any(
            v.get("status") == "available" 
            for v in results["local_models"].values()
        )
        api_ok = results["api_connection"].get("status") == "connected"
        
        if local_ok and api_ok:
            results["message"] = "All systems operational (local models + API)"
        elif local_ok:
            results["message"] = "Local models available, API unavailable"
        elif api_ok:
            results["message"] = "API available, local models unavailable"
        else:
            results["status"] = "error"
            results["message"] = "No working backends available"
        
        return results

    def generate_text(
        self,
        prompt: str,
        model: str = "microsoft/DialoGPT-medium",
        max_new_tokens: int = 250,
        temperature: float = 0.7,
        top_p: float = 0.95,
        use_local: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate text using either local model or HuggingFace API.
        
        Args:
            prompt: Input text prompt
            model: Model name/path
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            use_local: Force local model usage
            **kwargs: Additional generation parameters
        
        Returns:
            Dict with status, generated_text, model, and other metadata
        """
        # Try local model first if requested or if API is not available
        if use_local or not self._config.get("token"):
            if self._has_transformers:
                try:
                    return self._generate_text_local(
                        prompt, model, max_new_tokens, temperature, top_p, **kwargs
                    )
                except Exception as e:
                    if use_local:
                        return {
                            "status": "error",
                            "message": f"Local text generation failed: {str(e)}",
                            "model": model,
                            "error_type": type(e).__name__
                        }
                    # Fall through to API if not forcing local
            elif use_local:
                return {
                    "status": "error",
                    "message": "Local generation requested but transformers not installed",
                    "model": model
                }
        
        # Use API
        return self._generate_text_api(
            prompt, model, max_new_tokens, temperature, top_p, **kwargs
        )

    def _generate_text_local(
        self,
        prompt: str,
        model: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate text using local transformers model."""
        try:
            import torch
            
            model_obj, tokenizer = self._load_local_text_model(model)
            
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = model_obj.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    **kwargs
                )
            
            # Decode
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove prompt from output if present
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            return {
                "status": "success",
                "generated_text": generated_text,
                "model": model,
                "prompt": prompt,
                "backend": "local",
                "device": "cuda" if torch.cuda.is_available() else "cpu"
            }
        
        except Exception as e:
            raise RuntimeError(f"Local generation error: {str(e)}")

    def _generate_text_api(
        self,
        prompt: str,
        model: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate text using HuggingFace API."""
        try:
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "return_full_text": False,
                    **kwargs
                },
                "options": {"wait_for_model": True}
            }

            response = self._session.post(
                f"https://api-inference.huggingface.co/models/{model}",
                headers=self._get_headers(),
                json=payload,
                timeout=self._config["timeout"]
            )
            response.raise_for_status()

            result = response.json()

            # Handle different response formats
            if isinstance(result, list) and result:
                generated_text = result[0].get("generated_text", "")
            elif isinstance(result, dict):
                generated_text = result.get("generated_text", "")
            else:
                generated_text = str(result)

            return {
                "status": "success",
                "generated_text": generated_text,
                "model": model,
                "prompt": prompt,
                "backend": "api"
            }

        except requests.exceptions.RequestException as e:
            return {
                "status": "error",
                "message": f"API text generation failed: {str(e)}",
                "model": model,
                "error_type": type(e).__name__
            }

    def get_embeddings(
        self,
        texts: List[str],
        model: Optional[str] = None,
        use_local: bool = True
    ) -> Dict[str, Any]:
        """
        Get embeddings using local model (preferred) or HuggingFace API.
        
        Args:
            texts: List of texts to embed
            model: Model name (None = use configured default)
            use_local: Try local model first
        
        Returns:
            Dict with status, embeddings, model, and metadata
        """
        # Try local first if available and requested
        if use_local and self._has_sentence_transformers:
            try:
                return self.get_local_embeddings(texts, model)
            except Exception as e:
                # Fall back to API if local fails
                if self._config.get("token"):
                    pass  # Continue to API
                else:
                    return {
                        "status": "error",
                        "message": f"Local embeddings failed and no API token: {str(e)}",
                        "error_type": type(e).__name__
                    }
        
        # Use API
        if not model:
            model = "sentence-transformers/all-MiniLM-L6-v2"
        
        return self._get_embeddings_api(texts, model)

    def get_local_embeddings(
        self,
        texts: List[str],
        model: Optional[str] = None,
        batch_size: int = 32,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> Dict[str, Any]:
        """
        Get embeddings using local sentence-transformers model with batch processing.
        
        Args:
            texts: List of texts to embed
            model: Model name (None = use configured default)
            batch_size: Number of texts to process in each batch (default: 32)
            progress_callback: Optional callback for progress updates
        
        Returns:
            Dict with status, embeddings, model, and metadata
        """
        try:
            # Validate text lengths
            validation = self._validate_text_length(texts)
            if validation["warnings"]:
                import logging
                logging.warning(
                    f"{validation['truncated_count']} text(s) exceed recommended length "
                    f"and may be truncated by the model"
                )
            
            embedding_model = self._load_local_embedding_model(model, progress_callback)
            model_name = model or self._embedding_config.get("model", "all-MiniLM-L6-v2")
            
            # Get batch size from config if not specified
            if batch_size == 32:  # default value
                batch_size = config_service.settings.embedding_batch_size
            
            # Process in batches for large datasets
            all_embeddings = []
            total_batches = (len(texts) + batch_size - 1) // batch_size
            
            for batch_num, i in enumerate(range(0, len(texts), batch_size), 1):
                batch = texts[i:i+batch_size]
                
                if progress_callback:
                    progress_callback(f"Processing batch {batch_num}/{total_batches}")
                
                batch_embeddings = embedding_model.encode(
                    batch, 
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
                all_embeddings.extend(batch_embeddings.tolist())
            
            if progress_callback:
                progress_callback(f"Completed: {len(texts)} embeddings generated")
            
            result = {
                "status": "success",
                "embeddings": all_embeddings,
                "model": model_name,
                "text_count": len(texts),
                "batch_size": batch_size,
                "batches_processed": total_batches,
                "dimensions": len(all_embeddings[0]) if all_embeddings else 0,
                "backend": "local"
            }
            
            # Add validation warnings if any
            if validation["warnings"]:
                result["validation_warnings"] = validation["truncated_count"]
            
            return result
        
        except Exception as e:
            return {
                "status": "error",
                "message": f"Local embedding generation failed: {str(e)}",
                "model": model or "default",
                "error_type": type(e).__name__
            }

    def _get_embeddings_api(
        self,
        texts: List[str],
        model: str
    ) -> Dict[str, Any]:
        """Get embeddings using HuggingFace API."""
        try:
            payload = {
                "inputs": texts,
                "options": {"wait_for_model": True}
            }

            response = self._session.post(
                f"https://api-inference.huggingface.co/models/{model}",
                headers=self._get_headers(),
                json=payload,
                timeout=self._config["timeout"]
            )
            response.raise_for_status()

            embeddings = response.json()

            return {
                "status": "success",
                "embeddings": embeddings,
                "model": model,
                "text_count": len(texts),
                "dimensions": len(embeddings[0]) if embeddings else 0,
                "backend": "api"
            }

        except requests.exceptions.RequestException as e:
            return {
                "status": "error",
                "message": f"API embedding generation failed: {str(e)}",
                "model": model,
                "error_type": type(e).__name__
            }

    def list_models(
        self,
        search: str = "",
        limit: int = 10,
        filter_by: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        List available models on HuggingFace Hub.
        
        Args:
            search: Search query
            limit: Maximum number of results
            filter_by: Filter by task (e.g., 'text-generation', 'feature-extraction')
        
        Returns:
            Dict with status, models list, count, and search info
        """
        try:
            params: Dict[str, Any] = {"limit": limit, "sort": "downloads", "direction": -1}
            if search:
                params["search"] = search
            if filter_by:
                params["filter"] = filter_by

            response = self._session.get(
                "https://huggingface.co/api/models",
                headers=self._get_headers(),
                params=params,
                timeout=self._config["timeout"]
            )
            response.raise_for_status()

            models = response.json()

            return {
                "status": "success",
                "models": models,
                "count": len(models),
                "search": search,
                "filter": filter_by
            }

        except requests.exceptions.RequestException as e:
            return {
                "status": "error",
                "message": f"Failed to list models: {str(e)}",
                "error_type": type(e).__name__
            }

    def is_available(self) -> bool:
        """Check if service has any working backend."""
        return (
            self._has_sentence_transformers or 
            self._has_transformers or 
            bool(self._config.get("token"))
        )

    def get_capabilities(self) -> Dict[str, Any]:
        """Get information about available capabilities."""
        return {
            "local_embeddings": self._has_sentence_transformers,
            "local_generation": self._has_transformers,
            "api_access": bool(self._config.get("token")),
            "configured_embedding_model": self._embedding_config.get("model"),
        }

    @classmethod
    def _reset_instance(cls) -> None:
        """Reset the singleton instance. Used for testing."""
        if cls._instance is not None:
            # Clean up loaded models
            if hasattr(cls._instance, '_local_embedding_model'):
                cls._instance._local_embedding_model = None
            if hasattr(cls._instance, '_local_text_model'):
                cls._instance._local_text_model = None
            if hasattr(cls._instance, '_local_tokenizer'):
                cls._instance._local_tokenizer = None
        cls._instance = None


# Global service instance
huggingface_service = HuggingFaceService()


def test_huggingface_connection() -> Dict[str, Any]:
    """Test HuggingFace service connection - convenience function."""
    return huggingface_service.test_connection()


def generate_text_huggingface(
    prompt: str,
    model: str = "microsoft/DialoGPT-medium",
    **kwargs
) -> Dict[str, Any]:
    """Generate text using HuggingFace - convenience function."""
    return huggingface_service.generate_text(prompt, model, **kwargs)


def get_embeddings_huggingface(
    texts: List[str],
    model: Optional[str] = None,
    use_local: bool = True
) -> Dict[str, Any]:
    """Get embeddings using HuggingFace - convenience function."""
    return huggingface_service.get_embeddings(texts, model, use_local)