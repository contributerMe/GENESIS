import json
import os
import logging
from typing import Optional, Type, TypeVar, Any, Dict, List
import litellm
from pydantic import BaseModel
from langchain_core.embeddings import Embeddings
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from ai.settings import get_settings

logger = logging.getLogger(__name__)

# Suppress verbose litellm logs by default
litellm.suppress_debug_info = True

# Enable Langfuse observability if keys are configured
_settings = get_settings()
if _settings.langfuse_enabled:
    litellm.success_callback = ["langfuse"]
    litellm.failure_callback = ["langfuse"]
    logger.info("Langfuse Observability & Traceability enabled for LiteLLM completions.")

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODELS = {
    "gemini": "gemini-3.6-flash",
    "groq": "llama-3.3-70b-versatile",
    "openai": "gpt-4o-mini",
    "openrouter": "google/gemini-2.0-flash-lite:free",
    "ollama": "llama3.3"
}

class LiteLLMRouter:
    """
    Unified BYO-Key LLM Router using litellm.completion() directly.
    """

    @staticmethod
    def get_model_identifier(provider: str = "openai", model_name: Optional[str] = None) -> str:
        """
        Map provider and optional model_name to LiteLLM model identifier string.
        """
        provider_clean = provider.lower().strip()
        base_model = model_name or DEFAULT_MODELS.get(provider_clean, "gpt-4o-mini")

        if provider_clean == "groq":
            return base_model if base_model.startswith("groq/") else f"groq/{base_model}"
        elif provider_clean == "gemini":
            return base_model if base_model.startswith("gemini/") else f"gemini/{base_model}"
        elif provider_clean == "openrouter":
            return base_model if base_model.startswith("openrouter/") else f"openrouter/{base_model}"
        elif provider_clean == "ollama":
            return base_model if base_model.startswith("ollama/") else f"ollama/{base_model}"
        else:
            return base_model

    @staticmethod
    def get_api_key(provider: str = "openai", user_key: Optional[str] = None) -> Optional[str]:
        """
        Resolve API key for specified provider via centralized Settings.
        """
        settings = get_settings()
        return settings.get_provider_key(provider, user_key)

    @staticmethod
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1.5, min=2, max=10),
        retry=(retry_if_exception_type(litellm.exceptions.ServiceUnavailableError) | retry_if_exception_type(litellm.exceptions.RateLimitError) | retry_if_exception_type(litellm.exceptions.APIConnectionError)),
        reraise=True,
        before_sleep=lambda retry_state: logger.warning(f"Retrying litellm.completion (attempt {retry_state.attempt_number}) due to {retry_state.outcome.exception()}")
    )
    def _execute_with_retries(kwargs: Dict[str, Any], response_format: Optional[Type[T]] = None) -> Any:
        model_id = kwargs.get("model")
        logger.info(f"Executing litellm.completion(model='{model_id}')")
        response = litellm.completion(**kwargs)

        if response_format:
            raw_content = response.choices[0].message.content
            if isinstance(raw_content, str):
                return response_format.model_validate_json(raw_content)
            elif isinstance(raw_content, dict):
                return response_format.model_validate(raw_content)
            return raw_content
        else:
            return response.choices[0].message.content

    @staticmethod
    def load_prompt(template_name: str) -> str:
        """
        Loads a prompt template from the local ai/prompts directory.
        """
        base_dir = os.path.dirname(os.path.abspath(__file__))
        prompt_path = os.path.join(base_dir, "prompts", f"{template_name}.txt")
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except FileNotFoundError:
            logger.error(f"Prompt template {template_name} not found at {prompt_path}")
            return ""

    @staticmethod
    def build_messages(system_prompt: str, user_data: Any) -> List[Dict[str, str]]:
        """
        Constructs a structured JSON message list separating system instructions and user data.
        """
        messages = [{"role": "system", "content": system_prompt}]
        if isinstance(user_data, str):
            messages.append({"role": "user", "content": user_data})
        else:
            messages.append({"role": "user", "content": json.dumps(user_data, indent=2)})
        return messages

    @staticmethod
    def completion(
        provider: str = "openai",
        messages: Optional[List[Dict[str, str]]] = None,
        prompt: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        response_format: Optional[Type[T]] = None
    ) -> Any:
        """
        Execute completion using litellm.completion() with automatic fallback capabilities.
        """
        if not messages:
            if prompt:
                messages = [{"role": "user", "content": prompt}]
            else:
                raise ValueError("Either 'messages' or 'prompt' must be provided to completion()")

        settings = get_settings()
        primary_model_id = LiteLLMRouter.get_model_identifier(provider, model_name)
        primary_key = LiteLLMRouter.get_api_key(provider, api_key)

        base_kwargs: Dict[str, Any] = {
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens:
            base_kwargs["max_tokens"] = max_tokens
        if response_format:
            base_kwargs["response_format"] = response_format

        # Construct primary kwargs
        kwargs = base_kwargs.copy()
        kwargs["model"] = primary_model_id
        if primary_key:
            kwargs["api_key"] = primary_key

        try:
            return LiteLLMRouter._execute_with_retries(kwargs, response_format)
        except Exception as primary_exc:
            logger.error(f"Primary model '{primary_model_id}' failed: {primary_exc}")
            
            # Execute Fallback Chain
            fallback_models_str = settings.llm_fallback_models
            if not fallback_models_str:
                raise primary_exc
                
            fallback_models = [m.strip() for m in fallback_models_str.split(",") if m.strip() and m.strip() != primary_model_id]
            if not fallback_models:
                raise primary_exc
                
            logger.warning(f"Initiating LLM fallback chain. Attempting {len(fallback_models)} fallback models...")
            
            last_exc = primary_exc
            for fb_model in fallback_models:
                try:
                    logger.warning(f"--- Fallback Attempt: {fb_model} ---")
                    fb_provider = fb_model.split("/")[0] if "/" in fb_model else "openai"
                    fb_key = LiteLLMRouter.get_api_key(fb_provider)
                    if fb_provider != "ollama" and not fb_key:
                        logger.warning("Skipping fallback model '%s': no %s API key is configured.", fb_model, fb_provider)
                        continue
                    
                    fb_kwargs = base_kwargs.copy()
                    fb_kwargs["model"] = fb_model
                    if fb_key:
                        fb_kwargs["api_key"] = fb_key
                    if settings.litellm_drop_params:
                        fb_kwargs["drop_params"] = True
                        
                    return LiteLLMRouter._execute_with_retries(fb_kwargs, response_format)
                except Exception as fb_exc:
                    logger.error(f"Fallback model '{fb_model}' failed: {fb_exc}")
                    last_exc = fb_exc
                    
            logger.error("All fallback models exhausted.")
            raise last_exc

class EmbeddingFactory:

    @staticmethod
    def get_embeddings(
        provider: str = "openai",
        api_key: Optional[str] = None
    ) -> Embeddings:
        """
        Instantiate high-dimensional embedding model based on provider.
        """
        settings = get_settings()
        provider_clean = provider.lower().strip()
        
        if provider_clean == "gemini":
            key = api_key or settings.gemini_key
            if not key:
                raise ValueError("GEMINI_API_KEY is required for Gemini embeddings.")
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            return GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001",
                google_api_key=key
            )
        else:
            # Default to OpenAI if not gemini
            key = api_key or settings.openai_api_key
            if not key:
                raise ValueError("OPENAI_API_KEY is required for OpenAI embeddings.")
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=key
            )
