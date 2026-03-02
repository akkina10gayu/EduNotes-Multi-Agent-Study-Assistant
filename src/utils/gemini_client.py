"""
Chat LLM Client for the Conversational AI feature.
Supports: Google Gemini (primary), Cerebras (fallback).
Completely separate from the existing Groq-based LLM client to avoid rate limit conflicts.
"""
import os
import time
from typing import Optional, List, Dict, Any, Tuple
from src.utils.logger import get_logger

logger = get_logger(__name__)


class GeminiClient:
    """Google Gemini API client for conversational chat."""

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not set. Get a free key at: https://aistudio.google.com/apikey"
            )

        try:
            from google import genai
            self.client = genai.Client(api_key=api_key)
        except ImportError:
            raise ImportError("google-genai not installed. Run: pip install google-genai")

        self.model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self._last_request_time = 0.0
        self._min_interval = 6.0  # ~10 RPM
        logger.info(f"Initialized Gemini client with model: {self.model}")

    def _check_rate_limit(self):
        """Track request timing. Raises if called faster than min_interval."""
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self._min_interval:
            raise RuntimeError(
                f"Gemini rate limit: {self._min_interval - elapsed:.1f}s until next request"
            )
        self._last_request_time = now

    def chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str = None,
        max_tokens: int = 2048,
        temperature: float = 0.7
    ) -> str:
        """Send messages and get a response using Gemini."""
        from google.genai import types

        self._check_rate_limit()

        contents = []
        for msg in messages:
            role = "model" if msg["role"] == "assistant" else "user"
            contents.append(
                types.Content(role=role, parts=[types.Part(text=msg["content"])])
            )

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        if system_prompt:
            config.system_instruction = system_prompt

        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )
        return response.text

    def generate(
        self,
        prompt: str,
        system_prompt: str = None,
        max_tokens: int = 2048,
        temperature: float = 0.7
    ) -> str:
        """Single-shot generation (convenience wrapper)."""
        return self.chat(
            messages=[{"role": "user", "content": prompt}],
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )


class CerebrasClient:
    """Cerebras API client (fallback). Uses OpenAI-compatible interface."""

    def __init__(self):
        api_key = os.getenv("CEREBRAS_API_KEY", "")
        if not api_key:
            raise ValueError(
                "CEREBRAS_API_KEY not set. Get a free key at: https://cloud.cerebras.ai"
            )

        try:
            from cerebras.cloud.sdk import Cerebras
            self.client = Cerebras(api_key=api_key)
        except ImportError:
            raise ImportError(
                "cerebras-cloud-sdk not installed. Run: pip install cerebras-cloud-sdk"
            )

        self.model = os.getenv("CEREBRAS_MODEL", "llama-3.3-70b")
        logger.info(f"Initialized Cerebras client with model: {self.model}")

    def chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str = None,
        max_tokens: int = 2048,
        temperature: float = 0.7
    ) -> str:
        """Send messages and get a response using Cerebras."""
        api_messages = []
        if system_prompt:
            api_messages.append({"role": "system", "content": system_prompt})
        for msg in messages:
            api_messages.append({"role": msg["role"], "content": msg["content"]})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=api_messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content

    def generate(
        self,
        prompt: str,
        system_prompt: str = None,
        max_tokens: int = 2048,
        temperature: float = 0.7
    ) -> str:
        """Single-shot generation (convenience wrapper)."""
        return self.chat(
            messages=[{"role": "user", "content": prompt}],
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )


class ChatProviderManager:
    """Manages chat LLM providers with automatic fallback.

    Fallback chain: Gemini -> Cerebras -> Groq light model (last resort).
    """

    def __init__(self):
        self.gemini: Optional[GeminiClient] = None
        self.cerebras: Optional[CerebrasClient] = None
        self.provider_name = "none"
        self._init_providers()

    def _init_providers(self):
        """Initialize available providers."""
        try:
            self.gemini = GeminiClient()
            self.provider_name = "gemini"
            logger.info("Chat provider ready: Gemini (primary)")
        except Exception as e:
            logger.warning(f"Gemini not available: {e}")

        try:
            self.cerebras = CerebrasClient()
            if not self.gemini:
                self.provider_name = "cerebras"
                logger.info("Chat provider ready: Cerebras (primary fallback)")
        except Exception as e:
            logger.warning(f"Cerebras not available: {e}")

        if not self.gemini and not self.cerebras:
            logger.warning(
                "No dedicated chat providers available. "
                "Chat will fall back to Groq light model if available."
            )

    def chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str = None,
        max_tokens: int = 2048,
        temperature: float = 0.7
    ) -> Tuple[str, str]:
        """Send messages and get a response. Returns (text, provider_name)."""
        # Try Gemini
        if self.gemini:
            try:
                result = self.gemini.chat(messages, system_prompt, max_tokens, temperature)
                return result, "gemini"
            except Exception as e:
                logger.warning(f"Gemini chat failed: {e}")

        # Try Cerebras
        if self.cerebras:
            try:
                result = self.cerebras.chat(messages, system_prompt, max_tokens, temperature)
                return result, "cerebras"
            except Exception as e:
                logger.warning(f"Cerebras chat failed: {e}")

        # Last resort: Groq light model
        try:
            from src.utils.llm_client import get_llm_client
            llm = get_llm_client()
            if llm.is_available():
                prompt_parts = []
                if system_prompt:
                    prompt_parts.append(f"System: {system_prompt}")
                for msg in messages:
                    prefix = "User" if msg["role"] == "user" else "Assistant"
                    prompt_parts.append(f"{prefix}: {msg['content']}")
                prompt_parts.append("Assistant:")
                prompt = "\n\n".join(prompt_parts)

                result = llm.generate(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    model_override=getattr(llm, "light_model", None),
                )
                return result, "groq_fallback"
        except Exception as e:
            logger.error(f"Groq fallback also failed: {e}")

        raise RuntimeError(
            "No chat providers available. "
            "Please set GEMINI_API_KEY or CEREBRAS_API_KEY in your .env file."
        )

    def generate(
        self,
        prompt: str,
        system_prompt: str = None,
        max_tokens: int = 2048,
        temperature: float = 0.7
    ) -> Tuple[str, str]:
        """Single-shot generation. Returns (text, provider_name)."""
        return self.chat(
            messages=[{"role": "user", "content": prompt}],
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def is_available(self) -> bool:
        """Check if any chat provider is available."""
        if self.gemini or self.cerebras:
            return True
        # Check Groq fallback
        try:
            from src.utils.llm_client import get_llm_client
            return get_llm_client().is_available()
        except Exception:
            return False

    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about available providers."""
        return {
            "primary": "gemini" if self.gemini else None,
            "fallback": "cerebras" if self.cerebras else None,
            "active": self.provider_name,
            "available": self.is_available(),
        }


# Singleton
_chat_provider: Optional[ChatProviderManager] = None


def get_chat_provider() -> ChatProviderManager:
    """Get or create the chat provider manager singleton."""
    global _chat_provider
    if _chat_provider is None:
        _chat_provider = ChatProviderManager()
    return _chat_provider
