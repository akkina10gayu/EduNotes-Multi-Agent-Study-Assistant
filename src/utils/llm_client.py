"""
LLM Client for API-based inference (FREE APIs only)
Supports: Groq (primary), HuggingFace (backup)
Keeps local model as fallback when USE_LOCAL_MODEL=true
"""
import os
from typing import Optional, Dict, Any
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LLMClient:
    """
    Unified interface for LLM inference using FREE APIs.

    Supported providers (all free, no credit card required):
    - groq: Fastest, 14,400 requests/day free (RECOMMENDED)
    - huggingface: 30,000 requests/month free
    - local: Uses existing BART/Flan-T5 models (offline fallback)
    """

    def __init__(self, provider: str = None):
        """
        Initialize LLM client with specified provider.

        Args:
            provider: One of 'groq', 'huggingface', 'local'
                     If None, reads from LLM_PROVIDER env var (default: groq)
        """
        self.provider = provider or os.getenv("LLM_PROVIDER", "groq")
        self.client = None
        self.model = None

        # Check if local mode is forced
        use_local = os.getenv("USE_LOCAL_MODEL", "false").lower() == "true"
        if use_local:
            self.provider = "local"
            logger.info("Using local model (USE_LOCAL_MODEL=true)")

        self._init_client()

    def _init_client(self):
        """Initialize the appropriate client based on provider."""
        try:
            if self.provider == "groq":
                self._init_groq()
            elif self.provider == "huggingface":
                self._init_huggingface()
            elif self.provider == "local":
                self._init_local()
            else:
                logger.warning(f"Unknown provider: {self.provider}, falling back to groq")
                self.provider = "groq"
                self._init_groq()

        except Exception as e:
            logger.error(f"Failed to initialize {self.provider} client: {e}")
            logger.info("Falling back to local model")
            self.provider = "local"
            self._init_local()

    def _init_groq(self):
        """Initialize Groq client (FREE - 14,400 requests/day)."""
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found. Get free key at: https://console.groq.com"
            )

        try:
            from groq import Groq
            self.client = Groq(api_key=api_key)
            self.model = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")
            logger.info(f"Initialized Groq client with model: {self.model}")
        except ImportError:
            raise ImportError("groq package not installed. Run: pip install groq")

    def _init_huggingface(self):
        """Initialize HuggingFace Inference client (FREE - 30,000 requests/month)."""
        api_key = os.getenv("HF_TOKEN")
        if not api_key:
            raise ValueError(
                "HF_TOKEN not found. Get free token at: https://huggingface.co/settings/tokens"
            )

        try:
            from huggingface_hub import InferenceClient
            self.client = InferenceClient(token=api_key)
            self.model = os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
            logger.info(f"Initialized HuggingFace client with model: {self.model}")
        except ImportError:
            raise ImportError("huggingface_hub not installed. Run: pip install huggingface_hub")

    def _init_local(self):
        """Initialize local model flag (actual model loaded in summarizer)."""
        self.client = None
        self.model = "local"
        logger.info("Using local model mode (BART/Flan-T5)")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        system_prompt: str = None
    ) -> str:
        """
        Generate text using the configured LLM provider.

        Args:
            prompt: The user prompt/query
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-1.0)
            system_prompt: Optional system prompt for context

        Returns:
            Generated text string
        """
        try:
            if self.provider == "groq":
                return self._generate_groq(prompt, max_tokens, temperature, system_prompt)
            elif self.provider == "huggingface":
                return self._generate_huggingface(prompt, max_tokens, temperature)
            elif self.provider == "local":
                # Return None to signal caller should use local model
                return None
            else:
                raise ValueError(f"Unknown provider: {self.provider}")

        except Exception as e:
            logger.error(f"Error generating with {self.provider}: {e}")
            raise

    def _generate_groq(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        system_prompt: str = None
    ) -> str:
        """Generate using Groq API."""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )

        return response.choices[0].message.content

    def _generate_huggingface(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float
    ) -> str:
        """Generate using HuggingFace Inference API."""
        response = self.client.text_generation(
            prompt,
            model=self.model,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True
        )

        return response

    def summarize(
        self,
        text: str,
        max_length: int = 1024,
        style: str = "detailed"
    ) -> str:
        """
        Summarize text using the LLM.

        Args:
            text: Text to summarize
            max_length: Maximum summary length in tokens
            style: 'detailed', 'bullet_points', or 'brief'

        Returns:
            Summary string
        """
        if self.provider == "local":
            return None  # Signal to use local model

        # Build prompt based on style
        if style == "bullet_points":
            system_prompt = """You are an expert educational assistant specializing in creating comprehensive, well-structured study notes.
Your notes should be clear, accurate, and optimized for learning and retention.
Focus on extracting key information and presenting it in an easily digestible format."""

            prompt = f"""Create comprehensive bullet-point study notes from the following content.

**Structure your notes to include:**
• **Core Concepts**: Main ideas and fundamental principles
• **Definitions**: Clear explanations of key terms
• **Formulas/Equations**: Any mathematical expressions (if applicable)
• **Examples**: Concrete illustrations of concepts
• **Applications**: Real-world uses or practical applications
• **Key Relationships**: How concepts connect and interact
• **Important Details**: Critical facts and data points

**Format Guidelines:**
- Use hierarchical bullet points (main points → sub-points)
- Keep language clear and concise
- Highlight technical terms
- Include specific examples when available
- Organize from fundamental to advanced concepts

Content to summarize:
{text}

Provide comprehensive, well-structured bullet-point notes:"""

        elif style == "brief":
            system_prompt = "You are an expert at creating concise, accurate summaries that capture the essence of educational content."
            prompt = f"""Summarize the following content in 2-3 clear, informative sentences.
Focus on the main takeaways and key insights.

Content:
{text}

Brief summary:"""

        else:  # detailed
            system_prompt = """You are an expert educational content creator who produces detailed, comprehensive study materials.
Your summaries should be thorough, well-organized, and designed to facilitate deep understanding and retention.
Include definitions, examples, applications, and clear explanations of complex concepts."""

            prompt = f"""Create a comprehensive, detailed educational summary of the following content.

**Your summary should include:**

1. **Overview**: Brief introduction to the topic (2-3 sentences)

2. **Core Concepts**: Detailed explanation of main ideas
   - Define key terms clearly
   - Explain fundamental principles
   - Break down complex ideas into understandable parts

3. **Key Details**:
   - Important formulas, equations, or algorithms (if applicable)
   - Critical facts, data points, or statistics
   - Significant terminology and definitions

4. **Examples & Applications**:
   - Concrete examples illustrating concepts
   - Real-world applications or use cases
   - Practical scenarios demonstrating the ideas

5. **Relationships & Connections**:
   - How different concepts relate to each other
   - Dependencies and prerequisites
   - Common patterns or themes

6. **Important Takeaways**:
   - Essential points to remember
   - Key insights and conclusions

**Style Guidelines:**
- Use clear, accessible language
- Organize information logically
- Include specific details and examples
- Make it suitable for studying and reference
- Structure with clear headings and sections

Content to summarize:
{text}

Provide a detailed, well-organized educational summary:"""

        try:
            return self.generate(
                prompt=prompt,
                max_tokens=max_length,
                temperature=0.7,
                system_prompt=system_prompt
            )
        except Exception as e:
            logger.error(f"Error in summarize: {e}")
            raise

    def generate_flashcards(self, text: str, num_cards: int = 10) -> str:
        """
        Generate flashcards from text content.

        Args:
            text: Source text for flashcard generation
            num_cards: Number of flashcards to generate

        Returns:
            JSON-formatted string of flashcards
        """
        if self.provider == "local":
            return None

        system_prompt = """You are an educational assistant that creates effective flashcards for studying.
Create clear question-answer pairs that test understanding of key concepts."""

        prompt = f"""Create {num_cards} flashcards from the following content.
Each flashcard should have a clear question on one side and a concise answer on the other.
Focus on key concepts, definitions, and important facts.

Format each flashcard as:
Q: [Question]
A: [Answer]

Content:
{text}

Generate {num_cards} flashcards:"""

        try:
            return self.generate(
                prompt=prompt,
                max_tokens=1500,
                temperature=0.7,
                system_prompt=system_prompt
            )
        except Exception as e:
            logger.error(f"Error generating flashcards: {e}")
            raise

    def generate_quiz(self, text: str, num_questions: int = 5) -> str:
        """
        Generate quiz questions from text content.

        Args:
            text: Source text for quiz generation
            num_questions: Number of questions to generate

        Returns:
            JSON-formatted string of quiz questions
        """
        if self.provider == "local":
            return None

        system_prompt = """You are an educational assistant that creates quiz questions.
Create multiple choice questions that test understanding of the material."""

        prompt = f"""Create {num_questions} multiple choice quiz questions from the following content.
Each question should have 4 options (A, B, C, D) with one correct answer.

Format each question as:
Question: [Question text]
A) [Option A]
B) [Option B]
C) [Option C]
D) [Option D]
Correct: [Letter]
Explanation: [Brief explanation why this is correct]

Content:
{text}

Generate {num_questions} quiz questions:"""

        try:
            return self.generate(
                prompt=prompt,
                max_tokens=2000,
                temperature=0.7,
                system_prompt=system_prompt
            )
        except Exception as e:
            logger.error(f"Error generating quiz: {e}")
            raise

    def is_local_mode(self) -> bool:
        """Check if using local model mode."""
        return self.provider == "local"

    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about current provider."""
        return {
            "provider": self.provider,
            "model": self.model,
            "is_local": self.is_local_mode()
        }


# Singleton instance for reuse
_llm_client = None

def get_llm_client(provider: str = None) -> LLMClient:
    """
    Get or create LLM client instance.

    Args:
        provider: Optional provider override

    Returns:
        LLMClient instance
    """
    global _llm_client

    if _llm_client is None or (provider and provider != _llm_client.provider):
        _llm_client = LLMClient(provider=provider)

    return _llm_client
