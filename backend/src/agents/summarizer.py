"""
Summarization agent with FREE API support (Groq, HuggingFace)
Falls back to local model (Flan-T5) when USE_LOCAL_MODEL=true

Version 2.0 - API-first approach with local fallback
"""
from typing import Dict, Any, List
from src.agents.base import BaseAgent
from src.utils.cache_utils import cached
from src.utils.llm_client import get_llm_client
from config import settings


class SummarizerAgent(BaseAgent):
    """
    Agent for text summarization using FREE LLM APIs.

    Supports:
    - Groq API (primary, free, fastest)
    - HuggingFace Inference API (backup, free)
    - Local models (fallback, offline)
    """

    def __init__(self):
        super().__init__("SummarizerAgent")
        self.llm_client = None
        self.use_local = getattr(settings, 'USE_LOCAL_MODEL', False) and settings.ENABLE_LOCAL_FALLBACK

        # Local model components (lazy loaded)
        self._tokenizer = None
        self._model = None
        self._pipe = None
        self._llm = None

        self._initialize()

    def _initialize(self):
        """Initialize the appropriate summarization backend."""
        try:
            if self.use_local:
                self.logger.info("Using LOCAL model for summarization")
                self._initialize_local_model()
            else:
                self.logger.info("Using FREE API for summarization")
                self.llm_client = get_llm_client()
                self.logger.info(f"LLM Provider: {self.llm_client.get_provider_info()}")

                # If API client fell back to local, initialize local model
                if self.llm_client.is_local_mode():
                    self._initialize_local_model()

        except Exception as e:
            self.logger.error(f"Error initializing summarizer: {e}")
            if settings.ENABLE_LOCAL_FALLBACK:
                self.logger.info("Falling back to local model")
                self.use_local = True
                self._initialize_local_model()
            else:
                raise

    def _initialize_local_model(self):
        """Initialize local Flan-T5/BART model (lazy loading)."""
        if not settings.ENABLE_LOCAL_FALLBACK:
            self.logger.warning("Local model disabled by ENABLE_LOCAL_FALLBACK=false")
            return
        # Models are loaded on first use via properties
        self.use_local = True
        self.logger.info(f"Local model configured: {settings.SUMMARIZATION_MODEL}")

    @property
    def tokenizer(self):
        """Lazy load tokenizer."""
        if self._tokenizer is None:
            if not settings.ENABLE_LOCAL_FALLBACK:
                raise RuntimeError("Local model disabled. Set ENABLE_LOCAL_FALLBACK=true in .env to enable.")
            self.logger.info("Loading local tokenizer...")
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                settings.SUMMARIZATION_MODEL,
                cache_dir=str(settings.MODEL_CACHE_DIR)
            )
        return self._tokenizer

    @property
    def model(self):
        """Lazy load model."""
        if self._model is None:
            if not settings.ENABLE_LOCAL_FALLBACK:
                raise RuntimeError("Local model disabled. Set ENABLE_LOCAL_FALLBACK=true in .env to enable.")
            self.logger.info("Loading local model...")
            import torch
            from transformers import AutoModelForSeq2SeqLM
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                settings.SUMMARIZATION_MODEL,
                cache_dir=str(settings.MODEL_CACHE_DIR),
                torch_dtype=torch.float32
            )
        return self._model

    @property
    def pipe(self):
        """Lazy load pipeline."""
        if self._pipe is None:
            if not settings.ENABLE_LOCAL_FALLBACK:
                raise RuntimeError("Local model disabled. Set ENABLE_LOCAL_FALLBACK=true in .env to enable.")
            self.logger.info("Creating local summarization pipeline...")
            from transformers import pipeline
            self._pipe = pipeline(
                "summarization",
                model=self.model,
                tokenizer=self.tokenizer,
                max_length=1024,
                min_length=200,
                do_sample=True,
                early_stopping=True,
                device=-1  # CPU
            )
        return self._pipe

    @cached("summarizer", ttl=settings.CACHE_SUMMARY_TTL)
    def summarize_text(self, text: str, max_length: int = None, style: str = "paragraph_summary", output_length: str = "auto", extra_instructions: str = None) -> str:
        """
        Summarize text using API or local model.

        Args:
            text: Text to summarize
            max_length: Maximum output length
            style: 'paragraph_summary', 'important_points', or 'key_highlights'
            output_length: 'auto', 'detailed', 'medium', or 'brief' (only for paragraph_summary)
            extra_instructions: Optional content-specific instructions from ContentAgent

        Returns:
            Summary string
        """
        try:
            original_length = len(text)
            self.logger.info(f"Summarizing text of length: {original_length} characters in '{style}' mode")

            # Truncate very long text (increased limit for full paper coverage)
            if len(text) > 50000:
                text = text[:50000]
                self.logger.warning(f"Truncated input text from {original_length} to 50000 characters")

            self.logger.info(f"Processing {len(text)} characters with style='{style}', output_length='{output_length}'")

            # Try API first (if not in local mode)
            if not self.use_local and self.llm_client:
                try:
                    summary = self.llm_client.summarize(
                        text=text,
                        max_length=max_length or 3072,  # Tripled output space for quality
                        style=style,
                        output_length=output_length,
                        extra_instructions=extra_instructions
                    )
                    if summary:
                        self.logger.info(f"Summary generated using API - Length: {len(summary)} chars, Style: {style}, OutputLength: {output_length}")
                        return summary
                except Exception as e:
                    err_str = str(e)
                    is_rate_limit = '429' in err_str and (
                        'rate_limit' in err_str.lower() or 'rate limit' in err_str.lower()
                    )
                    if is_rate_limit:
                        self.logger.warning(f"Rate limit hit on main model: {e}")
                        # Try light model as fallback before giving up
                        fallback = self._summarize_rate_limit_fallback(text, style)
                        if fallback:
                            return fallback
                        # Both models exhausted — return clean message (not raw error)
                        return (
                            "**Rate Limit Reached**\n\n"
                            "The primary and backup models have reached their free-tier limits. "
                            "Your notes could not be generated at this time.\n\n"
                            "**What you can do:**\n"
                            "- Wait for the daily rate limit to reset (resets every 24 hours)\n"
                            "- Upgrade your Groq plan at https://console.groq.com/settings/billing\n"
                        )
                    self.logger.warning(f"API summarization failed: {e}, falling back to local")

            # Fallback to local model
            return self._summarize_local(text, max_length)

        except Exception as e:
            self.logger.error(f"Error summarizing text: {e}")
            return f"Summary of content about {text[:100]}... [Error: {str(e)}]"

    def _summarize_local(self, text: str, max_length: int = None) -> str:
        """Summarize using local model (Flan-T5)."""
        try:
            original_text = text

            # Prepare text for local model - Flan-T5 works best with task prefix
            if len(text) > 3000:
                text = text[:3000]

            # Add Flan-T5 task prefix for better summarization
            task_text = f"summarize: {text}"

            # Encode and truncate to model limit
            input_tokens = self.tokenizer.encode(task_text)
            if len(input_tokens) > 1024:
                input_tokens = input_tokens[:1024]
                task_text = self.tokenizer.decode(input_tokens, skip_special_tokens=True)

            # Calculate output length - aim for meaningful summary
            if max_length is None:
                max_length = max(150, min(512, len(input_tokens) // 2))

            min_len = max(50, min(max_length // 3, 150))

            # Generate summary using pipeline
            result = self.pipe(task_text, max_length=max_length, min_length=min_len)
            summary = result[0]['summary_text'] if result else ""

            # Ensure we never return empty - fallback to extractive summary
            if not summary or len(summary.strip()) < 20:
                self.logger.warning("Local model returned insufficient output, using extractive fallback")
                # Simple extractive fallback - take first few sentences
                sentences = original_text.replace('\n', ' ').split('. ')
                summary = '. '.join(sentences[:5]) + '.'
                if len(summary) > 500:
                    summary = summary[:500] + '...'

            self.logger.info(f"Summary generated using local model ({settings.SUMMARIZATION_MODEL}) - {len(summary)} chars")
            return summary

        except Exception as e:
            self.logger.error(f"Error in local summarization: {e}")
            # Last resort fallback - never return empty
            fallback = original_text[:500] + "..." if len(original_text) > 500 else original_text
            return f"Content Summary: {fallback}"

    def _summarize_rate_limit_fallback(self, text: str, style: str) -> str:
        """
        Try the light model when the main model hits a rate limit.
        Returns summary + rate limit notice, or None if fallback also fails.
        """
        light_model = getattr(self.llm_client, 'light_model', None) if self.llm_client else None
        if not light_model:
            return None

        try:
            # Truncate for light model (6K TPM — reserve ~1K for prompt + response)
            fallback_text = text[:12000]
            last_period = fallback_text.rfind('. ')
            if last_period > len(fallback_text) * 0.7:
                fallback_text = fallback_text[:last_period + 1]

            style_instruction = {
                'paragraph_summary': 'Write a clear summary in 2-3 paragraphs using flowing prose.',
                'important_points': 'List 8-10 important points as a numbered list (1. 2. 3.).',
                'key_highlights': 'List key terms with brief definitions using bullet points (•).'
            }.get(style, 'Write a clear summary in 2-3 paragraphs.')

            prompt = f"""{style_instruction}

Content:
{fallback_text}

Begin:"""

            summary = self.llm_client.generate(
                prompt=prompt,
                max_tokens=800,
                temperature=0.7,
                system_prompt="You are an expert educator who creates concise study notes.",
                model_override=light_model
            )

            if summary:
                self.logger.info(
                    f"Rate limit fallback successful using {light_model} "
                    f"({len(summary)} chars)"
                )
                notice = (
                    "\n\n"
                    "*Note: The primary model (llama-3.3-70b) hit its daily rate limit. "
                    "This summary was generated using a lighter model. "
                    "For full-quality notes, wait for the daily limit to reset.*"
                )
                return summary + notice

        except Exception as e:
            self.logger.warning(f"Rate limit fallback also failed: {e}")

        return None

    def summarize_documents(self, documents: List[str], style: str = "paragraph_summary", output_length: str = "auto", extra_instructions: str = None) -> str:
        """Summarize multiple documents."""
        try:
            # Combine documents
            combined = "\n\n---\n\n".join(documents[:5])  # Use top 5 documents

            # Summarize combined content
            return self.summarize_text(combined, style=style, output_length=output_length, extra_instructions=extra_instructions)

        except Exception as e:
            self.logger.error(f"Error summarizing documents: {e}")
            # Fallback to first document
            if documents:
                return self.summarize_text(documents[0])
            return ""

    def extract_bullet_points(self, text: str, num_points: int = 5) -> List[str]:
        """Extract key points as bullet points."""
        try:
            # Use important_points style for API
            summary = self.summarize_text(text, style="important_points")

            # Parse bullet points from response
            lines = summary.split('\n')
            bullet_points = []

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Remove bullet markers and clean
                if line.startswith(('-', '•', '*', '–')):
                    line = line.lstrip('-•*– ').strip()
                elif line[0].isdigit() and '.' in line[:3]:
                    # Handle numbered lists like "1. point"
                    line = line.split('.', 1)[-1].strip()

                if line and len(line) > 10:  # Skip very short lines
                    bullet_points.append(line)

            return bullet_points[:num_points]

        except Exception as e:
            self.logger.error(f"Error extracting bullet points: {e}")
            # Return summary as single point
            summary = self.summarize_text(text, style="paragraph_summary")
            return [summary] if summary else []

    def generate_flashcards(self, text: str, num_cards: int = 10) -> List[Dict[str, str]]:
        """
        Generate flashcards from text content.

        Args:
            text: Source text
            num_cards: Number of flashcards to generate

        Returns:
            List of flashcard dicts with 'front' and 'back' keys
        """
        try:
            if not self.use_local and self.llm_client:
                response = self.llm_client.generate_flashcards(text, num_cards)
                if response:
                    return self._parse_flashcards(response)

            # Fallback: Generate simple flashcards from bullet points
            points = self.extract_bullet_points(text, num_cards)
            flashcards = []
            for i, point in enumerate(points):
                flashcards.append({
                    'front': f"What is key concept #{i+1}?",
                    'back': point
                })
            return flashcards

        except Exception as e:
            self.logger.error(f"Error generating flashcards: {e}")
            return []

    def _parse_flashcards(self, response: str) -> List[Dict[str, str]]:
        """Parse flashcard response from LLM."""
        flashcards = []
        lines = response.split('\n')

        current_q = None
        for line in lines:
            line = line.strip()
            if line.startswith('Q:'):
                current_q = line[2:].strip()
            elif line.startswith('A:') and current_q:
                flashcards.append({
                    'front': current_q,
                    'back': line[2:].strip()
                })
                current_q = None

        return flashcards

    def generate_quiz(self, text: str, num_questions: int = 5) -> List[Dict[str, Any]]:
        """
        Generate quiz questions from text content.

        Args:
            text: Source text
            num_questions: Number of questions to generate

        Returns:
            List of quiz question dicts
        """
        try:
            if not self.use_local and self.llm_client:
                response = self.llm_client.generate_quiz(text, num_questions)
                if response:
                    return self._parse_quiz(response)

            # Fallback: Can't generate quiz with local model
            self.logger.warning("Quiz generation requires API mode")
            return []

        except Exception as e:
            self.logger.error(f"Error generating quiz: {e}")
            return []

    def _parse_quiz(self, response: str) -> List[Dict[str, Any]]:
        """Parse quiz response from LLM."""
        questions = []
        current_q = {}

        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('Question:'):
                if current_q:
                    questions.append(current_q)
                current_q = {'question': line[9:].strip(), 'options': {}}
            elif line.startswith(('A)', 'B)', 'C)', 'D)')):
                letter = line[0]
                current_q['options'][letter] = line[2:].strip()
            elif line.startswith('Correct:'):
                current_q['correct'] = line[8:].strip()
            elif line.startswith('Explanation:'):
                current_q['explanation'] = line[12:].strip()

        if current_q:
            questions.append(current_q)

        return questions

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process summarization request."""
        try:
            if not self.validate_input(input_data):
                return self.handle_error(ValueError("Invalid input"))

            content = input_data.get('content', '')
            mode = input_data.get('mode', 'paragraph_summary')
            output_length = input_data.get('output_length', 'auto')  # Only used for paragraph_summary
            extra_instructions = input_data.get('extra_instructions', None)

            if not content:
                return self.handle_error(ValueError("No content provided"))

            result = {
                'success': True,
                'agent': self.name,
                'provider': 'local' if self.use_local else self.llm_client.provider if self.llm_client else 'unknown'
            }

            if mode == 'important_points':
                # Use important_points style for numbered key points
                summary = self.summarize_text(content, style="important_points", extra_instructions=extra_instructions)
                result['summary'] = summary

            elif mode == 'key_highlights':
                # Use key_highlights style for brief term definitions
                summary = self.summarize_text(content, style="key_highlights", extra_instructions=extra_instructions)
                result['summary'] = summary

            elif mode == 'flashcards':
                num_cards = input_data.get('num_cards', 10)
                flashcards = self.generate_flashcards(content, num_cards)
                result['flashcards'] = flashcards
                result['summary'] = f"Generated {len(flashcards)} flashcards"

            elif mode == 'quiz':
                num_questions = input_data.get('num_questions', 5)
                quiz = self.generate_quiz(content, num_questions)
                result['quiz'] = quiz
                result['summary'] = f"Generated {len(quiz)} quiz questions"

            else:  # paragraph_summary mode (default)
                if isinstance(content, list):
                    summary = self.summarize_documents(content, style="paragraph_summary", output_length=output_length, extra_instructions=extra_instructions)
                else:
                    summary = self.summarize_text(content, style="paragraph_summary", output_length=output_length, extra_instructions=extra_instructions)
                result['summary'] = summary

            self.logger.info(f"Successfully processed content in {mode} mode, output_length={output_length}")
            return result

        except Exception as e:
            self.logger.error(f"Error in summarizer process: {e}")
            return {
                'success': False,
                'agent': self.name,
                'error': str(e),
                'summary': f"Error processing request: {str(e)}"
            }

    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about current provider."""
        # First check if we have an LLM client and what mode it's in
        if self.llm_client:
            client_info = self.llm_client.get_provider_info()
            # Return the actual client info (could be groq, huggingface, or local fallback)
            return client_info
        elif self.use_local:
            return {
                'provider': 'local',
                'model': settings.SUMMARIZATION_MODEL,
                'is_local': True
            }
        else:
            return {'provider': 'unknown', 'model': 'unknown', 'is_local': True}


# =============================================================================
# ORIGINAL LOCAL MODEL CODE (Preserved for reference)
# =============================================================================
# The following code is the original implementation using local models.
# It is kept here as reference and can be used by setting USE_LOCAL_MODEL=true
#
# """
# Summarization agent using Flan-T5 (Original Version)
# """
# from typing import Dict, Any, List
# import torch
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
# from langchain.llms import HuggingFacePipeline
# from langchain.chains.summarize import load_summarize_chain
# from langchain.docstore.document import Document
#
# from src.agents.base import BaseAgent
# from src.utils.cache_utils import cached
# from src.utils.text_utils import chunk_text
# from config import settings
#
# class SummarizerAgentLocal(BaseAgent):
#     """Agent for text summarization using local models"""
#
#     def __init__(self):
#         super().__init__("SummarizerAgent")
#         self.model_name = settings.SUMMARIZATION_MODEL
#         self._initialize_model()
#
#     def _initialize_model(self):
#         """Initialize the summarization model"""
#         try:
#             self.logger.info(f"Loading model: {self.model_name}")
#
#             # Load tokenizer and model
#             self.tokenizer = AutoTokenizer.from_pretrained(
#                 self.model_name,
#                 cache_dir=str(settings.MODEL_CACHE_DIR)
#             )
#
#             self.model = AutoModelForSeq2SeqLM.from_pretrained(
#                 self.model_name,
#                 cache_dir=str(settings.MODEL_CACHE_DIR),
#                 torch_dtype=torch.float32
#             )
#
#             # Create pipeline
#             self.pipe = pipeline(
#                 "summarization",
#                 model=self.model,
#                 tokenizer=self.tokenizer,
#                 max_length=1024,
#                 min_length=200,
#                 do_sample=True,
#                 early_stopping=True,
#                 device=-1  # CPU
#             )
#
#             # Create LangChain LLM
#             self.llm = HuggingFacePipeline(pipeline=self.pipe)
#
#             self.logger.info("Model loaded successfully")
#
#         except Exception as e:
#             self.logger.error(f"Error loading model: {e}")
#             raise
#
#     @cached("summarizer", ttl=settings.CACHE_SUMMARY_TTL)
#     def summarize_text(self, text: str, max_length: int = None) -> str:
#         """Summarize a single text"""
#         # ... (original implementation)
#         pass
#
#     def summarize_documents(self, documents: List[str]) -> str:
#         """Summarize multiple documents using LangChain"""
#         # ... (original implementation)
#         pass
#
#     def extract_bullet_points(self, text: str, num_points: int = 5) -> List[str]:
#         """Extract key points as bullet points"""
#         # ... (original implementation)
#         pass
#
#     async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
#         """Process summarization request"""
#         # ... (original implementation)
#         pass
# =============================================================================
