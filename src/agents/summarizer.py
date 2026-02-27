"""
Summarization agent with FREE API support (Groq, HuggingFace)

Version 2.0 - API-based approach with rate-limit fallback
"""
from typing import Dict, Any, List
from src.agents.base import BaseAgent
from src.utils.cache_utils import cached
from src.utils.llm_client import get_llm_client


class SummarizerAgent(BaseAgent):
    """
    Agent for text summarization using FREE LLM APIs.

    Supports:
    - Groq API (primary, free, fastest)
    - HuggingFace Inference API (backup, free)
    """

    def __init__(self):
        super().__init__("SummarizerAgent")
        self.llm_client = None
        self._initialize()

    def _initialize(self):
        """Initialize the LLM client for summarization."""
        try:
            self.logger.info("Initializing LLM client for summarization")
            self.llm_client = get_llm_client()
            self.logger.info(f"LLM Provider: {self.llm_client.get_provider_info()}")
        except Exception as e:
            self.logger.error(f"Error initializing summarizer: {e}")

    @cached("summarizer", ttl=3600)
    def summarize_text(self, text: str, max_length: int = None, style: str = "paragraph_summary", output_length: str = "auto", extra_instructions: str = None) -> str:
        """
        Summarize text using LLM API.

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

            # Check if LLM is available
            if not self.llm_client or not self.llm_client.is_available():
                return (
                    "**No LLM Available**\n\n"
                    "Summarization requires an LLM API provider. "
                    "No provider could be initialized.\n\n"
                    "**Setup instructions:**\n"
                    "1. Get a free API key from [console.groq.com](https://console.groq.com)\n"
                    "2. Add `GROQ_API_KEY=your_key` to your `.env` file\n"
                    "3. Restart the application\n"
                )

            try:
                summary = self.llm_client.summarize(
                    text=text,
                    max_length=max_length or 3072,
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
                self.logger.error(f"API summarization failed: {e}")

            return (
                "**Summarization Failed**\n\n"
                "The LLM API returned an error. Please try again later.\n"
            )

        except Exception as e:
            self.logger.error(f"Error summarizing text: {e}")
            return f"Summary of content about {text[:100]}... [Error: {str(e)}]"

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
            if self.llm_client and self.llm_client.is_available():
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
            if self.llm_client and self.llm_client.is_available():
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
                'provider': self.llm_client.provider if self.llm_client and self.llm_client.is_available() else 'none'
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
        if self.llm_client:
            return self.llm_client.get_provider_info()
        return {'provider': 'none', 'model': 'none', 'available': False}
