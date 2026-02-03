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
        style: str = "paragraph_summary",
        output_length: str = "auto"
    ) -> str:
        """
        Summarize text using the LLM.

        Args:
            text: Text to summarize
            max_length: Maximum summary length in tokens
            style: 'paragraph_summary', 'important_points', or 'key_highlights'
            output_length: 'auto', 'detailed', 'medium', or 'brief' (only for paragraph_summary)

        Returns:
            Summary string
        """
        if self.provider == "local":
            return None  # Signal to use local model

        # For paragraph_summary, determine actual length if auto mode
        actual_length = output_length
        if style == "paragraph_summary" and output_length == "auto":
            input_chars = len(text)
            if input_chars < 1000:
                actual_length = "brief"
            elif input_chars < 5000:
                actual_length = "medium"
            else:
                actual_length = "detailed"
            logger.info(f"Auto mode: input={input_chars} chars -> output_length='{actual_length}'")

        logger.info(f"Building prompt for style='{style}', output_length='{actual_length}', text_length={len(text)} chars")

        # Optimize max_tokens based on style and output_length (Phase 3 optimization)
        # This reduces token generation time significantly
        if style == "important_points":
            optimized_max_tokens = 1500  # 8-12 numbered points don't need 3072
        elif style == "key_highlights":
            optimized_max_tokens = 1500  # Bullet definitions are concise
        elif style == "paragraph_summary":
            # Vary based on requested output length
            tokens_by_length = {
                "brief": 512,
                "medium": 1024,
                "detailed": 2048
            }
            optimized_max_tokens = tokens_by_length.get(actual_length, 2048)
        else:
            optimized_max_tokens = max_length  # Fallback to passed value

        logger.info(f"Optimized max_tokens: {optimized_max_tokens} (style={style}, length={actual_length})")

        # Build prompt based on style
        if style == "key_highlights":
            # KEY HIGHLIGHTS: Key terms, terminology, topics with very brief descriptions
            system_prompt = """You are an expert at extracting key terminology, topics, and important items from educational content. You identify ALL important elements: technical terms, concepts, methods, techniques, formulas, acronyms, named entities, tools, and key topics. You create comprehensive, scannable glossary-style lists with brief definitions."""

            prompt = f"""Extract ALL KEY HIGHLIGHTS from this content. Create a comprehensive quick-reference list covering everything important.

WHAT TO EXTRACT (be thorough):
- Technical TERMINOLOGY and definitions
- Key CONCEPTS and ideas
- METHODS, techniques, and approaches mentioned
- FORMULAS, equations, or algorithms
- ACRONYMS and abbreviations (with full form)
- Important NAMES (people, tools, frameworks, models)
- KEY TOPICS and subtopics discussed
- Any NUMERICAL VALUES or statistics that are significant

FORMAT RULES:
- START DIRECTLY with "•" - no introduction or headers
- Each item = term/topic followed by colon and BRIEF explanation (under 15 words)
- Extract 10-20 highlights to be comprehensive
- Cover ALL categories above, not just concepts
- NO duplicates

OUTPUT FORMAT:
• [Term/Topic]: [Brief 5-15 word explanation]

EXAMPLE OUTPUT (notice the variety):
• Neural Network: Computing system with interconnected nodes processing information in layers
• CNN (Convolutional Neural Network): Architecture specialized for image and spatial data processing
• Backpropagation: Algorithm adjusting weights by propagating errors backward through layers
• ReLU: Activation function returning max(0, x), solving vanishing gradient problem
• ImageNet: Large-scale dataset with 14 million images used for benchmarking
• 95.6% Accuracy: State-of-the-art performance achieved by ResNet on ImageNet
• Transfer Learning: Technique of reusing pre-trained models for new tasks
• Dropout: Regularization method randomly disabling neurons during training
• Batch Normalization: Technique normalizing layer inputs to accelerate training
• PyTorch: Popular deep learning framework developed by Facebook

BAD OUTPUT (DO NOT DO THIS):
• Neural networks are computational systems inspired by biological neural networks. They consist of layers of interconnected nodes...
[Too long! Keep each item brief]

---
CONTENT:
{text}

---
START WITH "•" NOW (no introduction):"""

        elif style == "important_points":
            # IMPORTANT POINTS: Independent key points, numbered, no duplicates
            system_prompt = """You are an expert at extracting key information. You ONLY output numbered lists. You NEVER write introductions, headers, or explanations. You create your own fresh numbering starting from 1, completely ignoring any existing numbers in the source. You start your response directly with "1." and continue with numbered points only."""

            prompt = f"""Extract the important points from this content as a numbered list.

CRITICAL RULES:
- START YOUR RESPONSE DIRECTLY WITH "1." - no introduction, no headers, no preamble
- IGNORE any existing numbering, section numbers, or list numbers in the source content
- Create YOUR OWN fresh numbering: 1, 2, 3, 4... (independent of source)
- ONLY output numbered points (1. 2. 3. etc.)
- Each point: 1-3 sentences, independent, self-contained
- NO duplicates - each point must be unique information
- Extract 8-12 points
- DO NOT write any text before "1." or after the last point

IMPORTANT: The source content may have its own numbered sections (like "Section 1", "Chapter 2", etc.).
COMPLETELY IGNORE those numbers. Your output numbering is SEPARATE and must start from 1.

OUTPUT FORMAT (START EXACTLY LIKE THIS):
1. [First point here]

2. [Second point here]

3. [Third point here]

WRONG (DO NOT DO THIS):
- Starting from 13. or any number other than 1.
- "Here are the important points:"
- "The key points from this content are:"
- Continuing numbering from the source document
[Any text before "1." is WRONG]

CORRECT (DO THIS - always start from 1):
1. Machine learning algorithms improve their performance automatically through experience, without being explicitly programmed for specific tasks.

2. Supervised learning requires labeled training data where each input is paired with the correct output.

3. Neural networks consist of layers of interconnected nodes that transform input data through weighted connections.

---
CONTENT:
{text}

---
YOUR NUMBERED LIST (start fresh from 1, ignore any numbers in the content above):
1."""

        else:  # paragraph_summary (default)
            # PARAGRAPH SUMMARY: Overview in flowing paragraphs with length control
            system_prompt = """You are an expert educator who explains complex topics clearly. Your task is to write a well-structured summary that flows naturally in paragraph form. You write in complete sentences, use smooth transitions between ideas, and ensure each paragraph covers a coherent theme. You NEVER use bullet points or lists - only flowing prose. You strictly follow the requested length."""

            # LENGTH-BASED SUB-PROMPTS
            length_instructions = {
                "brief": """LENGTH REQUIREMENT - BRIEF (STRICTLY FOLLOW):
- MAXIMUM 5-6 sentences total - NO MORE
- Write as ONE short paragraph only
- Include ONLY the most critical information - ruthlessly filter out everything else
- Each sentence must pack maximum information
- This is an executive summary - extremely condensed

WHAT TO INCLUDE (pick only the essentials):
1. What is the topic/concept about (1-2 sentences)
2. The core method, approach, or key idea (1-2 sentences)
3. Main result, conclusion, or takeaway (1-2 sentences)

BRIEF EXAMPLE (notice: only 5 sentences, one paragraph):
Machine learning enables computers to learn patterns from data without explicit programming. The two main approaches are supervised learning using labeled examples and unsupervised learning for discovering hidden patterns. Deep neural networks have revolutionized the field, achieving human-level performance in image and speech recognition. The key challenge remains preventing overfitting while ensuring models generalize well to new data. Recent advances have pushed accuracy above 95% on major benchmarks.

WRONG - TOO LONG:
[Any output with more than 6 sentences is TOO LONG. Cut it down.]""",

                "medium": """LENGTH REQUIREMENT - MEDIUM (STRICTLY FOLLOW):
- EXACTLY 2-3 paragraphs - NO MORE than 3 paragraphs
- Each paragraph: MAXIMUM 4-5 sentences (not more)
- Total sentences: 10-15 sentences across all paragraphs
- DO NOT write 4, 5, 6 or more paragraphs - that is TOO LONG
- Cover main concepts with moderate detail, skip minor details

STRUCTURE FOR MEDIUM (strictly 2-3 paragraphs):
**Paragraph 1 (4-5 sentences):** Introduction - what is it, why it matters, main concept
**Paragraph 2 (4-5 sentences):** Core details - how it works, key methods or approaches
**Paragraph 3 (3-4 sentences, optional):** Applications or conclusions - if space allows

MEDIUM EXAMPLE (notice: exactly 3 paragraphs, 4 sentences each):
Machine learning is a branch of AI where systems learn from data rather than following explicit instructions. This approach has transformed how we solve complex problems across many domains. At its core, ML algorithms identify patterns in training data and use them to make predictions on new data. The field has grown rapidly due to increased computing power and data availability.

The main approaches include supervised learning with labeled data for classification tasks, and unsupervised learning for discovering hidden patterns. Deep learning uses multi-layered neural networks that have achieved remarkable results in image and speech recognition. These models automatically learn hierarchical features from raw data. Training requires large datasets and significant computational resources.

ML now powers recommendation systems, fraud detection, medical diagnosis, and autonomous vehicles. The technology continues advancing with new architectures and training techniques. Challenges remain in interpretability, fairness, and data efficiency.

WRONG - TOO LONG (DO NOT DO THIS):
[Any output with more than 3 paragraphs is TOO LONG for medium. If you wrote 4+ paragraphs, you wrote too much. Cut it down to exactly 2-3 paragraphs.]""",

                "detailed": """LENGTH REQUIREMENT - DETAILED (STRICTLY FOLLOW):
- Write 4-6 substantial paragraphs
- Each paragraph should be 5-7 sentences with thorough explanations
- Cover all major concepts comprehensively
- Include definitions, explanations, examples, and connections
- Provide depth and context for full understanding

STRUCTURE FOR DETAILED:
- Paragraph 1: Introduction, context, and significance (5-6 sentences)
- Paragraphs 2-3: Core concepts explained in depth (5-7 sentences each)
- Paragraph 4: Methods, techniques, or mechanisms (5-6 sentences)
- Paragraph 5: Applications, examples, and real-world impact (4-5 sentences)
- Paragraph 6: Synthesis, challenges, or future directions (3-4 sentences)"""
            }

            # Get the appropriate length instruction (default to detailed)
            length_instruction = length_instructions.get(actual_length, length_instructions["detailed"])

            prompt = f"""Write a PARAGRAPH SUMMARY of this content following the specified length requirement.

{length_instruction}

FORMAT RULES (ALWAYS FOLLOW):
- Write in COMPLETE PARAGRAPHS only - NO bullet points, NO numbered lists, NO dashes
- Each sentence must flow into the next using transitions (Furthermore, Additionally, However, etc.)
- Paragraphs should be CONTINUOUS PROSE, not disconnected statements
- Start writing directly - no headers or introductions like "Here is a summary:"

BAD OUTPUT (NEVER DO THIS):
- Using bullet points
- Writing single disconnected sentences
- Adding headers like "Summary:" or "Overview:"
• Using any kind of list format

---
CONTENT TO SUMMARIZE:
{text}

---
NOW write the paragraph summary (flowing prose, no bullets, follow the length requirement):"""

        try:
            logger.info(f"Sending to LLM - Provider: {self.provider}, Style: {style}, max_tokens: {optimized_max_tokens}, Prompt length: {len(prompt)} chars")
            result = self.generate(
                prompt=prompt,
                max_tokens=optimized_max_tokens,  # Use optimized value instead of max_length
                temperature=0.7,
                system_prompt=system_prompt
            )
            logger.info(f"LLM response received - Length: {len(result) if result else 0} chars")
            return result
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
