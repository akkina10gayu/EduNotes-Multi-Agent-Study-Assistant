"""
ContentAgent - Intelligent content processing agent.

Wraps SummarizerAgent as its processing tool, adding intelligence around it:

Pipeline (4 stages):
    Input Content
        -> [Stage 1: Content Type Analysis]     (LLM - classify content)
        -> [Stage 2: Strategy Selection]        (Rule-based - build preservation instructions)
        -> [Stage 3: Content Processing]        (SummarizerAgent - existing pipeline)
        -> [Stage 4: Self-Evaluation]           (LLM - quality check + gap detection)
        -> Return result (with optional gap_queries for Orchestrator)

Flashcards and quiz modes bypass Stages 1-2-4 and delegate directly to SummarizerAgent.
"""
import hashlib
from typing import Dict, Any

from src.agents.base import BaseAgent
from src.agents.summarizer import SummarizerAgent
from src.utils.llm_client import get_llm_client
from src.utils.cache_utils import cache, get_cache_key


class ContentAgent(BaseAgent):
    """
    Intelligent content processing agent with content-type awareness.

    Uses LLM reasoning at 2 decision points:
    1. Content Type Analysis: Classifies content for tailored processing
    2. Self-Evaluation: Checks output quality and identifies knowledge gaps

    Uses SummarizerAgent as its processing TOOL, LLM as the BRAIN.
    """

    def __init__(self):
        super().__init__("ContentAgent")
        self.llm_client = get_llm_client()
        self.summarizer = SummarizerAgent()

    # =================================================================
    # STAGE 1: CONTENT TYPE ANALYSIS (LLM-powered)
    # =================================================================

    def analyze_content_type(self, content: str) -> str:
        """
        Use LLM to classify content into a category for tailored processing.

        Categories: academic, tutorial, research, general, reference

        Returns 'general' as fallback in local mode or on error.
        """
        # Local mode fallback - skip LLM
        if self.llm_client.is_local_mode():
            self.logger.info("Stage 1: Local mode - defaulting to 'general'")
            return "general"

        # Cache check by content hash (24h TTL)
        content_hash = hashlib.md5(content[:2000].encode()).hexdigest()
        cache_key = get_cache_key("ca_type", content_hash)
        cached_type = cache.get(cache_key)
        if cached_type is not None:
            self.logger.info(f"Stage 1 cache HIT: {cached_type}")
            return cached_type

        preview = content[:1000]

        prompt = f"""Classify this content into ONE category: academic, tutorial, research, general, reference.

- academic: Papers, theses, scholarly content with citations/methodology
- tutorial: Step-by-step guides, how-to, code examples
- research: Technical reports, experiments, metrics, benchmarks
- general: News, blog posts, overview articles
- reference: Documentation, API docs, specifications

CONTENT PREVIEW:
{preview}

CATEGORY (one word):"""

        try:
            response = self.llm_client.generate(
                prompt=prompt,
                max_tokens=50,
                temperature=0.1,
                model_override=getattr(self.llm_client, 'light_model', None)
            )

            if not response:
                return "general"

            # Parse response - extract the category word
            response_clean = response.strip().lower()
            valid_types = {"academic", "tutorial", "research", "general", "reference"}

            for vtype in valid_types:
                if vtype in response_clean:
                    self.logger.info(f"Stage 1: Content classified as '{vtype}'")
                    cache.set(cache_key, vtype, expire=86400)
                    return vtype

            self.logger.warning(f"Stage 1: Unrecognized type '{response_clean}', defaulting to 'general'")
            return "general"

        except Exception as e:
            self.logger.error(f"Stage 1 error: {e}, defaulting to 'general'")
            return "general"

    # =================================================================
    # STAGE 2: STRATEGY SELECTION (Rule-based, no LLM)
    # =================================================================

    def select_strategy(self, content_type: str) -> str:
        """
        Return content-specific preservation instructions based on content type.

        Pure rule-based mapping - no LLM call needed.
        Returns empty string for 'general' (standard processing).
        """
        strategies = {
            "academic": (
                "CRITICAL FORMAT OVERRIDES (these OVERRIDE any paragraph-only or no-lists rules above):\n"
                "- EXCEPTION: Markdown tables and equations are DATA, not formatting. Include them verbatim.\n"
                "- Reproduce markdown tables EXACTLY as they appear (use | column | format |)\n"
                "- Keep ALL equations in their original LaTeX notation (e.g., $E=mc^2$ or $$\\sum_{i=1}^n$$)\n"
                "- Preserve citation references like [1], [Author, Year], etc.\n"
                "- Keep the abstract/introduction/methodology/results structure\n"
                "- Include author attributions and paper references with titles\n"
                "- Do NOT convert tables to prose - keep them as markdown tables\n"
                "- Do NOT paraphrase equations - keep original notation\n"
                "- Embed tables and equations WITHIN your paragraphs where relevant"
            ),
            "tutorial": (
                "CRITICAL FORMAT OVERRIDES (these OVERRIDE any paragraph-only or no-lists rules above):\n"
                "- EXCEPTION: Code blocks and step sequences are DATA, not formatting. Include them verbatim.\n"
                "- Preserve code blocks exactly as written using ```language fences\n"
                "- Keep step numbering (1. 2. 3.) intact\n"
                "- Maintain command-line examples with $ prefix\n"
                "- Keep configuration snippets and syntax examples in code blocks\n"
                "- Preserve prerequisites and dependencies lists\n"
                "- Do NOT merge separate code examples into one block"
            ),
            "research": (
                "CRITICAL FORMAT OVERRIDES (these OVERRIDE any paragraph-only or no-lists rules above):\n"
                "- EXCEPTION: Tables, equations, and metrics are DATA, not formatting. Include them verbatim.\n"
                "- Reproduce ALL markdown tables EXACTLY using | column | format |\n"
                "- Keep ALL numerical results, metrics, and statistics exact (precision matters)\n"
                "- Preserve formulas and equations in their original notation\n"
                "- Keep benchmark comparison tables as tables, NOT as prose\n"
                "- Preserve experimental conditions, parameters, and measurement units\n"
                "- Include figure/table references (e.g., 'as shown in Table 1', 'Figure 3')\n"
                "- Do NOT round numbers or approximate percentages\n"
                "- Do NOT convert tabular data into paragraph text\n"
                "- Embed tables and equations WITHIN your text where relevant"
            ),
            "reference": (
                "CRITICAL FORMAT OVERRIDES (these OVERRIDE any paragraph-only or no-lists rules above):\n"
                "- EXCEPTION: API signatures, parameter tables, and code are DATA. Include them verbatim.\n"
                "- Preserve exact definitions and API signatures in code blocks\n"
                "- Keep parameter tables as markdown tables\n"
                "- Maintain version numbers and compatibility notes\n"
                "- Preserve syntax examples in code blocks with language tags\n"
                "- Keep structured format (headings, lists, tables) of documentation\n"
                "- Do NOT paraphrase technical specifications"
            ),
            "general": ""
        }

        strategy = strategies.get(content_type, "")
        if strategy:
            self.logger.info(f"Stage 2: Applied '{content_type}' preservation strategy")
        else:
            self.logger.info("Stage 2: Standard processing (no extra instructions)")
        return strategy

    # =================================================================
    # STAGE 3: CONTENT PROCESSING (Delegates to SummarizerAgent)
    # =================================================================

    # (Handled inline in process() - calls self.summarizer.process())

    # =================================================================
    # STAGE 4: SELF-EVALUATION (LLM-powered)
    # =================================================================

    def evaluate_output(self, content: str, summary: str, content_type: str) -> Dict[str, Any]:
        """
        Use LLM to assess summary completeness and identify gaps.

        Returns:
            {
                'quality': 'pass' | 'needs_improvement',
                'gaps': [],       # Topics to search for if quality fails
                'reasoning': ''   # Brief explanation
            }
        """
        # Skip conditions
        if self.llm_client.is_local_mode():
            self.logger.info("Stage 4: Local mode - skipping evaluation")
            return {"quality": "pass", "gaps": [], "reasoning": "Local mode - evaluation skipped"}

        if not summary or len(summary) < 100:
            self.logger.info("Stage 4: Summary too short - skipping evaluation")
            return {"quality": "pass", "gaps": [], "reasoning": "Summary too short for evaluation"}

        content_preview = content[:2000]

        prompt = f"""You are evaluating a summary's completeness. Compare the original content with the generated summary.

ORIGINAL CONTENT (preview):
{content_preview}

GENERATED SUMMARY:
{summary}

CONTENT TYPE: {content_type}

Are there any CRITICAL topics, concepts, or data points from the original that are completely missing from the summary?

Respond in this format:
QUALITY: PASS or NEEDS_IMPROVEMENT
GAPS: topic1, topic2, topic3 (or NONE)
REASON: Brief explanation"""

        try:
            response = self.llm_client.generate(
                prompt=prompt,
                max_tokens=200,
                temperature=0.1,
                model_override=getattr(self.llm_client, 'light_model', None)
            )

            if not response:
                return {"quality": "pass", "gaps": [], "reasoning": "No LLM response"}

            return self._parse_evaluation(response)

        except Exception as e:
            self.logger.error(f"Stage 4 error: {e}")
            return {"quality": "pass", "gaps": [], "reasoning": f"Evaluation error: {e}"}

    def _parse_evaluation(self, response: str) -> Dict[str, Any]:
        """Parse the LLM evaluation response into structured data."""
        result = {"quality": "pass", "gaps": [], "reasoning": ""}

        lines = response.strip().split('\n')
        for line in lines:
            line = line.strip()
            upper = line.upper()

            if upper.startswith("QUALITY:"):
                value = line.split(":", 1)[1].strip().upper()
                if "NEEDS_IMPROVEMENT" in value or "NEEDS IMPROVEMENT" in value:
                    result["quality"] = "needs_improvement"
                else:
                    result["quality"] = "pass"

            elif upper.startswith("GAPS:"):
                value = line.split(":", 1)[1].strip()
                if value.upper() != "NONE" and value:
                    result["gaps"] = [g.strip() for g in value.split(",") if g.strip()]

            elif upper.startswith("REASON:"):
                result["reasoning"] = line.split(":", 1)[1].strip()

        self.logger.info(
            f"Stage 4: Quality={result['quality']}, "
            f"Gaps={result['gaps']}, Reason={result['reasoning']}"
        )
        return result

    # =================================================================
    # MAIN PROCESS METHOD (Full Pipeline)
    # =================================================================

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process content through the intelligent 4-stage pipeline.

        Input:
            {
                'content': str,
                'mode': str,           # paragraph_summary, important_points, key_highlights, flashcards, quiz
                'output_length': str   # auto, brief, medium, detailed
            }

        Output extends SummarizerAgent's output with:
            - 'content_type': detected content type
            - 'needs_more_info': whether gap resolution is needed
            - 'gap_queries': topics to search for
        """
        try:
            if not self.validate_input(input_data):
                return self.handle_error(ValueError("Invalid input"))

            content = input_data.get('content', '')
            mode = input_data.get('mode', 'paragraph_summary')
            output_length = input_data.get('output_length', 'auto')
            skip_evaluation = input_data.get('skip_evaluation', False)

            if not content:
                return self.handle_error(ValueError("No content provided"))

            # Flashcards/quiz mode: bypass intelligence, delegate directly
            if mode in ('flashcards', 'quiz'):
                self.logger.info(f"Mode '{mode}' - delegating directly to SummarizerAgent")
                result = await self.summarizer.process(input_data)
                result['agent'] = self.name
                result['content_type'] = 'general'
                result['needs_more_info'] = False
                result['gap_queries'] = []
                return result

            # Summarization modes: full 4-stage pipeline
            self.logger.info(f"Processing content ({len(content)} chars) in '{mode}' mode")

            # Stage 1: Content Type Analysis (LLM)
            self.logger.info("Stage 1: Analyzing content type...")
            content_type = self.analyze_content_type(content)

            # Stage 2: Strategy Selection (rule-based)
            self.logger.info("Stage 2: Selecting processing strategy...")
            extra_instructions = self.select_strategy(content_type)

            # Stage 3: Content Processing (SummarizerAgent)
            self.logger.info("Stage 3: Processing content via SummarizerAgent...")
            summarizer_input = {
                'content': content,
                'mode': mode,
                'output_length': output_length
            }
            if extra_instructions:
                summarizer_input['extra_instructions'] = extra_instructions

            result = await self.summarizer.process(summarizer_input)

            if not result.get('success'):
                result['agent'] = self.name
                result['content_type'] = content_type
                result['needs_more_info'] = False
                result['gap_queries'] = []
                return result

            summary = result.get('summary', '')

            # Stage 4: Self-Evaluation (LLM)
            # Skip for authoritative content (text/PDF/URL) - evaluation would
            # incorrectly flag "gaps" by comparing full content with shorter summary
            result['agent'] = self.name
            result['content_type'] = content_type
            result['provider'] = result.get('provider', 'unknown')

            if skip_evaluation:
                self.logger.info("Stage 4: Skipped (authoritative content, no gap resolution needed)")
                result['needs_more_info'] = False
                result['gap_queries'] = []
            else:
                self.logger.info("Stage 4: Evaluating output quality...")
                eval_result = self.evaluate_output(content, summary, content_type)

                if eval_result["quality"] == "needs_improvement" and eval_result["gaps"]:
                    result['needs_more_info'] = True
                    result['gap_queries'] = eval_result["gaps"]
                    self.logger.info(f"Content gaps identified: {eval_result['gaps']}")
                else:
                    result['needs_more_info'] = False
                    result['gap_queries'] = []

            self.logger.info(
                f"ContentAgent complete: type={content_type}, "
                f"gaps={len(result.get('gap_queries', []))}"
            )
            return result

        except Exception as e:
            self.logger.error(f"Error in ContentAgent process: {e}")
            return {
                'success': False,
                'agent': self.name,
                'error': str(e),
                'summary': f"Error processing request: {str(e)}",
                'content_type': 'general',
                'needs_more_info': False,
                'gap_queries': []
            }

    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about current provider (delegates to summarizer)."""
        return self.summarizer.get_provider_info()
