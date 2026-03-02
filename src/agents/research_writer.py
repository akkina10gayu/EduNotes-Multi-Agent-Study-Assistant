"""
Research Paper Writer Agent.
Guides users through iterative information gathering and generates
structured academic papers section by section.
"""
from typing import Dict, Any, List
from src.agents.base import BaseAgent
from src.utils.gemini_client import get_chat_provider
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------
SUFFICIENCY_PROMPT = """Analyze the following information for writing a {paper_type} paper on "{topic}".

PROVIDED INFORMATION:
{context}

Score the information sufficiency from 0.0 to 1.0:
- 0.0-0.3: Minimal, many critical gaps
- 0.3-0.6: Partial, important gaps remain
- 0.6-0.8: Mostly sufficient, minor gaps
- 0.8-1.0: Sufficient to write the paper

RESPOND IN THIS EXACT FORMAT:
SCORE: [number between 0.0 and 1.0]
AVAILABLE: [comma-separated list of what info IS available]
MISSING: [comma-separated list of what info is MISSING]
QUESTIONS:
1. [targeted follow-up question for the most critical gap]
2. [another follow-up question]
3. [another follow-up question]
ANALYSIS: [1-2 sentence summary of readiness]"""

OUTLINE_PROMPT = """Create a detailed outline for a {paper_type} paper on "{topic}".

AVAILABLE INFORMATION:
{context}

Generate a structured outline with main sections and subsections.
Use standard academic paper structure appropriate for a {paper_type} paper.

Return ONLY the outline, one item per line (e.g. "1. Abstract", "2. Introduction", "3.1 Dataset"):"""

SECTION_PROMPT = """Write the "{section}" section for a {paper_type} paper on "{topic}".

CONTEXT AND DATA:
{context}

FULL OUTLINE:
{outline}

PREVIOUSLY WRITTEN SECTIONS:
{previous_sections}
{additional_instructions}
Write this section in academic style with proper markdown formatting.
Use citations in [Author, Year] format where appropriate based on the context.
Be thorough but concise. Start writing the section content directly."""

ABSTRACT_PROMPT = """Write an abstract (150-250 words) for this research paper.
Summarize the research question, methodology, key results, and implications.

FULL PAPER:
{paper_content}

Write ONLY the abstract text, no heading:"""


class ResearchWriter(BaseAgent):
    """Research paper writing agent with iterative information gathering."""

    def __init__(self):
        super().__init__("ResearchWriter")
        self.provider = get_chat_provider()

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Route to the appropriate action handler."""
        action = input_data.get("action", "analyze")
        try:
            handler = {
                "analyze": self._analyze_sufficiency,
                "continue": self._continue_gathering,
                "outline": self._generate_outline,
                "section": self._generate_section,
                "assemble": self._assemble_paper,
            }.get(action)
            if not handler:
                return {"success": False, "error": f"Unknown action: {action}"}
            return await handler(input_data)
        except Exception as e:
            return self.handle_error(e)

    # ------------------------------------------------------------------
    # Stage 1: Sufficiency analysis
    # ------------------------------------------------------------------
    async def _analyze_sufficiency(self, data: Dict) -> Dict:
        topic = data.get("topic", "")
        context = data.get("context", "") or "(No additional context provided)"
        paper_type = data.get("paper_type", "research")

        prompt = SUFFICIENCY_PROMPT.format(
            paper_type=paper_type, topic=topic, context=context
        )
        result, provider = self.provider.generate(
            prompt=prompt,
            system_prompt="You are an academic advisor analyzing research paper readiness.",
            max_tokens=1024,
            temperature=0.3,
        )
        parsed = self._parse_sufficiency(result)
        related_papers = self._find_related_papers(topic)

        return {
            "success": True,
            "stage": "gathering",
            "sufficiency_score": parsed["score"],
            "info_available": parsed["available"],
            "info_missing": parsed["missing"],
            "questions": parsed["questions"],
            "analysis": parsed["analysis"],
            "related_papers": related_papers,
            "provider_used": provider,
        }

    # ------------------------------------------------------------------
    # Stage 2: Continue gathering (re-analyze with new answers)
    # ------------------------------------------------------------------
    async def _continue_gathering(self, data: Dict) -> Dict:
        topic = data.get("topic", "")
        paper_type = data.get("paper_type", "research")

        # Build full context from all gathered info
        context_parts = []
        if data.get("initial_context"):
            context_parts.append(f"Initial context: {data['initial_context']}")
        for key, value in data.get("gathered_info", {}).items():
            if value:
                context_parts.append(f"{key}: {value}")
        full_context = "\n\n".join(context_parts)

        # Re-analyze
        analysis = await self._analyze_sufficiency(
            {"topic": topic, "context": full_context, "paper_type": paper_type}
        )

        # If sufficient, also generate outline
        if analysis.get("sufficiency_score", 0) >= 0.7:
            outline = await self._generate_outline(
                {"topic": topic, "context": full_context, "paper_type": paper_type}
            )
            analysis["stage"] = "outlining"
            analysis["outline"] = outline.get("outline", [])

        return analysis

    # ------------------------------------------------------------------
    # Stage 3: Outline generation
    # ------------------------------------------------------------------
    async def _generate_outline(self, data: Dict) -> Dict:
        topic = data.get("topic", "")
        context = data.get("context", "")
        paper_type = data.get("paper_type", "research")

        prompt = OUTLINE_PROMPT.format(
            paper_type=paper_type, topic=topic, context=context
        )
        result, provider = self.provider.generate(
            prompt=prompt,
            system_prompt="You are an academic paper structure expert.",
            max_tokens=1024,
            temperature=0.3,
        )
        outline = [line.strip() for line in result.strip().split("\n") if line.strip()]

        return {"success": True, "stage": "outlining", "outline": outline}

    # ------------------------------------------------------------------
    # Stage 4: Section generation
    # ------------------------------------------------------------------
    async def _generate_section(self, data: Dict) -> Dict:
        topic = data.get("topic", "")
        section = data.get("section_name", "")
        context = data.get("context", "")
        paper_type = data.get("paper_type", "research")
        outline = data.get("outline", [])
        previous_sections = data.get("previous_sections", {})
        additional = data.get("additional_instructions", "")

        prev_text = "(First section)"
        if previous_sections:
            prev_text = "\n\n---\n\n".join(
                f"### {name}\n{content}"
                for name, content in previous_sections.items()
            )

        extra = f"\nAdditional instructions: {additional}" if additional else ""

        prompt = SECTION_PROMPT.format(
            section=section,
            paper_type=paper_type,
            topic=topic,
            context=context,
            outline="\n".join(outline) if outline else "(No outline)",
            previous_sections=prev_text,
            additional_instructions=extra,
        )
        result, provider = self.provider.generate(
            prompt=prompt,
            system_prompt="You are an expert academic writer producing publication-quality content.",
            max_tokens=3072,
            temperature=0.4,
        )

        return {
            "success": True,
            "stage": "writing",
            "section_name": section,
            "content": result,
        }

    # ------------------------------------------------------------------
    # Stage 5: Assembly
    # ------------------------------------------------------------------
    async def _assemble_paper(self, data: Dict) -> Dict:
        topic = data.get("topic", "")
        sections = data.get("sections", {})
        generate_abstract = data.get("generate_abstract", True)

        parts = [f"# {topic}\n"]

        # Generate abstract from the full content
        if generate_abstract and sections:
            full_content = "\n\n".join(sections.values())
            abstract_text, _ = self.provider.generate(
                prompt=ABSTRACT_PROMPT.format(paper_content=full_content[:8000]),
                system_prompt="You write concise, informative research paper abstracts.",
                max_tokens=512,
                temperature=0.3,
            )
            parts.append(f"## Abstract\n\n{abstract_text}\n")

        # Add all sections
        for section_name, content in sections.items():
            if section_name.lower() != "abstract":
                parts.append(f"\n{content}\n")

        full_paper = "\n".join(parts)
        word_count = len(full_paper.split())

        return {
            "success": True,
            "stage": "complete",
            "full_paper": full_paper,
            "word_count": word_count,
            "is_complete": True,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_sufficiency(text: str) -> Dict:
        """Parse the LLM's structured sufficiency analysis."""
        result = {
            "score": 0.0,
            "available": [],
            "missing": [],
            "questions": [],
            "analysis": "",
        }
        current_section = None
        for line in text.strip().split("\n"):
            line = line.strip()
            if line.upper().startswith("SCORE:"):
                try:
                    val = line.split(":", 1)[1].strip()
                    result["score"] = min(1.0, max(0.0, float(val)))
                except (ValueError, IndexError):
                    pass
            elif line.upper().startswith("AVAILABLE:"):
                items = line.split(":", 1)[1].strip()
                result["available"] = [i.strip() for i in items.split(",") if i.strip()]
            elif line.upper().startswith("MISSING:"):
                items = line.split(":", 1)[1].strip()
                result["missing"] = [i.strip() for i in items.split(",") if i.strip()]
            elif line.upper().startswith("QUESTIONS:"):
                current_section = "questions"
            elif line.upper().startswith("ANALYSIS:"):
                result["analysis"] = line.split(":", 1)[1].strip()
                current_section = None
            elif current_section == "questions" and line:
                q = line.lstrip("0123456789.-) ").strip()
                if q:
                    result["questions"].append(q)
        return result

    def _find_related_papers(self, topic: str) -> List[Dict]:
        """Search for related academic papers using the existing AcademicSearch."""
        try:
            from src.utils.academic_search import get_academic_search
            search = get_academic_search()
            papers = search.search_papers(topic, max_results=5)
            return papers if papers else []
        except Exception as e:
            logger.debug(f"Academic search skipped: {e}")
            return []
