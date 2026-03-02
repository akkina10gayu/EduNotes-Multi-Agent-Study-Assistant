"""
Conversational AI Agent for technical discussions, Q&A, and answer writing.
Uses Gemini/Cerebras (separate from the Groq summarization pipeline).
"""
from typing import Dict, Any, List
from src.agents.base import BaseAgent
from src.utils.gemini_client import get_chat_provider
from src.knowledge_base.vector_store import VectorStore
from src.utils.logger import get_logger
from config import settings

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# System prompts for each chat mode
# ---------------------------------------------------------------------------
SYSTEM_PROMPTS = {
    "chat": (
        "You are EduNotes AI, a knowledgeable study assistant. "
        "Help students understand technical concepts through clear explanations.\n\n"
        "Adapt your response format to the question:\n"
        "- For direct factual questions: start with a clear 1-2 sentence answer, "
        "then elaborate with details and an example.\n"
        "- For open-ended or discussion questions: be conversational, explore "
        "the topic thoroughly, and use examples to illustrate.\n"
        "- For 'how to' questions: provide step-by-step guidance.\n\n"
        "Always use markdown formatting for readability. "
        "If knowledge base context is provided below, reference it naturally."
    ),
    "answer_writer_brief": (
        "You are an exam answer writer helping students prepare for exams. "
        "The student will give you an exam question — write a model answer.\n\n"
        "Write a concise answer in ONE clear paragraph (100-150 words). "
        "Be direct, cover the essential points, use proper academic language, "
        "and structure it as a student would write in a timed exam."
    ),
    "answer_writer_standard": (
        "You are an exam answer writer helping students prepare for exams. "
        "The student will give you an exam question — write a model answer.\n\n"
        "Write a well-structured answer with:\n"
        "- Introduction: define key terms and state your approach (1-2 sentences)\n"
        "- Body: 2-3 paragraphs covering key points with examples and evidence\n"
        "- Conclusion: summarize the main takeaway (1-2 sentences)\n\n"
        "Target 400-600 words. Use clear academic English. "
        "Structure it as a model exam answer a student can learn from."
    ),
    "answer_writer_detailed": (
        "You are an exam answer writer helping students prepare for exams. "
        "The student will give you an exam question — write a comprehensive "
        "model answer.\n\n"
        "Write a thorough essay-style answer with:\n"
        "- Introduction: context, key definitions, and thesis (1 paragraph)\n"
        "- Multiple body sections with clear subheadings\n"
        "- Specific examples, evidence, and real-world applications in each section\n"
        "- Critical analysis — don't just describe, evaluate and compare\n"
        "- Conclusion: synthesize key insights and implications\n\n"
        "Target 800-1500 words. Demonstrate deep understanding. "
        "This should be a model answer worth full marks."
    ),
    "explain_eli5": (
        "You are explaining to a 5-year-old. Use very simple words, everyday "
        "analogies, and short sentences. NO jargon at all. Compare things to "
        "toys, animals, food, or games. Make it fun and memorable. "
        "Use emojis sparingly to keep it engaging."
    ),
    "explain_beginner": (
        "You are a patient teacher explaining to a complete beginner. "
        "Use simple, everyday language. Define ALL technical terms in plain "
        "English the first time you use them. Build understanding step by step — "
        "never assume prior knowledge. Include relatable real-world examples "
        "and analogies. End with a brief summary of the key takeaway."
    ),
    "explain_intermediate": (
        "You are explaining to someone with foundational knowledge of the field. "
        "Use proper technical terminology but explain complex or nuanced parts. "
        "Include practical examples showing how concepts work in practice. "
        "Connect ideas to related concepts the student likely already knows. "
        "Highlight common misconceptions where relevant."
    ),
    "explain_advanced": (
        "You are having a technical discussion with an advanced student or "
        "practitioner. Use precise technical language freely. Cover edge cases, "
        "trade-offs, and implementation details. Reference advanced concepts, "
        "recent developments, and seminal work in the field. Be thorough and "
        "rigorous — depth is more valued than simplicity here."
    ),
    "explain_analogy": (
        "Explain the concept using analogies from the domain of {domain}. "
        "Map EVERY key idea, component, and relationship to something concrete "
        "in {domain}. Build one coherent, extended analogy where all the parts "
        "relate to each other — not just isolated comparisons. After the analogy, "
        "briefly clarify where the analogy breaks down or simplifies."
    ),
    "explain_visual": (
        "Explain the concept using structured visual representations. For each "
        "key idea, choose the best visual format:\n"
        "- ASCII diagrams for architectures and flows\n"
        "- Markdown tables for comparisons and properties\n"
        "- Numbered step lists for processes\n"
        "- Indented tree structures for hierarchies\n\n"
        "EVERY visual MUST be followed by 1-2 sentences explaining what it shows. "
        "The visuals should tell the story — text is supplementary."
    ),
    "compare": (
        "You are comparing two concepts. BOTH concepts are EQUALLY important — "
        "give each one equal attention and depth.\n\n"
        "You MUST follow this EXACT structure:\n\n"
        "## 1. Overview\n"
        "Explain Concept A in 3-4 sentences, then Concept B in 3-4 sentences.\n\n"
        "## 2. Comparison Table\n"
        "Create a markdown table with at least 6 rows comparing specific aspects. "
        "Use this format:\n"
        "| Aspect | Concept A | Concept B |\n"
        "|--------|-----------|----------|\n"
        "| Purpose | ... | ... |\n"
        "| Origin / History | ... | ... |\n"
        "(continue with more rows)\n\n"
        "## 3. Key Differences\n"
        "Numbered list of the most important differences (at least 4).\n\n"
        "## 4. Similarities\n"
        "What do they have in common?\n\n"
        "## 5. When to Use Which\n"
        "Practical guidance on when each concept is the better choice.\n\n"
        "IMPORTANT: The comparison table is MANDATORY. Do NOT skip it."
    ),
    "socratic": (
        "You are a Socratic tutor. Your goal is to guide the student to "
        "discover the answer themselves through questions — NEVER give the "
        "answer directly.\n\n"
        "Method:\n"
        "1. Start with a thought-provoking question that targets the core of "
        "their query\n"
        "2. Based on their response, ask a more focused follow-up question\n"
        "3. If they're on the right track, give brief encouragement then ask "
        "a deeper question\n"
        "4. If they're stuck after 3-4 exchanges, give a hint (but NOT the "
        "full answer)\n"
        "5. Only reveal the complete answer if they explicitly say 'show answer' "
        "or 'tell me the answer'\n\n"
        "Keep your questions short and focused. Ask ONE question at a time. "
        "The student should be doing most of the thinking."
    ),
}

SUGGESTION_SUFFIX = (
    "\n\nAfter your response, add a blank line then write exactly "
    "\"---SUGGESTIONS---\" on its own line, followed by 3 brief follow-up "
    "questions the student might ask (one per line, no numbering)."
)

SUGGESTION_DELIMITER = "---SUGGESTIONS---"


class ChatAgent(BaseAgent):
    """Multi-mode conversational AI agent."""

    def __init__(self):
        super().__init__("ChatAgent")
        self.provider = get_chat_provider()
        self.vector_store = VectorStore()

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Main entry point (BaseAgent interface)."""
        try:
            message = input_data.get("message", "")
            mode = input_data.get("mode", "chat")
            history = input_data.get("history", [])
            use_kb = input_data.get("use_kb", True)

            # KB context injection
            kb_context, kb_sources = "", []
            if use_kb:
                kb_context, kb_sources = self._search_kb(message)

            # Build system prompt (with inline suggestion instruction)
            system_prompt = self._get_system_prompt(mode, input_data)
            if kb_context:
                system_prompt += (
                    "\n\nKNOWLEDGE BASE CONTEXT (use this to inform your answer):\n"
                    + kb_context
                )
            # Append suggestion instruction so it comes in ONE LLM call
            if mode != "socratic":
                system_prompt += SUGGESTION_SUFFIX

            # Build message list with history truncation
            max_history = int(getattr(settings, "CHAT_MAX_HISTORY", 50))
            messages = list(history[-max_history:])

            # Enhance message for compare mode
            if mode == "compare":
                concept2 = input_data.get("compare_concept_2", "")
                if concept2:
                    message = (
                        f"Concept A: {message}\n"
                        f"Concept B: {concept2}\n\n"
                        f"Compare and contrast these two concepts in full detail."
                    )

            messages.append({"role": "user", "content": message})

            # Call LLM (single call produces response + suggestions)
            raw_response, provider = self.provider.chat(
                messages=messages,
                system_prompt=system_prompt,
                max_tokens=self._get_max_tokens(mode, input_data),
                temperature=self._get_temperature(mode),
            )

            # Parse out inline suggestions from the response
            response_text, suggestions = self._parse_suggestions(raw_response, mode)

            return {
                "success": True,
                "response": response_text,
                "suggestions": suggestions,
                "sources": kb_sources,
                "provider_used": provider,
                "mode": mode,
            }
        except Exception as e:
            return self.handle_error(e)

    # ------------------------------------------------------------------
    # System prompt selection
    # ------------------------------------------------------------------
    def _get_system_prompt(self, mode: str, input_data: Dict) -> str:
        if mode == "answer_writer":
            depth = input_data.get("answer_depth", "standard")
            key = f"answer_writer_{depth}"
            return SYSTEM_PROMPTS.get(key, SYSTEM_PROMPTS["answer_writer_standard"])

        if mode == "explain":
            style = input_data.get("explain_style", "technical")
            level = input_data.get("explain_level", "intermediate")

            if style == "analogy":
                domain = input_data.get("analogy_domain", "everyday life")
                return SYSTEM_PROMPTS["explain_analogy"].replace("{domain}", domain)
            if style == "visual":
                return SYSTEM_PROMPTS["explain_visual"]
            key = f"explain_{level}"
            return SYSTEM_PROMPTS.get(key, SYSTEM_PROMPTS["explain_intermediate"])

        return SYSTEM_PROMPTS.get(mode, SYSTEM_PROMPTS["chat"])

    # ------------------------------------------------------------------
    # Token and temperature tuning per mode
    # ------------------------------------------------------------------
    @staticmethod
    def _get_max_tokens(mode: str, input_data: Dict) -> int:
        if mode == "answer_writer":
            depth = input_data.get("answer_depth", "standard")
            return {"brief": 512, "standard": 1536, "detailed": 3072}.get(depth, 1536)
        if mode == "compare":
            return 2560
        if mode == "socratic":
            return 512
        return 2048

    @staticmethod
    def _get_temperature(mode: str) -> float:
        if mode == "socratic":
            return 0.8
        if mode == "compare":
            return 0.4
        return 0.7

    # ------------------------------------------------------------------
    # Knowledge base context injection
    # ------------------------------------------------------------------
    def _search_kb(self, query: str) -> tuple:
        """Search the knowledge base for relevant context chunks."""
        try:
            k = int(getattr(settings, "CHAT_CONTEXT_CHUNKS", 5))
            results = self.vector_store.search(query, k=k, score_threshold=1.0)
            if not results:
                return "", []

            context_parts = []
            sources = []
            seen = set()
            for i, doc in enumerate(results, 1):
                context_parts.append(f"[{i}] {doc['content']}")
                url = doc["metadata"].get("url", "")
                title = doc["metadata"].get("title", "Knowledge Base")
                identifier = url or title
                if identifier not in seen:
                    seen.add(identifier)
                    source = {
                        "title": title,
                        "type": "kb",
                        "score": round(doc["score"], 3),
                    }
                    if url:
                        source["url"] = url
                    sources.append(source)

            return "\n\n".join(context_parts), sources
        except Exception as e:
            logger.warning(f"KB search failed: {e}")
            return "", []

    # ------------------------------------------------------------------
    # Parse inline suggestions from LLM response (saves a second API call)
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_suggestions(raw_response: str, mode: str) -> tuple:
        """Split the LLM response into (clean_response, suggestions_list)."""
        if mode == "socratic" or SUGGESTION_DELIMITER not in raw_response:
            return raw_response.strip(), []

        parts = raw_response.split(SUGGESTION_DELIMITER, 1)
        clean_response = parts[0].strip()
        suggestion_block = parts[1].strip() if len(parts) > 1 else ""

        suggestions = [
            line.strip().lstrip("0123456789.-•) ")
            for line in suggestion_block.split("\n")
            if line.strip() and len(line.strip()) > 10
        ]
        return clean_response, suggestions[:3]
