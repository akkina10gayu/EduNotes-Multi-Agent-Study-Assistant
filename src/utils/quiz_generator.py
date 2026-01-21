"""
Quiz Generator for EduNotes
Generates quizzes from text content using LLM
"""
import re
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

from config import settings
from src.models.quiz import Quiz, QuizQuestion, QuizAttempt, QuestionType
from src.utils.llm_client import get_llm_client
from src.utils.logger import get_logger

logger = get_logger(__name__)


class QuizStore:
    """Manages quiz storage and retrieval"""

    def __init__(self, storage_path: Path = None):
        """Initialize the quiz store"""
        self.storage_path = storage_path or (settings.DATA_DIR / "quizzes")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.index_file = self.storage_path / "index.json"
        self._ensure_index()

    def _ensure_index(self):
        """Ensure the index file exists"""
        if not self.index_file.exists():
            self._save_index({})

    def _load_index(self) -> Dict[str, Any]:
        """Load the index"""
        try:
            with open(self.index_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}

    def _save_index(self, index: Dict[str, Any]):
        """Save the index"""
        with open(self.index_file, 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=2, default=str)

    def save_quiz(self, quiz: Quiz) -> bool:
        """Save a quiz to storage"""
        try:
            quiz_path = self.storage_path / f"{quiz.id}.json"
            quiz_data = quiz.model_dump()

            with open(quiz_path, 'w', encoding='utf-8') as f:
                json.dump(quiz_data, f, indent=2, default=str)

            # Update index
            index = self._load_index()
            index[quiz.id] = {
                "title": quiz.title,
                "topic": quiz.topic,
                "question_count": len(quiz.questions),
                "created_at": str(quiz.created_at),
                "attempts": len(quiz.attempts)
            }
            self._save_index(index)

            logger.info(f"Saved quiz: {quiz.title} ({quiz.id})")
            return True
        except Exception as e:
            logger.error(f"Error saving quiz: {e}")
            return False

    def load_quiz(self, quiz_id: str) -> Optional[Quiz]:
        """Load a quiz from storage"""
        try:
            quiz_path = self.storage_path / f"{quiz_id}.json"
            if not quiz_path.exists():
                return None

            with open(quiz_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Parse dates
            data['created_at'] = datetime.fromisoformat(data['created_at']) if isinstance(data['created_at'], str) else data['created_at']

            # Parse questions
            questions = []
            for q_data in data.get('questions', []):
                if isinstance(q_data.get('question_type'), str):
                    q_data['question_type'] = QuestionType(q_data['question_type'])
                questions.append(QuizQuestion(**q_data))
            data['questions'] = questions

            # Parse attempts
            attempts = []
            for a_data in data.get('attempts', []):
                if isinstance(a_data.get('started_at'), str):
                    a_data['started_at'] = datetime.fromisoformat(a_data['started_at'])
                if a_data.get('completed_at') and isinstance(a_data['completed_at'], str):
                    a_data['completed_at'] = datetime.fromisoformat(a_data['completed_at'])
                attempts.append(QuizAttempt(**a_data))
            data['attempts'] = attempts

            return Quiz(**data)
        except Exception as e:
            logger.error(f"Error loading quiz {quiz_id}: {e}")
            return None

    def delete_quiz(self, quiz_id: str) -> bool:
        """Delete a quiz"""
        try:
            quiz_path = self.storage_path / f"{quiz_id}.json"
            if quiz_path.exists():
                quiz_path.unlink()

            index = self._load_index()
            if quiz_id in index:
                del index[quiz_id]
                self._save_index(index)

            return True
        except Exception as e:
            logger.error(f"Error deleting quiz: {e}")
            return False

    def list_quizzes(self) -> List[Dict[str, Any]]:
        """List all quizzes"""
        index = self._load_index()
        quizzes = [{"id": qid, **info} for qid, info in index.items()]
        quizzes.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        return quizzes

    def get_quizzes_by_topic(self, topic: str) -> List[Dict[str, Any]]:
        """Get quizzes for a specific topic"""
        all_quizzes = self.list_quizzes()
        return [q for q in all_quizzes if q.get('topic', '').lower() == topic.lower()]


class QuizGenerator:
    """
    Generates quizzes from educational content using LLM.
    """

    def __init__(self):
        """Initialize the quiz generator"""
        self.llm_client = get_llm_client()
        self.store = QuizStore()

    def generate_quiz(
        self,
        content: str,
        topic: str,
        num_questions: int = 5,
        title: Optional[str] = None
    ) -> Optional[Quiz]:
        """
        Generate a quiz from content.

        Args:
            content: The source text to generate quiz from
            topic: The topic for the quiz
            num_questions: Number of questions to generate
            title: Optional custom title

        Returns:
            Quiz if successful, None otherwise
        """
        try:
            logger.info(f"Generating quiz with {num_questions} questions for topic: {topic}")

            # Generate quiz using LLM
            raw_output = self._generate_raw_quiz(content, num_questions)

            if not raw_output:
                logger.error("Failed to generate quiz from LLM")
                return None

            # Parse the raw output into Question objects
            questions = self._parse_questions(raw_output, topic)

            if not questions:
                logger.error("Failed to parse questions from LLM output")
                return None

            # Create the quiz
            quiz = Quiz(
                title=title or f"{topic} Quiz",
                description=f"Quiz generated from {topic} content",
                topic=topic,
                questions=questions,
                source_content=content[:500]  # Store first 500 chars as reference
            )

            # Save to store
            if self.store.save_quiz(quiz):
                logger.info(f"Created quiz with {len(questions)} questions")
                return quiz
            else:
                logger.error("Failed to save quiz")
                return None

        except Exception as e:
            logger.error(f"Error generating quiz: {e}")
            return None

    def _generate_raw_quiz(self, content: str, num_questions: int) -> Optional[str]:
        """Generate raw quiz text using the LLM"""
        try:
            if self.llm_client.is_local_mode():
                return self._generate_local_quiz(content, num_questions)

            result = self.llm_client.generate_quiz(content, num_questions)

            if result:
                return result

            logger.warning("LLM quiz generation failed, using local fallback")
            return self._generate_local_quiz(content, num_questions)

        except Exception as e:
            logger.error(f"Error in raw quiz generation: {e}")
            return self._generate_local_quiz(content, num_questions)

    def _generate_local_quiz(self, content: str, num_questions: int) -> str:
        """
        Generate quiz using rule-based extraction (fallback for local mode).
        """
        logger.info("Using local rule-based quiz generation")
        questions = []

        # Split content into sentences
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 30]

        # Look for factual statements that can become questions
        for i, sentence in enumerate(sentences[:num_questions * 2]):
            if len(questions) >= num_questions:
                break

            # Try to create a question from the sentence
            question = self._sentence_to_question(sentence, i)
            if question:
                questions.append(question)

        return "\n\n".join(questions)

    def _sentence_to_question(self, sentence: str, index: int) -> Optional[str]:
        """Convert a sentence into a quiz question format"""
        try:
            # Pattern: "X is Y" -> "What is X?"
            is_match = re.search(r'([A-Z][^.]+?)\s+(?:is|are)\s+([^.]+)', sentence)
            if is_match:
                subject = is_match.group(1).strip()
                predicate = is_match.group(2).strip()

                # Create wrong answers by modifying the correct one
                wrong1 = f"Not related to {subject}"
                wrong2 = f"The opposite of {predicate}"
                wrong3 = "None of the above"

                return f"""Question: What is {subject}?
A) {predicate}
B) {wrong1}
C) {wrong2}
D) {wrong3}
Correct: A
Explanation: {sentence}"""

            # Pattern: Key terms - create True/False
            if any(kw in sentence.lower() for kw in ['important', 'key', 'main', 'essential', 'primary']):
                return f"""Question: True or False: {sentence}
A) True
B) False
C) Partially true
D) Cannot be determined
Correct: A
Explanation: This statement is from the source material."""

        except Exception as e:
            logger.debug(f"Error converting sentence to question: {e}")

        return None

    def _parse_questions(self, raw_output: str, topic: str) -> List[QuizQuestion]:
        """Parse raw LLM output into QuizQuestion objects"""
        questions = []

        try:
            # Split by question patterns
            question_blocks = re.split(r'(?=Question:|(?=\d+\.\s*Question)|(?=Q\d+:))', raw_output)

            for block in question_blocks:
                block = block.strip()
                if not block or len(block) < 20:
                    continue

                question = self._parse_single_question(block, topic)
                if question:
                    questions.append(question)

        except Exception as e:
            logger.error(f"Error parsing questions: {e}")

        return questions

    def _parse_single_question(self, block: str, topic: str) -> Optional[QuizQuestion]:
        """Parse a single question block"""
        try:
            question_text = None
            options = []
            correct_answer = None
            correct_index = None
            explanation = None

            # Extract question text
            q_match = re.search(r'Question:\s*(.+?)(?=\n[A-D]\)|$)', block, re.DOTALL | re.IGNORECASE)
            if q_match:
                question_text = q_match.group(1).strip()

            # Extract options
            option_pattern = r'([A-D])\)\s*(.+?)(?=\n[A-D]\)|(?=\nCorrect)|(?=\nExplanation)|$)'
            option_matches = re.findall(option_pattern, block, re.DOTALL | re.IGNORECASE)

            for letter, text in option_matches:
                options.append(text.strip())

            # Extract correct answer
            correct_match = re.search(r'Correct:\s*([A-D])', block, re.IGNORECASE)
            if correct_match:
                correct_letter = correct_match.group(1).upper()
                correct_index = ord(correct_letter) - ord('A')
                if 0 <= correct_index < len(options):
                    correct_answer = options[correct_index]

            # Extract explanation
            exp_match = re.search(r'Explanation:\s*(.+?)(?=\n\n|$)', block, re.DOTALL | re.IGNORECASE)
            if exp_match:
                explanation = exp_match.group(1).strip()

            # Validate we have required fields
            if question_text and options and correct_answer:
                # Determine question type
                if len(options) == 2 and any('true' in o.lower() or 'false' in o.lower() for o in options):
                    question_type = QuestionType.TRUE_FALSE
                else:
                    question_type = QuestionType.MULTIPLE_CHOICE

                return QuizQuestion(
                    question=question_text,
                    question_type=question_type,
                    options=options,
                    correct_answer=correct_answer,
                    correct_index=correct_index,
                    explanation=explanation,
                    topic=topic
                )

        except Exception as e:
            logger.debug(f"Error parsing question block: {e}")

        return None

    def submit_answer(
        self,
        quiz_id: str,
        attempt_id: str,
        question_id: str,
        answer: str
    ) -> Dict[str, Any]:
        """
        Submit an answer for a quiz question.

        Returns dict with correct status and explanation.
        """
        try:
            quiz = self.store.load_quiz(quiz_id)
            if not quiz:
                return {"success": False, "error": "Quiz not found"}

            # Find the attempt
            attempt = None
            for a in quiz.attempts:
                if a.id == attempt_id:
                    attempt = a
                    break

            if not attempt:
                return {"success": False, "error": "Attempt not found"}

            # Find the question
            question = quiz.get_question(question_id)
            if not question:
                return {"success": False, "error": "Question not found"}

            # Check the answer
            correct = question.check_answer(answer)
            attempt.submit_answer(question_id, answer, correct)

            # Save updated quiz
            self.store.save_quiz(quiz)

            return {
                "success": True,
                "correct": correct,
                "correct_answer": question.correct_answer,
                "explanation": question.explanation
            }

        except Exception as e:
            logger.error(f"Error submitting answer: {e}")
            return {"success": False, "error": str(e)}

    def complete_attempt(self, quiz_id: str, attempt_id: str) -> Dict[str, Any]:
        """Complete a quiz attempt and calculate final score with detailed results"""
        try:
            quiz = self.store.load_quiz(quiz_id)
            if not quiz:
                return {"success": False, "error": "Quiz not found"}

            attempt = None
            for a in quiz.attempts:
                if a.id == attempt_id:
                    attempt = a
                    break

            if not attempt:
                return {"success": False, "error": "Attempt not found"}

            attempt.complete()
            self.store.save_quiz(quiz)

            # Build detailed results for each question
            detailed_results = []
            for question in quiz.questions:
                user_answer = attempt.answers.get(question.id, "")
                is_correct = attempt.results.get(question.id, False)

                detailed_results.append({
                    "question_id": question.id,
                    "question_text": question.question,
                    "user_answer": user_answer,
                    "correct_answer": question.correct_answer,
                    "is_correct": is_correct,
                    "options": question.options,
                    "explanation": question.explanation or "No explanation available."
                })

            return {
                "success": True,
                "score": attempt.score,
                "correct_count": attempt.correct_count,
                "total_questions": attempt.total_questions,
                "results": attempt.results,
                "detailed_results": detailed_results
            }

        except Exception as e:
            logger.error(f"Error completing attempt: {e}")
            return {"success": False, "error": str(e)}


# Singleton instances
_quiz_store = None
_quiz_generator = None


def get_quiz_store() -> QuizStore:
    """Get or create quiz store singleton"""
    global _quiz_store
    if _quiz_store is None:
        _quiz_store = QuizStore()
    return _quiz_store


def get_quiz_generator() -> QuizGenerator:
    """Get or create quiz generator singleton"""
    global _quiz_generator
    if _quiz_generator is None:
        _quiz_generator = QuizGenerator()
    return _quiz_generator
