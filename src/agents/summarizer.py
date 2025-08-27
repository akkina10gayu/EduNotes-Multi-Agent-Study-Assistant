"""
Summarization agent using Flan-T5
"""
from typing import Dict, Any, List
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document

from src.agents.base import BaseAgent
from src.utils.cache_utils import cached
from src.utils.text_utils import chunk_text
from config import settings

class SummarizerAgent(BaseAgent):
    """Agent for text summarization"""
    
    def __init__(self):
        super().__init__("SummarizerAgent")
        self.model_name = settings.SUMMARIZATION_MODEL
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the summarization model"""
        try:
            self.logger.info(f"Loading model: {self.model_name}")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=str(settings.MODEL_CACHE_DIR)
            )
            
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                cache_dir=str(settings.MODEL_CACHE_DIR),
                torch_dtype=torch.float32
            )
            
            # Create pipeline
            self.pipe = pipeline(
                "summarization",
                model=self.model,
                tokenizer=self.tokenizer,
                max_length=150,
                min_length=30,
                do_sample=False,
                device=-1  # CPU
            )
            
            # Create LangChain LLM
            self.llm = HuggingFacePipeline(pipeline=self.pipe)
            
            self.logger.info("Model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
    
    @cached("summarizer", ttl=settings.CACHE_SUMMARY_TTL)
    def summarize_text(self, text: str, max_length: int = None) -> str:
        """Summarize a single text"""
        try:
            # For Flan-T5, we need to add a prompt
            prompt = f"summarize: {text}"
            
            # Truncate if too long
            max_input_length = 512
            input_tokens = self.tokenizer.encode(prompt)
            if len(input_tokens) > max_input_length:
                input_tokens = input_tokens[:max_input_length]
                prompt = self.tokenizer.decode(input_tokens, skip_special_tokens=True)
            
            # Set dynamic max_length based on input length
            if max_length is None:
                input_length = len(input_tokens)
                # Set max_length to 80% of input length, minimum 30, maximum 150
                max_length = max(30, min(150, int(input_length * 0.8)))
            
            # Generate summary
            result = self.pipe(prompt, max_length=max_length, min_length=min(30, max_length-10))
            summary = result[0]['summary_text'] if result else ""
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error summarizing text: {e}")
            return ""
    
    def summarize_documents(self, documents: List[str]) -> str:
        """Summarize multiple documents using LangChain"""
        try:
            # Convert to LangChain documents
            docs = [Document(page_content=doc) for doc in documents]
            
            # Use map-reduce chain for multiple documents
            chain = load_summarize_chain(
                self.llm,
                chain_type="map_reduce",
                verbose=False
            )
            
            summary = chain.run(docs)
            return summary
            
        except Exception as e:
            self.logger.error(f"Error summarizing documents: {e}")
            # Fallback to simple concatenation and summarization
            combined = " ".join(documents[:3])  # Take first 3 documents
            return self.summarize_text(combined)
    
    def extract_bullet_points(self, text: str, num_points: int = 5) -> List[str]:
        """Extract key points as bullet points"""
        try:
            prompt = f"Extract {num_points} key points from this text as a bullet list: {text}"
            
            # Truncate if needed
            input_tokens = self.tokenizer.encode(prompt)
            if len(input_tokens) > 512:
                input_tokens = input_tokens[:512]
                prompt = self.tokenizer.decode(input_tokens, skip_special_tokens=True)
            
            # Set dynamic max_length for bullet points (should be longer to accommodate multiple points)
            input_length = len(input_tokens)
            max_length = max(50, min(200, int(input_length * 1.2)))  # Allow expansion for bullet points
            
            result = self.pipe(prompt, max_length=max_length, min_length=min(40, max_length-10))
            summary = result[0]['summary_text'] if result else ""
            
            # Parse bullet points
            lines = summary.split('\n')
            bullet_points = []
            for line in lines:
                line = line.strip()
                if line and (line.startswith('-') or line.startswith('•') or line.startswith('*')):
                    bullet_points.append(line.lstrip('-•* '))
                elif line:
                    bullet_points.append(line)
            
            return bullet_points[:num_points]
            
        except Exception as e:
            self.logger.error(f"Error extracting bullet points: {e}")
            # Fallback to simple summary
            summary = self.summarize_text(text)
            return [summary] if summary else []
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process summarization request"""
        try:
            if not self.validate_input(input_data):
                return self.handle_error(ValueError("Invalid input"))
            
            content = input_data.get('content', '')
            mode = input_data.get('mode', 'summary')  # summary or bullet_points
            
            if not content:
                return self.handle_error(ValueError("No content provided"))
            
            result = {
                'success': True,
                'agent': self.name
            }
            
            if mode == 'bullet_points':
                num_points = input_data.get('num_points', 5)
                bullet_points = self.extract_bullet_points(content, num_points)
                result['bullet_points'] = bullet_points
                result['summary'] = '\n'.join(f"• {point}" for point in bullet_points)
            else:
                # Check if content is a list (multiple documents)
                if isinstance(content, list):
                    summary = self.summarize_documents(content)
                else:
                    summary = self.summarize_text(content)
                result['summary'] = summary
            
            self.logger.info(f"Successfully summarized content in {mode} mode")
            return result
            
        except Exception as e:
            return self.handle_error(e)