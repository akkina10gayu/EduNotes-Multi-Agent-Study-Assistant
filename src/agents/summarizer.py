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
            
            # Create pipeline with dramatically increased context window
            self.pipe = pipeline(
                "summarization",
                model=self.model,
                tokenizer=self.tokenizer,
                max_length=1024,  # Increased to 1024 tokens for much longer output
                min_length=200,   # Increased minimum for substantial content
                do_sample=True,   # Enable sampling
                early_stopping=True,  # Better for BART
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
            self.logger.info(f"Summarizing text of length: {len(text)}")

            # Increase input text limit dramatically
            if len(text) > 8000:  # Much higher limit for comprehensive content
                text = text[:8000]
                self.logger.warning(f"Truncated input text to 8000 characters")

            # Use much more input text for comprehensive summaries
            if "machine learning" in text.lower() or "data science" in text.lower() or "algorithm" in text.lower():
                prompt = f"Technical Summary: {text[:3000]}"  # Tripled input size
            else:
                prompt = text[:3000]  # Much larger input for better summaries
            
            # BART can handle up to 1024 input tokens, but we'll use full capacity
            max_input_length = 1024
            input_tokens = self.tokenizer.encode(prompt)
            if len(input_tokens) > max_input_length:
                input_tokens = input_tokens[:max_input_length]
                prompt = self.tokenizer.decode(input_tokens, skip_special_tokens=True)
            
            # Set much higher dynamic max_length for comprehensive summaries
            if max_length is None:
                input_length = len(input_tokens)
                # Dramatically increased range: minimum 200, maximum 1024
                max_length = max(200, min(1024, int(input_length * 1.0)))
            
            # Generate summary with much higher min_length for comprehensive content
            min_len = max(150, min(max_length//2, 400))  # Much higher minimum for detailed summaries
            result = self.pipe(prompt, max_length=max_length, min_length=min_len)
            summary = result[0]['summary_text'] if result else ""
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error summarizing text: {e}")
            # Return a fallback summary instead of empty string
            return f"Summary of content about {text[:100]}... [Error in detailed summarization: {str(e)}]"
    
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
            # For BART, use direct text for better bullet point extraction
            if "machine learning" in text.lower() or "data science" in text.lower():
                prompt = f"Key technical concepts: {text[:800]}"
            else:
                prompt = text[:800]

            # BART token limits
            input_tokens = self.tokenizer.encode(prompt)
            if len(input_tokens) > 1024:
                input_tokens = input_tokens[:1024]
                prompt = self.tokenizer.decode(input_tokens, skip_special_tokens=True)

            # BART-optimized length for bullet points
            input_length = len(input_tokens)
            max_length = max(150, min(400, int(input_length * 0.7)))
            
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
            self.logger.error(f"Error in summarizer process: {e}")
            return {
                'success': False,
                'agent': self.name,
                'error': str(e),
                'summary': f"Error processing summarization request: {str(e)}"
            }