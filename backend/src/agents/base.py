"""
Base agent class for all agents
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from src.utils.logger import get_logger

class BaseAgent(ABC):
    """Abstract base class for all agents"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = get_logger(name)
    
    @abstractmethod
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input and return result"""
        pass
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data"""
        if not input_data:
            self.logger.error("Empty input data")
            return False
        return True
    
    def handle_error(self, error: Exception) -> Dict[str, Any]:
        """Handle errors gracefully"""
        self.logger.error(f"Error in {self.name}: {str(error)}")
        return {
            'success': False,
            'error': str(error),
            'agent': self.name
        }