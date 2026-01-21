"""
Note-making agent for creating structured study notes
"""
from typing import Dict, Any, List, Optional
from datetime import datetime
from src.agents.base import BaseAgent
from src.utils.text_utils import format_as_markdown, extract_keywords

class NoteMakerAgent(BaseAgent):
    """Agent for creating structured study notes"""
    
    def __init__(self):
        super().__init__("NoteMakerAgent")
    
    def create_notes(self,
                    title: str,
                    summary: str,
                    key_points: List[str] = None,
                    sources: List[Dict[str, str]] = None,
                    topic: str = None,
                    summarization_mode: str = "detailed") -> str:
        """Create structured notes in markdown format"""
        try:
            content = {}

            # Add summary with mode information
            if summary:
                # If summary is already in bullet points, parse it
                if '\n•' in summary or '\n-' in summary:
                    points = []
                    for line in summary.split('\n'):
                        line = line.strip()
                        if line and (line.startswith('•') or line.startswith('-')):
                            points.append(line.lstrip('•- '))
                    content['summary'] = points if points else summary
                else:
                    content['summary'] = summary
            
            # Add key points
            if key_points:
                content['key_points'] = key_points
            
            # Add sources
            if sources:
                content['sources'] = sources
            
            # Add metadata
            content['metadata'] = {
                'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'topic': topic or 'General',
                'summarization_mode': summarization_mode  # Pass mode to formatter
            }

            # Format as markdown
            notes = format_as_markdown(title, content)
            
            self.logger.info(f"Created notes for: {title}")
            return notes
            
        except Exception as e:
            self.logger.error(f"Error creating notes: {e}")
            return ""
    
    def merge_notes(self, notes_list: List[str]) -> str:
        """Merge multiple notes into a single document"""
        try:
            merged = "# Merged Study Notes\n\n"
            merged += f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
            merged += "---\n\n"
            
            for idx, notes in enumerate(notes_list, 1):
                # Add section separator
                if idx > 1:
                    merged += "\n---\n\n"
                
                # Add the notes (remove the title if it exists)
                lines = notes.split('\n')
                if lines and lines[0].startswith('#'):
                    # Convert main title to section title
                    lines[0] = f"## Section {idx}: {lines[0].lstrip('#').strip()}"
                
                merged += '\n'.join(lines)
            
            return merged
            
        except Exception as e:
            self.logger.error(f"Error merging notes: {e}")
            return ""
    
    def organize_by_topic(self, notes_dict: Dict[str, List[str]]) -> str:
        """Organize notes by topic"""
        try:
            organized = "# Study Notes by Topic\n\n"
            organized += f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
            organized += "## Table of Contents\n\n"
            
            # Add TOC
            for topic in notes_dict.keys():
                organized += f"- [{topic}](#{topic.lower().replace(' ', '-')})\n"
            
            organized += "\n---\n\n"
            
            # Add content for each topic
            for topic, notes_list in notes_dict.items():
                organized += f"## {topic}\n\n"
                
                for notes in notes_list:
                    # Remove title from individual notes
                    lines = notes.split('\n')
                    if lines and lines[0].startswith('#'):
                        lines = lines[1:]  # Skip the title
                    
                    organized += '\n'.join(lines)
                    organized += "\n\n"
                
                organized += "---\n\n"
            
            return organized
            
        except Exception as e:
            self.logger.error(f"Error organizing notes: {e}")
            return ""
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process note-making request"""
        try:
            if not self.validate_input(input_data):
                return self.handle_error(ValueError("Invalid input"))
            
            mode = input_data.get('mode', 'create')  # create, merge, or organize
            
            if mode == 'create':
                title = input_data.get('title', 'Study Notes')
                summary = input_data.get('summary', '')
                key_points = input_data.get('key_points', [])
                sources = input_data.get('sources', [])
                topic = input_data.get('topic', 'General')
                summarization_mode = input_data.get('summarization_mode', 'detailed')

                notes = self.create_notes(
                    title=title,
                    summary=summary,
                    key_points=key_points,
                    sources=sources,
                    topic=topic,
                    summarization_mode=summarization_mode
                )
                
                return {
                    'success': True,
                    'notes': notes,
                    'format': 'markdown',
                    'agent': self.name
                }
            
            elif mode == 'merge':
                notes_list = input_data.get('notes_list', [])
                merged = self.merge_notes(notes_list)
                
                return {
                    'success': True,
                    'notes': merged,
                    'format': 'markdown',
                    'agent': self.name
                }
            
            elif mode == 'organize':
                notes_dict = input_data.get('notes_dict', {})
                organized = self.organize_by_topic(notes_dict)
                
                return {
                    'success': True,
                    'notes': organized,
                    'format': 'markdown',
                    'agent': self.name
                }
            
            else:
                return self.handle_error(ValueError(f"Unknown mode: {mode}"))
                
        except Exception as e:
            return self.handle_error(e)