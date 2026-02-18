"""
Text processing utilities
"""
import re
from typing import List, Dict, Any

def clean_text(text: str) -> str:
    """Clean and normalize text"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\-\:\;]', '', text)
    return text.strip()

def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """Extract keywords from text"""
    # Simple keyword extraction (can be enhanced with NLP)
    words = text.lower().split()
    # Remove common stop words
    stop_words = {'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'is', 'are', 'was', 'were'}
    keywords = [w for w in words if w not in stop_words and len(w) > 3]
    # Return unique keywords
    seen = set()
    unique_keywords = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            unique_keywords.append(kw)
    return unique_keywords[:max_keywords]

def format_as_markdown(title: str, content: Dict[str, Any]) -> str:
    """Format content as markdown based on summarization mode"""
    md = f"### {title}\n\n"

    if "summary" in content:
        # Determine section header and formatting based on summarization mode
        summarization_mode = content.get('metadata', {}).get('summarization_mode', 'paragraph_summary')

        # Define section headers for each mode
        # Note: important_points and key_highlights don't need headers since title already indicates the type
        section_headers = {
            'paragraph_summary': '#### Overview\n\n',
            'important_points': '',
            'key_highlights': ''
        }
        md += section_headers.get(summarization_mode, '#### Overview\n\n')

        summary_text = content['summary']

        if isinstance(summary_text, list):
            # Format as bullet points if it's a list
            for point in summary_text:
                md += f"- {point}\n"
            md += "\n"
        elif summarization_mode == 'key_highlights':
            # Key highlights: Keep bullet format, minimal processing
            # The LLM output should already be in bullet format
            lines = summary_text.split('\n')
            for line in lines:
                line = line.strip()
                if line:
                    # Ensure bullet format
                    if not line.startswith(('•', '-', '*')):
                        md += f"• {line}\n"
                    else:
                        md += f"{line}\n"
            md += "\n"
        elif summarization_mode == 'important_points':
            # Important points: Keep ONLY numbered/bullet items
            # Filter out any preamble text the LLM might generate
            lines = summary_text.split('\n')

            # Pattern to match numbered points (1. 2. 3. etc.) or bullets
            numbered_pattern = re.compile(r'^\d+[\.\)]\s*')
            bullet_pattern = re.compile(r'^[\•\-\*]\s*')

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Only include lines that are actual numbered points or bullets
                if numbered_pattern.match(line) or bullet_pattern.match(line):
                    md += f"{line}\n\n"
        else:
            # Paragraph summary: Format as flowing paragraphs
            # Check if content already has paragraph breaks
            if '\n\n' in summary_text:
                # Already has paragraph structure
                paragraphs = summary_text.split('\n\n')
                for paragraph in paragraphs:
                    paragraph = paragraph.strip()
                    if paragraph:
                        md += f"{paragraph}\n\n"
            else:
                # Try to create paragraphs from continuous text
                # Split by double newline first, then by period patterns
                summary_text = summary_text.replace('\n', ' ').strip()
                sentences = summary_text.split('. ')

                # Group sentences into paragraphs (every 3-4 sentences)
                paragraphs = []
                current_paragraph = []

                for i, sentence in enumerate(sentences):
                    sentence = sentence.strip()
                    if sentence:
                        current_paragraph.append(sentence)
                        # Create new paragraph every 3-4 sentences
                        if (i + 1) % 4 == 0:
                            if current_paragraph:
                                paragraph_text = '. '.join(current_paragraph)
                                if not paragraph_text.endswith('.'):
                                    paragraph_text += '.'
                                paragraphs.append(paragraph_text)
                                current_paragraph = []

                # Add remaining sentences as final paragraph
                if current_paragraph:
                    paragraph_text = '. '.join(current_paragraph)
                    if not paragraph_text.endswith('.'):
                        paragraph_text += '.'
                    paragraphs.append(paragraph_text)

                # Add paragraphs with proper spacing
                for paragraph in paragraphs:
                    if paragraph.strip():
                        md += f"{paragraph}\n\n"

    if "key_points" in content and content["key_points"]:
        md += "#### Additional Points\n\n"
        for point in content["key_points"]:
            md += f"- {point}\n"
        md += "\n"

    if "sources" in content and content["sources"]:
        # Filter to only include sources with valid URLs (must start with http)
        valid_sources = [
            s for s in content["sources"]
            if s.get('url') and s['url'].startswith(('http://', 'https://'))
        ]
        if valid_sources:
            md += "#### Sources\n\n"
            for source in valid_sources:
                md += f"- [{source['title']}]({source['url']})\n"
            md += "\n"

    if "metadata" in content:
        md += "---\n"
        md += f"*Generated on: {content['metadata'].get('date', 'N/A')}*\n"
        if 'topic' in content['metadata']:
            md += f"*Topic: {content['metadata']['topic']}*\n"

    return md

def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """Split text into chunks with overlap"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    
    return chunks