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
    """Format content as markdown with paragraph-style formatting"""
    md = f"# {title}\n\n"

    if "summary" in content:
        md += "## Overview\n\n"
        if isinstance(content["summary"], list):
            # Format as bullet points if it's a list
            for point in content["summary"]:
                md += f"- {point}\n"
        else:
            # Format as paragraphs for detailed content
            summary_text = content['summary']
            # Split long text into paragraphs for better readability
            sentences = summary_text.split('. ')

            # Group sentences into paragraphs (roughly every 3-4 sentences)
            paragraphs = []
            current_paragraph = []

            for i, sentence in enumerate(sentences):
                current_paragraph.append(sentence.strip())
                # Create new paragraph every 3-4 sentences or if sentence is very long
                if (i + 1) % 3 == 0 or len(sentence) > 200:
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

        md += "\n"

    if "key_points" in content and content["key_points"]:
        md += "## Key Points\n\n"
        for point in content["key_points"]:
            md += f"- {point}\n"
        md += "\n"

    if "sources" in content and content["sources"]:
        md += "## Sources\n\n"
        for source in content["sources"]:
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