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
    """Format content as markdown"""
    md = f"# {title}\n\n"
    
    if "summary" in content:
        md += "## Summary\n\n"
        if isinstance(content["summary"], list):
            for point in content["summary"]:
                md += f"- {point}\n"
        else:
            md += f"{content['summary']}\n"
        md += "\n"
    
    if "key_points" in content:
        md += "## Key Points\n\n"
        for point in content["key_points"]:
            md += f"- {point}\n"
        md += "\n"
    
    if "sources" in content:
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