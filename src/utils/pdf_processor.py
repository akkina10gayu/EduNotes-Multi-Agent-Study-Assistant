"""
PDF Processing Utility for EduNotes
Extracts text content from PDF files for note generation.

Supports two extraction backends:
- pymupdf4llm (preferred): Markdown output with tables, equations, structure preserved
- PyPDF2 (fallback): Basic text extraction
"""
import base64
import hashlib
from typing import Optional, Dict, Any, List
from src.utils.logger import get_logger
from src.utils.cache_utils import cache, get_cache_key

logger = get_logger(__name__)

# Try to import pymupdf4llm (preferred), fall back to PyPDF2
try:
    import pymupdf4llm
    import pymupdf
    HAS_PYMUPDF4LLM = True
    logger.info("Using pymupdf4llm for PDF extraction (tables, equations, structure preserved)")
except ImportError:
    HAS_PYMUPDF4LLM = False
    import PyPDF2
    logger.info("pymupdf4llm not available, using PyPDF2 (basic text extraction)")


class PDFProcessor:
    """Handles PDF file processing and text extraction"""

    def __init__(self):
        self.logger = logger

    def extract_text_from_pdf(self, pdf_path: str) -> Optional[str]:
        """
        Extract text from a PDF file.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Extracted text as string, or None if extraction fails
        """
        try:
            if HAS_PYMUPDF4LLM:
                doc = pymupdf.open(pdf_path)
                md_text = pymupdf4llm.to_markdown(doc)
                doc.close()

                if not md_text or not md_text.strip():
                    self.logger.warning("No text extracted from PDF")
                    return None

                self.logger.info(f"Successfully extracted {len(md_text)} characters from PDF (pymupdf4llm)")
                return md_text
            else:
                return self._extract_with_pypdf2_file(pdf_path)

        except Exception as e:
            self.logger.error(f"Error extracting text from PDF: {e}")
            if HAS_PYMUPDF4LLM:
                self.logger.info("Falling back to PyPDF2")
                try:
                    return self._extract_with_pypdf2_file(pdf_path)
                except Exception as e2:
                    self.logger.error(f"PyPDF2 fallback also failed: {e2}")
            return None

    def extract_text_from_bytes(self, pdf_bytes: bytes) -> Optional[str]:
        """
        Extract text from PDF bytes (for uploaded files).

        Args:
            pdf_bytes: PDF file content as bytes

        Returns:
            Extracted text as string, or None if extraction fails
        """
        # Check cache by content hash (first 10KB is fast and unique enough)
        content_hash = hashlib.md5(pdf_bytes[:10000]).hexdigest()
        cache_key = get_cache_key("pdf_text", content_hash)
        cached_text = cache.get(cache_key)
        if cached_text is not None:
            self.logger.info(f"PDF text extraction cache HIT ({len(cached_text)} chars)")
            return cached_text

        try:
            if HAS_PYMUPDF4LLM:
                doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
                md_text = pymupdf4llm.to_markdown(doc)
                doc.close()

                if not md_text or not md_text.strip():
                    self.logger.warning("No text extracted from uploaded PDF")
                    return None

                self.logger.info(f"Successfully extracted {len(md_text)} characters from uploaded PDF (pymupdf4llm)")
                cache.set(cache_key, md_text, expire=86400)
                return md_text
            else:
                result = self._extract_with_pypdf2_bytes(pdf_bytes)
                if result:
                    cache.set(cache_key, result, expire=86400)
                return result

        except Exception as e:
            self.logger.error(f"Error extracting text from uploaded PDF: {e}")
            if HAS_PYMUPDF4LLM:
                self.logger.info("Falling back to PyPDF2 for bytes extraction")
                try:
                    result = self._extract_with_pypdf2_bytes(pdf_bytes)
                    if result:
                        cache.set(cache_key, result, expire=86400)
                    return result
                except Exception as e2:
                    self.logger.error(f"PyPDF2 fallback also failed: {e2}")
            return None

    def extract_for_research(self, pdf_bytes: bytes) -> Dict[str, Any]:
        """
        Enhanced extraction for Research Mode.

        Returns markdown text (with tables/equations preserved) plus
        base64-encoded images of pages that contain figures.

        Args:
            pdf_bytes: PDF file content as bytes

        Returns:
            {
                'text': str,           # Markdown-formatted text
                'figure_pages': [      # Pages with embedded images
                    {'page_num': 1, 'image_base64': '...'},
                    ...
                ],
                'total_pages': int,
                'pages_with_figures': int
            }
        """
        # Check cache by content hash
        content_hash = hashlib.md5(pdf_bytes[:10000]).hexdigest()
        cache_key = get_cache_key("pdf_research", content_hash)
        cached_result = cache.get(cache_key)
        if cached_result is not None:
            self.logger.info(f"PDF research extraction cache HIT ({len(cached_result.get('text', ''))} chars)")
            return cached_result

        result = {
            'text': '',
            'figure_pages': [],
            'total_pages': 0,
            'pages_with_figures': 0
        }

        if not HAS_PYMUPDF4LLM:
            # Without pymupdf4llm, return basic extraction with no figure detection
            text = self._extract_with_pypdf2_bytes(pdf_bytes)
            result['text'] = text or ''
            self.logger.warning("Research mode without pymupdf4llm: no figure detection available")
            return result

        try:
            doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
            result['total_pages'] = len(doc)

            # Step 1: Extract markdown text (tables, equations, structure preserved)
            md_text = pymupdf4llm.to_markdown(doc)
            result['text'] = md_text or ''
            self.logger.info(f"Research extraction: {len(result['text'])} chars from {len(doc)} pages")

            # Step 2: Detect pages with embedded images
            figure_page_nums = []
            for page_num in range(len(doc)):
                page = doc[page_num]
                images = page.get_images()
                if images:
                    figure_page_nums.append(page_num)

            result['pages_with_figures'] = len(figure_page_nums)
            self.logger.info(f"Research extraction: {len(figure_page_nums)} pages contain figures")

            # Step 3: Convert figure pages to images (max 3 to limit API calls)
            for page_num in figure_page_nums[:3]:
                try:
                    page = doc[page_num]
                    pix = page.get_pixmap(dpi=150)
                    img_bytes = pix.tobytes("png")
                    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

                    result['figure_pages'].append({
                        'page_num': page_num + 1,  # 1-indexed for display
                        'image_base64': img_b64
                    })
                except Exception as e:
                    self.logger.warning(f"Failed to convert page {page_num + 1} to image: {e}")

            doc.close()

            self.logger.info(
                f"Research extraction complete: {result['total_pages']} pages, "
                f"{result['pages_with_figures']} with figures, "
                f"{len(result['figure_pages'])} converted to images"
            )
            cache.set(cache_key, result, expire=86400)
            return result

        except Exception as e:
            self.logger.error(f"Error in research extraction: {e}")
            # Fallback to basic extraction
            text = self._extract_with_pypdf2_bytes(pdf_bytes)
            result['text'] = text or ''
            return result

    # =========================================================================
    # PyPDF2 fallback methods (original implementation)
    # =========================================================================

    def _extract_with_pypdf2_file(self, pdf_path: str) -> Optional[str]:
        """Extract text from PDF file using PyPDF2."""
        try:
            import PyPDF2
        except ImportError:
            self.logger.error("PyPDF2 not available for fallback")
            return None

        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            num_pages = len(pdf_reader.pages)
            self.logger.info(f"Extracting text from PDF with {num_pages} pages (PyPDF2)")

            text_content = []
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                if text.strip():
                    text_content.append(text)

            full_text = "\n\n".join(text_content)
            if not full_text.strip():
                return None

            self.logger.info(f"Successfully extracted {len(full_text)} characters (PyPDF2)")
            return full_text

    def _extract_with_pypdf2_bytes(self, pdf_bytes: bytes) -> Optional[str]:
        """Extract text from PDF bytes using PyPDF2."""
        try:
            import PyPDF2
        except ImportError:
            self.logger.error("PyPDF2 not available for fallback")
            return None

        from io import BytesIO

        pdf_file = BytesIO(pdf_bytes)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        num_pages = len(pdf_reader.pages)
        self.logger.info(f"Extracting text from uploaded PDF with {num_pages} pages (PyPDF2)")

        text_content = []
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            if text.strip():
                text_content.append(text)

        full_text = "\n\n".join(text_content)
        if not full_text.strip():
            return None

        self.logger.info(f"Successfully extracted {len(full_text)} characters (PyPDF2)")
        return full_text



# Singleton instance
_pdf_processor = None


def get_pdf_processor() -> PDFProcessor:
    """Get or create PDF processor instance"""
    global _pdf_processor
    if _pdf_processor is None:
        _pdf_processor = PDFProcessor()
    return _pdf_processor
