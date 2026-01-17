"""
FILE 1: DOCUMENT LOADING
========================

CONCEPT: How to extract text from PDFs and prepare it for RAG

WHAT THIS FILE DOES:
1. Loads PDF files and extracts text
2. Splits text into smaller chunks (for LLM context window)
3. Handles overlap between chunks (to preserve context)

WHY WE NEED THIS:
- PDFs store text in complex binary format (not plain text)
- LLMs have token limits (~8K for LLaMA-3.1)
- A 50-page PDF = 40K tokens → Must chunk it!

INTERVIEW Q&A:
--------------
Q: Why chunk documents instead of sending full text to LLM?
A: "LLMs have context window limits. LLaMA-3.1 has 8,192 tokens (~6,000 words).
   A typical medical PDF is 20-50 pages (~15K words). Chunking lets us send
   only the RELEVANT sections to the LLM, saving tokens and improving accuracy."

Q: Why use overlap between chunks?
A: "Imagine splitting: 'diabetes symptoms' | 'include increased thirst'.
   Without overlap, we lose context at boundaries. With 100-char overlap,
   both chunks contain complete sentences, preserving meaning."

Q: Why 1000 characters per chunk?
A: "It's a balance:
   - Too small (100 chars): Fragments sentences, loses context
   - Too large (5000 chars): Uses too many tokens, may include irrelevant info
   - 1000 chars ≈ 150-200 words ≈ 1-2 paragraphs (sweet spot)"
"""

import PyPDF2
from pathlib import Path
from typing import List, Dict


class DocumentLoader:
    """
    Loads and processes documents for RAG system.
    
    Design Decision: Use PyPDF2 instead of LangChain
    WHY: Want to understand text extraction from first principles.
         LangChain abstracts this away. For learning, better to see
         how PDFs actually work.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        """
        Initialize document loader.
        
        Args:
            chunk_size: Characters per chunk (default: 1000)
                       ~150-200 words, ~1-2 paragraphs
            chunk_overlap: Characters to overlap between chunks (default: 100)
                          ~15-20 words, preserves context at boundaries
        
        Interview Note:
        "I chose 1000/100 based on LLM token limits. LLaMA-3.1 has 8K tokens.
         With top-5 retrieval, that's 5×1000 chars = 5K chars ≈ 1250 tokens.
         Leaves 6750 tokens for response. Good balance!"
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        print(f"✅ DocumentLoader initialized")
        print(f"   Chunk size: {chunk_size} characters")
        print(f"   Chunk overlap: {chunk_overlap} characters")
    
    
    def load_pdf(self, file_path: str) -> str:
        """
        Extract text from a PDF file.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Extracted text as string
            
        How it Works:
        1. Open PDF in binary mode ('rb')
        2. PyPDF2.PdfReader parses PDF structure
        3. Iterate through each page
        4. Extract text objects from each page
        5. Concatenate all text
        
        Interview Note:
        "PDFs don't store text as plain strings. They use PostScript-like
         commands to position text objects. PyPDF2 parses these commands
         and reconstructs the text in reading order."
        """
        file_path = Path(file_path)
        
        # Validate file exists and is PDF
        if not file_path.exists():
            raise FileNotFoundError(f"PDF not found: {file_path}")
        
        if file_path.suffix.lower() != '.pdf':
            raise ValueError(f"Expected PDF file, got: {file_path.suffix}")
        
        print(f"\n📄 Loading PDF: {file_path.name}")
        
        # Extract text from PDF
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                
                print(f"   Pages: {num_pages}")
                
                # Extract text from each page
                text = ""
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    text += page_text
                    
                    # Progress indicator
                    if (page_num + 1) % 10 == 0:
                        print(f"   Processed {page_num + 1}/{num_pages} pages...")
                
                print(f"✅ Extracted {len(text)} characters")
                return text
                
        except Exception as e:
            print(f"❌ Error reading PDF: {str(e)}")
            raise
    
    
    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Full text to chunk
            metadata: Optional metadata to attach to each chunk (e.g., source file)
            
        Returns:
            List of chunk dictionaries with text and metadata
            
        Algorithm:
        1. Start at position 0
        2. Take chunk_size characters
        3. Move forward by (chunk_size - overlap)
        4. Repeat until end of text
        
        Example with chunk_size=10, overlap=3:
        Text: "Hello world from Python"
                 
        Chunk 1: "Hello worl" (pos 0-10)
        Chunk 2: "rld from P" (pos 7-17)  ← Overlaps "rld" with chunk 1
        Chunk 3: "m Python"   (pos 14-22) ← Overlaps "m P" with chunk 2
        
        Interview Note:
        "Overlapping chunks ensure we don't split important context.
         If 'diabetes symptoms include thirst' is split at 'include',
         one chunk loses 'diabetes symptoms' and the other loses 'thirst'.
         With overlap, both chunks have complete information."
        """
        if not text or len(text) == 0:
            return []
        
        print(f"\n✂️  Chunking text...")
        print(f"   Total length: {len(text)} characters")
        
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(text):
            # Calculate end position
            end = start + self.chunk_size
            
            # Don't go beyond text length
            if end > len(text):
                end = len(text)
            
            # Extract chunk
            chunk_text = text[start:end]
            
            # Create chunk dictionary
            chunk = {
                'chunk_id': chunk_id,
                'text': chunk_text,
                'start_pos': start,
                'end_pos': end,
                'char_count': len(chunk_text)
            }
            
            # Add metadata if provided
            if metadata:
                chunk['metadata'] = metadata
            
            chunks.append(chunk)
            
            # Move to next chunk (with overlap)
            start += self.chunk_size - self.chunk_overlap
            chunk_id += 1
        
        print(f"✅ Created {len(chunks)} chunks")
        print(f"   Average chunk size: {sum(c['char_count'] for c in chunks) // len(chunks)} chars")
        
        return chunks
    
    
    def load_and_chunk_pdf(self, file_path: str) -> List[Dict]:
        """
        Convenience method: Load PDF and chunk in one step.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            List of text chunks with metadata
            
        Interview Note:
        "This is the main entry point for document processing.
         In production, I'd add error handling, progress bars (tqdm),
         and possibly async processing for multiple PDFs."
        """
        # Extract text
        text = self.load_pdf(file_path)
        
        # Prepare metadata
        metadata = {
            'source': Path(file_path).name,
            'source_path': str(file_path),
            'total_chars': len(text)
        }
        
        # Chunk text
        chunks = self.chunk_text(text, metadata)
        
        return chunks


# ==============================================================================
# DEMO & TESTING
# ==============================================================================

def demo_document_loading():
    """
    Demo script showing how document loading works.
    
    Run this to test: python minimal_rag/1_load_documents.py
    """
    print("="*70)
    print(" "*20 + "DOCUMENT LOADING DEMO")
    print("="*70)
    
    # Initialize loader
    loader = DocumentLoader(chunk_size=1000, chunk_overlap=100)
    
    # Test with sample text (since we may not have a PDF yet)
    print("\n" + "="*70)
    print("TEST 1: Chunking Sample Text")
    print("="*70)
    
    sample_text = """
    Type 2 diabetes is a chronic condition that affects the way your body 
    metabolizes sugar (glucose). With type 2 diabetes, your body either resists 
    the effects of insulin or doesn't produce enough insulin to maintain normal 
    glucose levels.
    
    Symptoms of type 2 diabetes often develop slowly. You might not notice them 
    at first. Common symptoms include increased thirst, frequent urination, 
    increased hunger, unintended weight loss, fatigue, blurred vision, slow-healing 
    sores, and frequent infections.
    
    Treatment for type 2 diabetes includes managing your blood sugar levels through 
    diet, exercise, and medications. Monitoring your blood sugar is essential. 
    Regular physical activity and a healthy diet are the foundation of diabetes 
    management.
    """ * 3  # Repeat to make longer text
    
    chunks = loader.chunk_text(sample_text, metadata={'source': 'sample_diabetes.txt'})
    
    # Display first 3 chunks
    print(f"\n📋 Showing first 3 chunks:\n")
    for i, chunk in enumerate(chunks[:3]):
        print(f"Chunk {i+1}:")
        print(f"  ID: {chunk['chunk_id']}")
        print(f"  Position: {chunk['start_pos']}-{chunk['end_pos']}")
        print(f"  Length: {chunk['char_count']} chars")
        print(f"  Text preview: {chunk['text'][:100]}...")
        print()
    
    # Show overlap
    if len(chunks) >= 2:
        print("="*70)
        print("OVERLAP DEMONSTRATION")
        print("="*70)
        chunk1_end = chunks[0]['text'][-50:]
        chunk2_start = chunks[1]['text'][:50]
        
        print(f"Chunk 1 ends with:\n  ...{chunk1_end}\n")
        print(f"Chunk 2 starts with:\n  {chunk2_start}...\n")
        print("👆 Notice the overlapping text!")
    
    print("\n" + "="*70)
    print("KEY TAKEAWAYS")
    print("="*70)
    print("""
    1. ✅ PDFs are extracted page-by-page to text
    2. ✅ Text is split into fixed-size chunks (1000 chars)
    3. ✅ Chunks overlap (100 chars) to preserve context
    4. ✅ Each chunk has metadata (source, position, etc.)
    
    In Interview:
    "I implemented document loading from scratch using PyPDF2 to understand
     how text extraction works. The key challenge was choosing chunk size -
     too small loses context, too large exceeds LLM token limits. I chose
     1000 chars based on LLaMA-3.1's 8K token limit."
    """)


if __name__ == "__main__":
    demo_document_loading()
