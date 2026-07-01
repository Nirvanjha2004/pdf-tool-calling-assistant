"""Print the raw chunks stored in memory to see what OCR/pypdfium2 extracted."""
import sys
sys.path.insert(0, '.')

from tools.pdf_search import _chunks, _document_loaded

if not _document_loaded:
    print("No document loaded — start the server and upload a PDF first, then run this.")
else:
    print(f"Total chunks: {len(_chunks)}\n")
    for c in _chunks:
        print(f"--- Chunk {c['index']+1} ---")
        print(c['text'])
        print()
