#!/usr/bin/env python3
"""Direct citation extraction for testing."""

from pypdf import PdfReader

from scripts.ingest.academic.parser import extract_citations

pdf_path = "/workspaces/governance-rag/data_raw/academic_papers/Author Final.pdf"
output_path = "/workspaces/governance-rag/data_raw/extracted_citations.txt"

# Read PDF
reader = PdfReader(pdf_path)
full_text = "".join(page.extract_text() for page in reader.pages)

# Extract citations
citations = extract_citations(full_text)

# Save to file
with open(output_path, "w") as f:
    for citation in citations:
        f.write(citation.raw_text + "\n")

print(f"Extracted {len(citations)} citations to {output_path}")
