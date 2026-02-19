#!/usr/bin/env python3
"""Dump academic PDF(s) to plain text or markdown format.

Extracts text from PDF files with optional structure detection and metadata.
Supports single file or batch processing.

Usage:
    # Single PDF to plain text
    python examples/academic/dump_pdf_text.py input.pdf
    
    # Single PDF to markdown with structure
    python examples/academic/dump_pdf_text.py input.pdf --markdown --output output.md
    
    # Batch process directory
    python examples/academic/dump_pdf_text.py --dir data_raw/academic_papers/ --output-dir extracted_text/
    
    # Include metadata
    python examples/academic/dump_pdf_text.py input.pdf --metadata --markdown
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.ingest.pdfparser import extract_text_from_pdf, extract_pdf_metadata, extract_structure_from_text
from scripts.utils.logger import create_module_logger

get_logger, audit = create_module_logger("pdf_dump")


def format_metadata(metadata: Dict[str, Any]) -> str:
    """Format PDF metadata as readable text.
    
    Args:
        metadata: Dictionary with title, author, year, etc.
        
    Returns:
        Formatted metadata string
    """
    lines = []
    
    if metadata.get('title'):
        lines.append(f"Title: {metadata['title']}")
    if metadata.get('author'):
        lines.append(f"Author: {metadata['author']}")
    if metadata.get('year'):
        lines.append(f"Year: {metadata['year']}")
    if metadata.get('subject'):
        lines.append(f"Subject: {metadata['subject']}")
    if metadata.get('keywords'):
        lines.append(f"Keywords: {metadata['keywords']}")
    
    return '\n'.join(lines)


def format_as_markdown(
    text: str,
    metadata: Dict[str, Any] | None = None,
    structure: List[Dict[str, Any]] | None = None,
    filename: str | None = None,
) -> str:
    """Format extracted text as markdown with optional structure.
    
    Args:
        text: Extracted PDF text
        metadata: Optional PDF metadata
        structure: Optional document structure (chapters/sections)
        filename: Optional source filename for title
        
    Returns:
        Markdown formatted text
    """
    lines = []
    
    # Add title from metadata or filename
    if metadata and metadata.get('title'):
        lines.append(f"# {metadata['title']}")
        lines.append("")
    elif filename:
        lines.append(f"# {Path(filename).stem}")
        lines.append("")
    
    # Add metadata section
    if metadata:
        meta_text = format_metadata(metadata)
        if meta_text:
            lines.append("## Document Metadata")
            lines.append("")
            for line in meta_text.split('\n'):
                lines.append(f"- {line}")
            lines.append("")
            lines.append("---")
            lines.append("")
    
    # Add structure if available
    if structure and len(structure) > 0:
        lines.append("## Document Structure")
        lines.append("")
        
        for section in structure:
            level = section.get('level', 1)
            title = section.get('title', 'Untitled')
            section_type = section.get('section_type', 'chapter')
            
            # Use markdown heading level (add 2 to offset from title and metadata)
            heading_level = min(level + 2, 6)
            lines.append(f"{'#' * heading_level} {title}")
            
            if section_type != 'chapter':
                lines.append(f"*({section_type})*")
            lines.append("")
        
        lines.append("---")
        lines.append("")
    
    # Add main content
    lines.append("## Content")
    lines.append("")
    lines.append(text)
    
    return '\n'.join(lines)


def dump_pdf(
    pdf_path: Path,
    output_path: Path | None = None,
    markdown: bool = False,
    include_metadata: bool = False,
    include_structure: bool = False,
) -> bool:
    """Extract and dump text from a single PDF.
    
    Args:
        pdf_path: Path to PDF file
        output_path: Optional output path (defaults to <pdf_name>.txt or .md)
        markdown: Format as markdown
        include_metadata: Include PDF metadata
        include_structure: Extract and include document structure
        
    Returns:
        True if successful, False otherwise
    """
    logger = get_logger()
    
    if not pdf_path.exists():
        logger.error(f"PDF not found: {pdf_path}")
        return False
    
    if not pdf_path.suffix.lower() == '.pdf':
        logger.error(f"Not a PDF file: {pdf_path}")
        return False
    
    try:
        # Extract text
        logger.info(f"Extracting text from {pdf_path.name}...")
        text = extract_text_from_pdf(str(pdf_path))
        
        if not text or not text.strip():
            logger.error(f"No text extracted from {pdf_path}")
            return False
        
        logger.info(f"Extracted {len(text):,} characters")
        
        # Extract metadata if requested
        metadata = None
        if include_metadata or markdown:
            try:
                metadata = extract_pdf_metadata(str(pdf_path))
                logger.info(f"Extracted metadata: {metadata.get('title', 'No title')}")
            except Exception as e:
                logger.warning(f"Failed to extract metadata: {e}")
                metadata = {}
        
        # Extract structure if requested
        structure = None
        if include_structure or markdown:
            try:
                structure = extract_structure_from_text(text)
                if structure:
                    logger.info(f"Extracted {len(structure)} structural sections")
            except Exception as e:
                logger.warning(f"Failed to extract structure: {e}")
                structure = None
        
        # Format output
        if markdown:
            output_text = format_as_markdown(text, metadata, structure, str(pdf_path))
        else:
            parts = []
            if include_metadata and metadata:
                parts.append("=" * 80)
                parts.append("METADATA")
                parts.append("=" * 80)
                parts.append(format_metadata(metadata))
                parts.append("")
                parts.append("=" * 80)
                parts.append("CONTENT")
                parts.append("=" * 80)
                parts.append("")
            
            parts.append(text)
            output_text = '\n'.join(parts)
        
        # Determine output path
        if output_path is None:
            ext = '.md' if markdown else '.txt'
            output_path = pdf_path.with_suffix(ext)
        
        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write output
        output_path.write_text(output_text, encoding='utf-8')
        logger.info(f"Written to {output_path}")
        
        # Log summary
        print(f"✓ {pdf_path.name}")
        print(f"  → {len(text):,} characters")
        if metadata and metadata.get('title'):
            print(f"  → Title: {metadata['title'][:60]}...")
        if structure:
            print(f"  → Structure: {len(structure)} sections")
        print(f"  → Output: {output_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to process {pdf_path}: {e}", exc_info=True)
        print(f"✗ {pdf_path.name}: {e}")
        return False


def dump_directory(
    input_dir: Path,
    output_dir: Path | None = None,
    markdown: bool = False,
    include_metadata: bool = False,
    include_structure: bool = False,
) -> tuple[int, int]:
    """Process all PDFs in a directory.
    
    Args:
        input_dir: Directory containing PDFs
        output_dir: Output directory (defaults to input_dir)
        markdown: Format as markdown
        include_metadata: Include PDF metadata
        include_structure: Extract and include document structure
        
    Returns:
        Tuple of (success_count, total_count)
    """
    logger = get_logger()
    
    if not input_dir.exists():
        logger.error(f"Directory not found: {input_dir}")
        return 0, 0
    
    # Find all PDFs
    pdf_files = list(input_dir.rglob('*.pdf'))
    
    if not pdf_files:
        logger.warning(f"No PDF files found in {input_dir}")
        return 0, 0
    
    logger.info(f"Found {len(pdf_files)} PDF files")
    print(f"\nProcessing {len(pdf_files)} PDFs from {input_dir}")
    print("=" * 80)
    
    # Process each PDF
    success_count = 0
    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"\n[{i}/{len(pdf_files)}]", end=" ")
        
        # Determine output path
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            ext = '.md' if markdown else '.txt'
            output_path = output_dir / pdf_path.with_suffix(ext).name
        else:
            output_path = None
        
        if dump_pdf(pdf_path, output_path, markdown, include_metadata, include_structure):
            success_count += 1
    
    print("\n" + "=" * 80)
    print(f"Processed {success_count}/{len(pdf_files)} PDFs successfully")
    
    return success_count, len(pdf_files)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Dump academic PDF(s) to plain text or markdown format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single PDF to plain text
  %(prog)s thesis.pdf
  
  # Single PDF to markdown with structure
  %(prog)s thesis.pdf --markdown --output thesis.md
  
  # Batch process directory
  %(prog)s --dir data_raw/academic_papers/ --output-dir extracted_text/
  
  # Include metadata in output
  %(prog)s thesis.pdf --metadata --markdown
        """,
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "pdf",
        nargs="?",
        type=Path,
        help="Path to single PDF file",
    )
    input_group.add_argument(
        "--dir",
        type=Path,
        help="Directory containing PDF files (processes all PDFs)",
    )
    
    # Output options
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output file path (for single PDF) or directory (for batch)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (for batch processing)",
    )
    
    # Format options
    parser.add_argument(
        "--markdown",
        "-m",
        action="store_true",
        help="Format output as markdown with structure",
    )
    parser.add_argument(
        "--metadata",
        action="store_true",
        help="Include PDF metadata in output",
    )
    parser.add_argument(
        "--structure",
        action="store_true",
        help="Extract and include document structure (chapters/sections)",
    )
    
    args = parser.parse_args()
    
    # Process single PDF or directory
    if args.pdf:
        # Single PDF mode
        output_path = args.output
        success = dump_pdf(
            args.pdf,
            output_path,
            markdown=args.markdown,
            include_metadata=args.metadata,
            include_structure=args.structure,
        )
        return 0 if success else 1
    
    elif args.dir:
        # Directory mode
        output_dir = args.output_dir or args.output
        success_count, total_count = dump_directory(
            args.dir,
            output_dir,
            markdown=args.markdown,
            include_metadata=args.metadata,
            include_structure=args.structure,
        )
        return 0 if success_count == total_count else 1
    
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
