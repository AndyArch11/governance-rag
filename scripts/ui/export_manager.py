"""
Export Manager for RAG Dashboard

Handles exporting conversations, query results, and analyses to various formats
including PDF, Markdown, and CSV.
"""

import io
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
    from reportlab.lib.pagesizes import A4, letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        PageBreak,
        Paragraph,
        Preformatted,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


class ExportManager:
    """Manager for exporting RAG dashboard data to various formats."""

    @staticmethod
    def export_conversation_markdown(
        conversation_id: str,
        turns: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Export conversation to Markdown format.

        Args:
            conversation_id: Unique conversation identifier
            turns: List of conversation turns with query/answer/sources
            metadata: Optional metadata (creation time, title, etc.)

        Returns:
            Markdown-formatted string
        """
        lines = []
        lines.append(f"# Conversation: {conversation_id}\n")

        if metadata:
            lines.append("## Metadata\n")
            if "created_at" in metadata:
                lines.append(f"- **Created:** {metadata['created_at']}")
            if "title" in metadata:
                lines.append(f"- **Title:** {metadata['title']}")
            if "total_turns" in metadata:
                lines.append(f"- **Total Turns:** {metadata['total_turns']}")
            lines.append("")

        lines.append("---\n")

        for i, turn in enumerate(turns, 1):
            lines.append(f"## Turn {i}\n")

            # Query
            query = turn.get("query", "")
            lines.append(f"**Query:** {query}\n")

            # Answer
            answer = turn.get("answer", "")
            lines.append(f"**Answer:**\n\n{answer}\n")

            # Sources
            sources = turn.get("sources", [])
            if sources:
                lines.append(f"**Sources ({len(sources)}):**\n")
                for j, source in enumerate(sources, 1):
                    doc_id = source.get("doc_id", "Unknown")
                    distance = source.get("distance", 0)
                    lines.append(f"{j}. `{doc_id}` (distance: {distance:.4f})")
                lines.append("")

            # Metadata
            if "generation_time" in turn or "total_time" in turn:
                lines.append("**Metrics:**")
                if "generation_time" in turn:
                    lines.append(f"- Generation Time: {turn['generation_time']:.2f}s")
                if "total_time" in turn:
                    lines.append(f"- Total Time: {turn['total_time']:.2f}s")
                lines.append("")

            lines.append("---\n")

        return "\n".join(lines)

    @staticmethod
    def export_conversation_pdf(
        conversation_id: str,
        turns: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
        output_path: Optional[str] = None,
    ) -> bytes:
        """Export conversation to PDF format using ReportLab.

        Args:
            conversation_id: Unique conversation identifier
            turns: List of conversation turns
            metadata: Optional metadata
            output_path: Optional file path to save PDF

        Returns:
            PDF as bytes

        Raises:
            ImportError: If reportlab is not installed
        """
        if not REPORTLAB_AVAILABLE:
            raise ImportError(
                "reportlab is required for PDF export. " "Install with: pip install reportlab"
            )

        # Create PDF buffer
        buffer = io.BytesIO()

        # Create document
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18,
        )

        # Build story (content)
        story = []
        styles = getSampleStyleSheet()

        # Custom styles
        title_style = ParagraphStyle(
            "CustomTitle",
            parent=styles["Heading1"],
            fontSize=18,
            textColor=colors.HexColor("#2c3e50"),
            spaceAfter=12,
            alignment=TA_CENTER,
        )

        heading_style = ParagraphStyle(
            "CustomHeading",
            parent=styles["Heading2"],
            fontSize=14,
            textColor=colors.HexColor("#34495e"),
            spaceAfter=8,
        )

        body_style = ParagraphStyle(
            "CustomBody",
            parent=styles["BodyText"],
            fontSize=10,
            alignment=TA_JUSTIFY,
            spaceAfter=6,
        )

        code_style = ParagraphStyle(
            "CustomCode",
            parent=styles["Code"],
            fontSize=8,
            fontName="Courier",
            leftIndent=12,
            rightIndent=12,
            spaceAfter=6,
            backColor=colors.HexColor("#f5f5f5"),
        )

        # Title
        story.append(Paragraph(f"RAG Conversation: {conversation_id}", title_style))
        story.append(Spacer(1, 0.2 * inch))

        # Metadata
        if metadata:
            meta_data = []
            if "created_at" in metadata:
                meta_data.append(["Created:", metadata["created_at"]])
            if "title" in metadata:
                meta_data.append(["Title:", metadata["title"]])
            if "total_turns" in metadata:
                meta_data.append(["Total Turns:", str(metadata["total_turns"])])

            if meta_data:
                table = Table(meta_data, colWidths=[1.5 * inch, 4.5 * inch])
                table.setStyle(
                    TableStyle(
                        [
                            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                            ("FONTSIZE", (0, 0), (-1, -1), 9),
                            ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#555")),
                            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                        ]
                    )
                )
                story.append(table)
                story.append(Spacer(1, 0.3 * inch))

        # Conversation turns
        for i, turn in enumerate(turns, 1):
            # Turn header
            story.append(Paragraph(f"Turn {i}", heading_style))
            story.append(Spacer(1, 0.1 * inch))

            # Query
            query = turn.get("query", "")
            story.append(Paragraph(f"<b>Query:</b> {query}", body_style))
            story.append(Spacer(1, 0.1 * inch))

            # Answer
            answer = turn.get("answer", "")
            # Clean up markdown formatting for PDF
            answer_clean = answer.replace("**", "").replace("*", "")
            # Handle code blocks
            if "```" in answer_clean:
                parts = answer_clean.split("```")
                for j, part in enumerate(parts):
                    if j % 2 == 0:  # Regular text
                        if part.strip():
                            story.append(Paragraph(part.strip(), body_style))
                    else:  # Code block
                        # Remove language identifier (first line)
                        code_lines = part.split("\n", 1)
                        code = code_lines[1] if len(code_lines) > 1 else code_lines[0]
                        story.append(Preformatted(code.strip(), code_style))
            else:
                story.append(Paragraph(f"<b>Answer:</b>", body_style))
                story.append(Paragraph(answer_clean, body_style))

            story.append(Spacer(1, 0.1 * inch))

            # Sources
            sources = turn.get("sources", [])
            if sources:
                story.append(Paragraph(f"<b>Sources ({len(sources)}):</b>", body_style))
                source_data = []
                for j, source in enumerate(sources[:10], 1):  # Limit to 10 sources
                    doc_id = source.get("doc_id", "Unknown")
                    distance = source.get("distance", 0)
                    source_data.append([f"{j}.", doc_id, f"{distance:.4f}"])

                if source_data:
                    source_table = Table(source_data, colWidths=[0.3 * inch, 4.2 * inch, 1 * inch])
                    source_table.setStyle(
                        TableStyle(
                            [
                                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                                ("FONTSIZE", (0, 0), (-1, -1), 8),
                                ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#333")),
                                ("ALIGN", (0, 0), (0, -1), "RIGHT"),
                                ("ALIGN", (2, 0), (2, -1), "RIGHT"),
                                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#ddd")),
                            ]
                        )
                    )
                    story.append(source_table)

            # Metrics
            if "generation_time" in turn or "total_time" in turn:
                metrics = []
                if "generation_time" in turn:
                    metrics.append(f"Generation: {turn['generation_time']:.2f}s")
                if "total_time" in turn:
                    metrics.append(f"Total: {turn['total_time']:.2f}s")

                metrics_text = " | ".join(metrics)
                story.append(Spacer(1, 0.05 * inch))
                story.append(
                    Paragraph(f"<font size=8 color='#777'>{metrics_text}</font>", body_style)
                )

            # Separator between turns
            if i < len(turns):
                story.append(Spacer(1, 0.2 * inch))
                story.append(Paragraph("<hr/>", body_style))
                story.append(Spacer(1, 0.1 * inch))

        # Build PDF
        doc.build(story)

        # Get PDF bytes
        pdf_bytes = buffer.getvalue()
        buffer.close()

        # Optionally save to file
        if output_path:
            Path(output_path).write_bytes(pdf_bytes)

        return pdf_bytes

    @staticmethod
    def export_batch_conversations(
        conversations: List[Dict[str, Any]],
        format: str = "markdown",
        output_path: Optional[str] = None,
    ) -> bytes:
        """Export multiple conversations to a ZIP archive.

        Args:
            conversations: List of conversation dicts with id, turns, metadata
            format: Export format - "markdown" or "pdf"
            output_path: Optional path to save ZIP file

        Returns:
            ZIP file as bytes
        """
        buffer = io.BytesIO()

        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
            for conv in conversations:
                conv_id = conv.get("id", "unknown")
                turns = conv.get("turns", [])
                metadata = conv.get("metadata", {})

                if format == "markdown":
                    content = ExportManager.export_conversation_markdown(conv_id, turns, metadata)
                    filename = f"{conv_id}.md"
                    zipf.writestr(filename, content)

                elif format == "pdf":
                    if not REPORTLAB_AVAILABLE:
                        raise ImportError("reportlab required for PDF export")

                    content = ExportManager.export_conversation_pdf(conv_id, turns, metadata)
                    filename = f"{conv_id}.pdf"
                    zipf.writestr(filename, content)

        zip_bytes = buffer.getvalue()
        buffer.close()

        if output_path:
            Path(output_path).write_bytes(zip_bytes)

        return zip_bytes

    @staticmethod
    def export_results_table_csv(
        sources: List[Dict[str, Any]],
        output_path: Optional[str] = None,
    ) -> str:
        """Export query results/sources to CSV format.

        Args:
            sources: List of source dicts with metadata
            output_path: Optional path to save CSV

        Returns:
            CSV as string
        """
        import csv

        buffer = io.StringIO()

        if not sources:
            return ""

        # Get all unique keys from sources
        fieldnames = set()
        for source in sources:
            fieldnames.update(source.keys())

        fieldnames = sorted(fieldnames)

        writer = csv.DictWriter(buffer, fieldnames=fieldnames)
        writer.writeheader()

        for source in sources:
            writer.writerow(source)

        csv_content = buffer.getvalue()
        buffer.close()

        if output_path:
            Path(output_path).write_text(csv_content)

        return csv_content
