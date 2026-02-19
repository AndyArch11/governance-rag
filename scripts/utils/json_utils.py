"""JSON utility functions for safe encoding and escaping.

Provides common JSON operations needed across modules:
- Sanitisation of strings for safe JSON embedding
- Handling of backslashes and quotes
- JSON validation helpers
- Repair of malformed JSON from LLM outputs
- Extraction of JSON blocks from text
"""

import json
import re
from typing import Any, Dict


def repair_json(text: str, attempt: int = 1, max_attempts: int = 3) -> str:
    """Repair common JSON formatting issues produced by LLMs.

    Handles:
    - Missing commas between fields
    - Missing commas between objects/arrays
    - Trailing commas before closing braces/brackets
    - Extra whitespace and indentation
    - Unescaped backslashes in string values
    - Single quotes instead of double quotes
    - Missing colons in key-value pairs
    - Malformed boolean/null values
    - Unescaped quotes within string values
    - Missing closing braces/brackets

    Args:
        text: Raw text potentially containing malformed JSON.
        attempt: Current repair attempt (1-indexed, default 1).
        max_attempts: Maximum repair attempts before failing (default 3).

    Returns:
        str: Repaired JSON string.

    Raises:
        json.JSONDecodeError: If repair fails after max_attempts.
    """
    # Remove all leading/trailing whitespace outside the JSON structure
    text = text.strip()

    # Remove leading whitespace/indentation from each line to normalise
    lines = text.split("\n")
    min_indent = float("inf")
    for line in lines:
        if line.strip():  # Only check non-empty lines
            indent = len(line) - len(line.lstrip())
            if indent < min_indent:
                min_indent = indent

    if min_indent != float("inf") and min_indent > 0:
        # Remove the common indentation
        text = "\n".join(line[min_indent:] if len(line) >= min_indent else line for line in lines)

    # Strategy 1: Fix unescaped backslashes that are not part of valid JSON escape sequences
    # This handles cases like \$ → \\$ or :\${VAR} → :\\${VAR}
    # We escape all backslashes that aren't already part of valid escape sequences (\", \\, \/, \b, \f, \n, \r, \t, \uXXXX)
    # Pattern: find backslashes that aren't already escaped and aren't followed by valid escape characters
    # Valid escape chars in JSON: " \ / b f n r t u
    text = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", text)  # Escape unescaped backslashes
    text = re.sub(
        r'([^\\])\\([^\\"])', r"\1\\\\\2", text
    )  # Also catch unescaped backslashes preceded by non-backslash

    # Strategy 2: Fix improperly formatted boolean/null values (capitalisation) BEFORE fixing commas
    text = re.sub(r"\bTrue\b", "true", text)
    text = re.sub(r"\bFalse\b", "false", text)
    text = re.sub(r"\bNone\b", "null", text)
    text = re.sub(r"\bNULL\b", "null", text)

    # Strategy 3: Remove trailing commas before closing brace (do this early to prevent false comma insertions)
    text = re.sub(r",\s*}", "}", text)

    # Strategy 4: Remove trailing commas before closing bracket
    text = re.sub(r",\s*\]", "]", text)

    # Strategy 5: Remove trailing commas followed by other closing structures
    text = re.sub(r",\s*(\n\s*[}\]])", r"\1", text)

    # Strategy 6: Fix missing commas between non-string values and next field (numbers, booleans, null, objects, arrays)
    # true\n  "next": → true,\n  "next":
    # Do this BEFORE string quote fixes to avoid conflicts
    text = re.sub(r'(true|false|null|[0-9])\s*\n\s*"', r'\1,\n    "', text, flags=re.IGNORECASE)
    text = re.sub(r'(\]|\})\s*\n\s*"', r'\1,\n    "', text)

    # Strategy 7: Fix missing commas between string fields on different lines:
    # "value"\n      "next": → "value",\n      "next":
    text = re.sub(r'"\s*\n\s*"', '",\n    "', text)

    # Strategy 8: Fix missing commas between objects on different lines:
    # "value"}\n      "next": → "value"},\n      "next":
    text = re.sub(r'}\s*\n\s*"', '},\n    "', text)

    # Strategy 9: Fix missing colons after keys (conservative - only key-like patterns)
    # "key" value → "key": value
    text = re.sub(r'"([a-zA-Z_][a-zA-Z0-9_]*)"\s+(?=[{\["\-0-9tfn])', r'"\1": ', text)

    # Strategy 10: Single quotes conversion - only around identifiers and simple values (conservative)
    # 'key': 'value' → "key": "value"
    text = re.sub(r"'([a-zA-Z_][a-zA-Z0-9_]*)'\s*:", r'"\1":', text)

    # Strategy 11: Balance closing braces/brackets if missing
    # Count open and close braces/brackets
    open_braces = text.count("{") - text.count("}")
    open_brackets = text.count("[") - text.count("]")
    if open_braces > 0:
        text = text.rstrip() + ("}" * open_braces)
    if open_brackets > 0:
        text = text.rstrip() + ("]" * open_brackets)

    return text


def sanitise_for_json(text: str) -> str:
    """Escape special characters for safe JSON embedding.

    Escapes backslashes and quotes to ensure the text can be safely embedded
    as a JSON string value without causing parsing errors.

    Args:
        text: Raw text that may contain JSON-breaking characters.

    Returns:
        Escaped text safe for JSON strings.

    Example:
        >>> sanitise_for_json('endpoints: direct:\\${VAR}')
        'endpoints: direct:\\\\${VAR}'
        >>> sanitise_for_json('quoted "text"')
        'quoted \\"text\\"'
    """
    return text.replace("\\", "\\\\").replace('"', '\\"')


def extract_first_json_block(text: str, max_repair_attempts: int = 3) -> Dict[str, Any]:
    """Extract and parse the first valid JSON object from a string.

    Scans for balanced braces to extract valid JSON even if the model outputs
    extra text before or after. Attempts repair if initial parsing fails using
    multiple strategies to handle common LLM JSON formatting issues.

    Args:
        text: Raw text potentially containing JSON and extra content.
        max_repair_attempts: Maximum repair attempts (default 3).

    Returns:
        Dict[str, Any]: Parsed JSON object.

    Raises:
        ValueError: If no balanced JSON object is found or JSON parsing fails
                   after all repair attempts.

    Example:
        >>> text = 'Here is the result: {"status": "ok"} Done!'
        >>> extract_first_json_block(text)
        {'status': 'ok'}
    """
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model output")

    brace_count = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        char = text[i]

        if char == '"' and not escape:
            in_string = not in_string

        if not in_string:
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1

            if brace_count == 0:
                json_str = text[start : i + 1]
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError as e:
                    # Attempt repair of JSON (includes escaping unescaped backslashes like \${VAR})
                    try:
                        repaired = repair_json(
                            json_str, attempt=1, max_attempts=max_repair_attempts
                        )
                        return json.loads(repaired)
                    except json.JSONDecodeError as e2:
                        # Strategy 2: Try to extract from first { to last }
                        last_brace = json_str.rfind("}")
                        if last_brace > start:
                            json_str_trimmed = json_str[: last_brace + 1]
                            try:
                                repaired = repair_json(
                                    json_str_trimmed, attempt=1, max_attempts=max_repair_attempts
                                )
                                return json.loads(repaired)
                            except json.JSONDecodeError as e3:
                                pass  # Continue to next strategy

                        # Strategy 3: Try partial JSON recovery - find last valid key-value pair
                        try:
                            # Try to find the last complete field and truncate there
                            last_quote = json_str.rfind('"')
                            if last_quote > 0:
                                # Look backwards for a closing bracket/brace before the last quote
                                for j in range(last_quote, -1, -1):
                                    if json_str[j] in "]}":
                                        partial_json = json_str[: j + 1]
                                        try:
                                            repaired = repair_json(partial_json)
                                            return json.loads(repaired)
                                        except json.JSONDecodeError:
                                            pass
                        except Exception:
                            pass

                        # Strategy 4: Remove problematic trailing content
                        # Sometimes LLM outputs extra text after closing brace
                        try:
                            json_str_cleaned = re.sub(r"}\s*[^}\]]*$", "}", json_str)
                            repaired = repair_json(json_str_cleaned)
                            return json.loads(repaired)
                        except json.JSONDecodeError:
                            pass

                        # Strategy 5: One more aggressive repair attempt
                        try:
                            repaired = repair_json(
                                json_str, attempt=1, max_attempts=max_repair_attempts
                            )
                            # Try to salvage by parsing with strict=False (Python 3.9+)
                            return json.loads(repaired, strict=False)
                        except (json.JSONDecodeError, TypeError):
                            # Last resort: log the problematic JSON for debugging
                            raise ValueError(
                                f"Unable to parse JSON after repair attempts: {e}\n"
                                f"Original JSON (first 500 chars):\n{json_str[:500]}"
                            )

        escape = char == "\\" and not escape

    raise ValueError("Unbalanced JSON braces in model output")
