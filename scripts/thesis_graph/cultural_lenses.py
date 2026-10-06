"""Load and validate cultural lens profiles for thesis processing."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


def load_cultural_lens_profile(
    profile_id: str,
    profiles_dir: Path,
    environment: str,
) -> dict[str, Any]:
    """Load a schema-valid profile and enforce environment-specific status rules.

    Draft and under-review profiles are restricted to Dev. Approved profiles may be
    used in any supported environment. Retired profiles cannot be applied.

    Args:
        profile_id: Identifier of the cultural lens profile to load.
        profiles_dir: Directory containing the profile JSON files and schema.
        environment: Current environment (e.g., "dev", "test", "prod").

    Returns:
        The loaded and validated cultural lens profile as a dictionary.
    """
    if not re.fullmatch(r"[a-z0-9_]+", profile_id):
        raise ValueError("Cultural lens profile ID must contain lowercase letters, digits or '_'.")

    normalised_environment = environment.strip().casefold()
    if normalised_environment not in {"dev", "test", "prod"}:
        raise ValueError(f"Unsupported environment for cultural lens: {environment!r}.")

    profile_path = profiles_dir / f"{profile_id}.json"
    schema_path = profiles_dir / "cultural_lens.schema.json"
    with profile_path.open(encoding="utf-8") as profile_file:
        profile = json.load(profile_file)
    with schema_path.open(encoding="utf-8") as schema_file:
        schema = json.load(schema_file)

    Draft202012Validator.check_schema(schema)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(profile),
        key=lambda error: list(error.absolute_path),
    )
    if errors:
        location = ".".join(str(part) for part in errors[0].absolute_path) or "profile"
        raise ValueError(
            f"Invalid cultural lens profile {profile_id!r} at {location}: {errors[0].message}"
        )
    if profile.get("profile_id") != profile_id:
        raise ValueError(
            f"Cultural lens profile ID mismatch: requested {profile_id!r}, "
            f"file declares {profile.get('profile_id')!r}."
        )

    status = profile["status"]
    if status == "retired":
        raise ValueError(f"Cultural lens profile {profile_id!r} is retired and cannot be applied.")
    if status != "approved" and normalised_environment != "dev":
        raise ValueError(
            f"Cultural lens profile {profile_id!r} has status {status!r}; "
            "draft and under-review profiles are restricted to ENVIRONMENT=Dev."
        )

    return profile


def get_cultural_lens_guidance_for_sources(
    sources: list[dict[str, Any]],
    *,
    registry_path: Path,
    profiles_dir: Path,
    environment: str,
) -> str | None:
    """Return the explicitly assigned lens guidance for one thesis' sources.

    Args:
        sources: A list of source dictionaries associated with a thesis.
        registry_path: Path to the thesis registry.
        profiles_dir: Directory containing cultural lens profiles.
        environment: The current environment (e.g., "dev", "test", "prod").

    Returns:
        A string containing the cultural lens guidance if available, otherwise None.
    """
    thesis_ids = {
        str(source.get("thesis_id") or source.get("doc_id"))
        for source in sources
        if isinstance(source, dict)
        and source.get("source_kind") == "thesis_document"
        and (source.get("thesis_id") or source.get("doc_id"))
    }
    if len(thesis_ids) != 1:
        return None
    thesis_id = next(iter(thesis_ids))
    profile = get_assigned_cultural_lens_profile(
        thesis_id,
        registry_path=registry_path,
        profiles_dir=profiles_dir,
        environment=environment,
    )
    if profile is None:
        return None

    answer_guidance = profile.get("answer_guidance") or []
    if not answer_guidance:
        return None

    draft_notice = (
        "DRAFT CULTURAL LENS: development use only. Do not treat this guidance as approved.\n"
        if profile["status"] != "approved"
        else ""
    )
    guidance = "\n".join(f"- {instruction}" for instruction in answer_guidance)
    return (
        f"{draft_notice}CULTURAL LENS: {profile['name']} (version {profile['version']})\n"
        f"Apply this guidance only to the thesis it is assigned to.\n{guidance}"
    )


def get_assigned_cultural_lens_profile(
    thesis_id: str,
    *,
    registry_path: Path,
    profiles_dir: Path,
    environment: str,
) -> dict[str, Any] | None:
    """Load a thesis' explicitly assigned profile after environment/version checks.

    Args:
        thesis_id: The ID of the thesis whose cultural lens profile is to be loaded.
        registry_path: Path to the thesis registry.
        profiles_dir: Directory containing cultural lens profiles.
        environment: The current environment (e.g., "dev", "test", "prod").

    Returns:
        The assigned cultural lens profile as a dictionary if available and valid, otherwise None.
    """
    if not registry_path.exists():
        return None

    from scripts.thesis_graph.thesis_registry import ThesisRegistry

    thesis = ThesisRegistry(registry_path).get_thesis(thesis_id)
    if thesis is None or not thesis.get("cultural_lens_id"):
        return None

    profile = load_cultural_lens_profile(thesis["cultural_lens_id"], profiles_dir, environment)
    if thesis.get("cultural_lens_version") != profile.get("version") or thesis.get(
        "cultural_lens_status"
    ) != profile.get("status"):
        raise ValueError(
            f"Cultural lens profile for thesis {thesis_id!r} changed after ingestion; "
            "re-ingest the thesis before using this lens."
        )
    return profile
