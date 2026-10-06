"""Environment and schema checks for cultural lens profiles."""

import json
from pathlib import Path

import pytest

from scripts.thesis_graph.cultural_lenses import (
    get_cultural_lens_guidance_for_sources,
    load_cultural_lens_profile,
)
from scripts.thesis_graph.thesis_registry import ThesisRegistry

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROFILE_DIRECTORY = REPOSITORY_ROOT / "rag_data" / "cultural_lenses"


@pytest.mark.parametrize("status", ["draft", "under_review"])
def test_development_environment_can_load_unapproved_profile(status: str, tmp_path: Path) -> None:
    profile = json.loads(
        (PROFILE_DIRECTORY / "aboriginal_torres_strait_islander.json").read_text(encoding="utf-8")
    )
    profile["status"] = status
    (tmp_path / "cultural_lens.schema.json").write_text(
        (PROFILE_DIRECTORY / "cultural_lens.schema.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (tmp_path / f"{profile['profile_id']}.json").write_text(json.dumps(profile), encoding="utf-8")

    loaded = load_cultural_lens_profile(profile["profile_id"], tmp_path, "Dev")

    assert loaded["status"] == status


@pytest.mark.parametrize("environment", ["Test", "Prod"])
@pytest.mark.parametrize("status", ["draft", "under_review"])
def test_unapproved_profile_is_rejected_outside_development(environment: str, status: str) -> None:
    with pytest.raises(ValueError, match="restricted to ENVIRONMENT=Dev"):
        load_cultural_lens_profile(
            "aboriginal_torres_strait_islander", PROFILE_DIRECTORY, environment
        )


def test_approved_profile_can_be_loaded_in_production(tmp_path: Path) -> None:
    profile = json.loads(
        (PROFILE_DIRECTORY / "aboriginal_torres_strait_islander.json").read_text(encoding="utf-8")
    )
    profile["status"] = "approved"
    (tmp_path / "cultural_lens.schema.json").write_text(
        (PROFILE_DIRECTORY / "cultural_lens.schema.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (tmp_path / f"{profile['profile_id']}.json").write_text(json.dumps(profile), encoding="utf-8")

    loaded = load_cultural_lens_profile(profile["profile_id"], tmp_path, "Prod")

    assert loaded["status"] == "approved"


def test_invalid_profile_id_is_rejected_before_path_resolution() -> None:
    with pytest.raises(ValueError, match="profile ID"):
        load_cultural_lens_profile("../private", PROFILE_DIRECTORY, "Dev")


def test_lens_guidance_uses_registered_profile_for_one_thesis(tmp_path: Path) -> None:
    profile_path = PROFILE_DIRECTORY / "aboriginal_torres_strait_islander.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile_dir = tmp_path / "cultural_lenses"
    profile_dir.mkdir()
    (profile_dir / "cultural_lens.schema.json").write_text(
        (PROFILE_DIRECTORY / "cultural_lens.schema.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (profile_dir / f"{profile['profile_id']}.json").write_text(
        json.dumps(profile), encoding="utf-8"
    )
    registry_path = tmp_path / "thesis_graphs" / "registry.sqlite"
    ThesisRegistry(registry_path).register_thesis(
        thesis_id="thesis-one",
        title="Study of Healing",
        authors=["A. Author"],
        source_path=tmp_path / "thesis.pdf",
        file_hash="fixture-hash",
        graph_path=tmp_path / "thesis_graphs" / "thesis-one.sqlite",
        cultural_lens_id=profile["profile_id"],
        cultural_lens_version=profile["version"],
        cultural_lens_status=profile["status"],
    )

    guidance = get_cultural_lens_guidance_for_sources(
        [{"source_kind": "thesis_document", "thesis_id": "thesis-one"}],
        registry_path=registry_path,
        profiles_dir=profile_dir,
        environment="Dev",
    )

    assert guidance is not None
    assert "draft cultural lens" in guidance.lower()
    assert "Country" in guidance


def test_lens_guidance_is_not_applied_to_mixed_theses(tmp_path: Path) -> None:
    guidance = get_cultural_lens_guidance_for_sources(
        [
            {"source_kind": "thesis_document", "thesis_id": "thesis-one"},
            {"source_kind": "thesis_document", "thesis_id": "thesis-two"},
        ],
        registry_path=tmp_path / "missing-registry.sqlite",
        profiles_dir=tmp_path / "missing-profiles",
        environment="Dev",
    )

    assert guidance is None
