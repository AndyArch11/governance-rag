# Git Ingestion Repository Configuration

## Overview

The `--repos-file` parameter supports two formats for specifying repositories to ingest:

1. **Text Format** (simple): One repository name per line
2. **JSON Format** (advanced): Per-repository configuration with branch and project overrides

## Text Format (Simple)

For basic use cases where all repos are in the same project and use the default branch:

```text
adapter-canvas-v1
app-shell-card-v1
adapter-siras-v1
```

Usage:
```bash
python -m scripts.ingest.ingest_git \
  --repos-file repos.txt \
  --project myproject \
  --branch main
```

## JSON Format (Advanced)

For complex scenarios where repos may:
- Use different branches (master vs main vs develop)
- Belong to different projects
- Require different per-repo settings

```json
[
  {
    "repo": "my-old_repo-v1",
    "project": "my-project",
    "branch": "master"
  },
  {
    "repo": "my-new-repo-v1",
    "project": "my-project",
    "branch": "main"
  },
  {
    "repo": "some-other-repo",
    "project": "my-other-project",
    "branch": "develop"
  }
]
```

### JSON Schema

Each repository object supports:

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `repo` | string | Yes | - | Repository slug/name |
| `project` | string | No | CLI `--project` | Project/workspace key |
| `branch` | string | No | CLI `--branch` | Git branch to clone |

### Usage

```bash
python -m scripts.ingest.ingest_git \
  --repos-file repos_config.json \
  --project myproject  # Default project if not specified per-repo
```

**Note:** When using JSON format, the CLI `--project` parameter is optional if all repos specify their own project. However, it's recommended to provide a default.

## Benefits of JSON Format

1. **Per-Repo Branch Control**: Clone `my-repo` from `main` while others use `master`
2. **Multi-Project Support**: Ingest repos from different Bitbucket projects in one run
3. **Future Extensibility**: Easy to add more per-repo settings without breaking compatibility
4. **Clear Documentation**: Self-documenting which branch each repo uses

## Migration from Text to JSON

Convert existing text file:

```bash
# repos.txt
my-old-repo-v1
my-new-repo-v1
```

To JSON (assuming all use master branch):

```json
[
  {"repo": "my-old--v1", "branch": "master"},
  {"repo": "my-new-repo", "branch": "master"}
]
```

The `project` field will use the CLI `--project` parameter if omitted.

## Auto-Detection

The system automatically detects which format you're using:
- If file starts with `[`, it's treated as JSON
- Otherwise, it's treated as text format (one repo per line)

This means you can use either format without changing your command-line arguments!
