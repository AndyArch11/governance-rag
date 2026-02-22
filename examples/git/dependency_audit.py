"""Dependency presence audit across ingested repositories.

This script scans the code chunks stored in ChromaDB and reports which
repositories contain specific dependency patterns (e.g., JDBC, AMQP/Artemis,
SMB/file shares). It illustrates how to craft simple metadata + content
filters to answer cross-repo dependency questions.

Usage examples (from repo root):
    python examples/git/dependency_audit.py --profile jdbc
    python examples/git/dependency_audit.py --profile amqp --verbose
    python examples/git/dependency_audit.py --profile fileshare --show-samples
    python examples/git/dependency_audit.py --profile all --limit 20000

Defaults:
- Uses the same ChromaDB path and collection name as ingestion (rag_data, chunk_collection).
- Scans only source_category='code' chunks.
- Chunk-level search is substring-based; add your own patterns as needed.

Profiles included:
- jdbc: JDBC drivers, java.sql usage, JdbcTemplate
- amqp: Artemis/AMQP clients
- fileshare: SMB/NFS/file-share style access

Notes:
- This is a lightweight content scan, not a full parser. For higher precision,
  refine the patterns or add metadata filters when ingesting.
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from chromadb import PersistentClient

# Profiles map to lists of case-insensitive substrings/regexes to test
PROFILES: Dict[str, Sequence[str]] = {
    "jdbc": [
        r"jdbc:",
        r"java\.sql",
        r"javax\.sql",
        r"drivermanager\.getconnection",
        r"datasource",
        r"jdbctemplate",
        r"spring-boot-starter-jdbc",
        r"jdbc-driver",
    ],
    "amqp": [
        r"jms",
        r"artemis",
        r"amq",
        r"amqp",
        r"activemq",
        r"qpid",
        r"rabbitmq" r"connectionfactory",
        r"queue",
    ],
    "fileshare": [
        r"smb",
        r"cifs",
        r"nfs",
        # r"\\\\[a-zA-Z0-9_.-]+\\\\",  # UNC paths like \\server\share
        r"smbclient",
        r"filename",
        r"share",
        r"fileshare",
    ],
    "kms": [
        r"kms",
        r"contact",
    ],
    "idempotent": [
        r"idempotent",
        r"@Idempotent",
        r"camel_messageprocesed",
        r"idempotentconsumer",
        r"idempotentrepository",
    ],
    "secrets": [
        r"keystore",
        r"trust",
        r"trustmanager",
        r"credentials",
        r"password",
    ],
}

DEFAULT_DB_PATH = "rag_data/chromadb"
DEFAULT_COLLECTION = "governance_docs_chunks"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find repositories with specific dependency patterns",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--profile",
        choices=["jdbc", "amqp", "fileshare", "kms", "idempotent", "secrets", "all"],
        default="jdbc",
        help="Pattern profile to search for (or 'all' to run every profile)",
    )
    parser.add_argument(
        "--db-path",
        default=DEFAULT_DB_PATH,
        help="Path to ChromaDB directory (default: rag_data/chromadb)",
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
        help="ChromaDB collection name (default: governance_docs_chunks)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10000,
        help="Max chunks to scan per profile (pagination in batches of 1000)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print file paths for each matched repository",
    )
    parser.add_argument(
        "--show-samples",
        action="store_true",
        help="Print a small sample of matching lines per repository",
    )
    return parser.parse_args()


def compile_patterns(patterns: Iterable[str]) -> List[re.Pattern]:
    return [re.compile(pat, re.IGNORECASE) for pat in patterns]


def chunk_matches(doc: str, compiled: List[re.Pattern]) -> bool:
    return any(p.search(doc) for p in compiled)


def scan_profile(
    client: PersistentClient,
    collection_name: str,
    profile_name: str,
    patterns: Sequence[str],
    limit: int,
    show_samples: bool,
    verbose: bool,
) -> None:
    collection = client.get_collection(collection_name)
    compiled = compile_patterns(patterns)

    repos = defaultdict(lambda: {"files": set(), "samples": []})

    batch = 1000
    scanned = 0
    offset = 0

    print(f"\n[PROFILE] {profile_name} — scanning up to {limit} chunks...")

    while scanned < limit:
        remaining = limit - scanned
        take = min(batch, remaining)

        results = collection.get(
            where={"source_category": "code"},
            limit=take,
            offset=offset,
            include=["documents", "metadatas"],
        )

        docs = results.get("documents") or []
        metas = results.get("metadatas") or []
        ids = results.get("ids") or []

        if not ids:
            break

        for doc, meta in zip(docs, metas):
            if not doc:
                continue
            if chunk_matches(doc, compiled):
                repo = meta.get("repository") or meta.get("source") or "unknown_repo"
                project = meta.get("project_key") or meta.get("project") or ""
                repo_key = f"{project}/{repo}" if project else repo
                file_path = meta.get("file_path") or meta.get("source") or ""
                repos[repo_key]["files"].add(file_path)

                if show_samples and len(repos[repo_key]["samples"]) < 3:
                    # Capture a small snippet around the first hit
                    lines = doc.splitlines()
                    hit_lines = [ln for ln in lines if chunk_matches(ln, compiled)]
                    snippet = " | ".join(hit_lines[:2]) if hit_lines else doc[:200]
                    repos[repo_key]["samples"].append(snippet)

        scanned += len(ids)
        offset += len(ids)
        if len(ids) < take:
            break

    # Print summary
    print(f"Found {len(repos)} repo(s) matching profile '{profile_name}'.")
    for repo_key in sorted(repos.keys()):
        info = repos[repo_key]
        print(f"- {repo_key}: {len(info['files'])} file(s)")
        if verbose:
            for path in sorted(info["files"]):
                print(f"    • {path}")
        if show_samples and info["samples"]:
            print("    Samples:")
            for sample in info["samples"]:
                print(f"      - {sample}")


def main() -> None:
    args = parse_args()

    client = PersistentClient(path=args.db_path)

    profiles_to_run = PROFILES.keys() if args.profile == "all" else [args.profile]

    for profile in profiles_to_run:
        scan_profile(
            client=client,
            collection_name=args.collection,
            profile_name=profile,
            patterns=PROFILES[profile],
            limit=args.limit,
            show_samples=args.show_samples,
            verbose=args.verbose,
        )


if __name__ == "__main__":
    main()
