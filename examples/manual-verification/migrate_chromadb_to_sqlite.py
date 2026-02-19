#!/usr/bin/env python3
"""
Migration script: ChromaDB → SQLite

Converts existing ChromaDB data to SQLite backend while preserving all data and functionality.
Safe to run multiple times - performs idempotent migration with validation.

Usage:
    python migrate_chromadb_to_sqlite.py [--backup] [--validate]
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.ingest.chromadb_sqlite import ChromaSQLiteClient, ChromaSQLiteCollection


def backup_chromadb(chromadb_path: str) -> str:
    """Create backup of ChromaDB directory."""
    import shutil
    from datetime import datetime

    chromadb_dir = Path(chromadb_path)
    if not chromadb_dir.exists():
        print(f"⚠️  ChromaDB directory not found: {chromadb_path}")
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = chromadb_dir.parent / f"chromadb_backup_{timestamp}"

    print(f"📦 Creating backup: {backup_dir}")
    shutil.copytree(chromadb_dir, backup_dir)
    print(f"✅ Backup created: {backup_dir}")

    return str(backup_dir)


def migrate_chromadb_to_sqlite(
    chromadb_path: str, sqlite_db_path: str, backup: bool = True, validate: bool = True
) -> Dict[str, Any]:
    """
    Migrate ChromaDB data to SQLite.

    Args:
        chromadb_path: Path to ChromaDB directory
        sqlite_db_path: Path to SQLite database file
        backup: Whether to create backup before migrating
        validate: Whether to validate after migration

    Returns:
        Migration statistics
    """
    try:
        from chromadb import PersistentClient as ChromaDBClient
    except ImportError:
        print("❌ ChromaDB not installed. Install with: pip install chromadb")
        sys.exit(1)

    stats = {
        "collections": [],
        "total_documents": 0,
        "total_chunks": 0,
        "errors": [],
        "status": "pending",
    }

    # Backup
    if backup:
        backup_path = backup_chromadb(chromadb_path)
        stats["backup_path"] = backup_path

    # Connect to both databases
    print("\n🔄 Connecting to ChromaDB and SQLite...")
    chroma_client = ChromaDBClient(path=chromadb_path)
    sqlite_client = ChromaSQLiteClient(db_path=sqlite_db_path)

    # Get all collections from ChromaDB
    chroma_collections = chroma_client.list_collections()
    print(f"📊 Found {len(chroma_collections)} ChromaDB collections")

    # Migrate each collection
    for collection_name in chroma_collections:
        print(f"\n📋 Migrating collection: {collection_name}")

        try:
            # Get ChromaDB collection
            chroma_collection = chroma_client.get_collection(collection_name)
            count = chroma_collection.count()
            print(f"   Documents: {count}")

            # Create SQLite collection
            sqlite_collection = sqlite_client.get_or_create_collection(collection_name)

            # Migrate in batches
            batch_size = 100
            migrated = 0

            for batch_start in range(0, count, batch_size):
                batch_end = min(batch_start + batch_size, count)

                # Get batch from ChromaDB
                results = chroma_collection.get(
                    limit=batch_size,
                    offset=batch_start,
                    include=["documents", "metadatas", "embeddings"],
                )

                # Add to SQLite
                if results["ids"]:
                    sqlite_collection.add(
                        ids=results["ids"],
                        documents=results.get("documents"),
                        metadatas=results.get("metadatas"),
                        embeddings=results.get("embeddings"),
                    )

                migrated = min(batch_end, count)
                progress = (migrated / count) * 100
                print(f"   Progress: {migrated}/{count} ({progress:.1f}%)", end="\r")

            print(f"   ✅ Migrated {migrated} documents")

            stats["collections"].append({
                "name": collection_name,
                "documents": migrated,
                "status": "success",
            })

            if "chunk" in collection_name:
                stats["total_chunks"] += migrated
            elif "document" in collection_name:
                stats["total_documents"] += migrated

        except Exception as e:
            error_msg = f"Failed to migrate {collection_name}: {str(e)}"
            print(f"   ❌ {error_msg}")
            stats["errors"].append(error_msg)

    # Validate migration
    if validate:
        print("\n🔍 Validating migration...")
        validation_passed = True

        for collection_info in stats["collections"]:
            collection_name = collection_info["name"]
            expected_count = collection_info["documents"]

            try:
                sqlite_collection = sqlite_client.get_collection(collection_name)
                actual_count = sqlite_collection.count()

                if actual_count == expected_count:
                    print(f"   ✅ {collection_name}: {actual_count} documents")
                else:
                    print(
                        f"   ❌ {collection_name}: Expected {expected_count}, got {actual_count}"
                    )
                    validation_passed = False
            except Exception as e:
                print(f"   ❌ {collection_name}: Validation failed - {str(e)}")
                validation_passed = False

        if validation_passed:
            print("   ✅ All collections validated successfully")
            stats["status"] = "success"
        else:
            print("   ⚠️  Some collections failed validation")
            stats["status"] = "partial_success"

    else:
        stats["status"] = "success"

    # Summary
    print("\n" + "=" * 60)
    print("📊 MIGRATION SUMMARY")
    print("=" * 60)
    print(f"Collections: {len(stats['collections'])}")
    print(f"Total Documents: {stats['total_documents']}")
    print(f"Total Chunks: {stats['total_chunks']}")
    print(f"Errors: {len(stats['errors'])}")
    print(f"Status: {stats['status'].upper()}")
    print(f"SQLite Database: {sqlite_db_path}")
    print(f"Database Size: {Path(sqlite_db_path).stat().st_size / 1024 / 1024:.2f} MB")
    print("=" * 60)

    if stats["errors"]:
        print("\n⚠️  ERRORS:")
        for error in stats["errors"]:
            print(f"  - {error}")

    # Close connections
    chroma_client.close()
    sqlite_client.close()

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Migrate ChromaDB to SQLite backend",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python migrate_chromadb_to_sqlite.py --backup --validate
  python migrate_chromadb_to_sqlite.py \\
    --chromadb-path /custom/path/chromadb \\
    --sqlite-path /custom/path/rag.db
        """,
    )

    parser.add_argument(
        "--chromadb-path",
        default="~/rag-project/rag_data/chromadb",
        help="Path to ChromaDB directory (default: ~/rag-project/rag_data/chromadb)",
    )

    parser.add_argument(
        "--sqlite-path",
        default="~/rag-project/rag_data/chromadb.db",
        help="Path to SQLite database file (default: ~/rag-project/rag_data/chromadb.db)",
    )

    parser.add_argument(
        "--backup",
        action="store_true",
        default=True,
        help="Create backup of ChromaDB before migration (default: True)",
    )

    parser.add_argument(
        "--no-backup",
        dest="backup",
        action="store_false",
        help="Skip backup creation",
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        default=True,
        help="Validate migration after completion (default: True)",
    )

    parser.add_argument(
        "--no-validate",
        dest="validate",
        action="store_false",
        help="Skip validation",
    )

    args = parser.parse_args()

    # Expand paths
    chromadb_path = str(Path(args.chromadb_path).expanduser())
    sqlite_path = str(Path(args.sqlite_path).expanduser())

    print("🚀 ChromaDB → SQLite Migration Tool")
    print(f"📁 ChromaDB: {chromadb_path}")
    print(f"📁 SQLite:   {sqlite_path}")

    # Run migration
    stats = migrate_chromadb_to_sqlite(chromadb_path, sqlite_path, args.backup, args.validate)

    # Exit with appropriate code
    sys.exit(0 if stats["status"] in ["success", "partial_success"] else 1)


if __name__ == "__main__":
    main()
