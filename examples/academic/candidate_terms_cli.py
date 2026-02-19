"""CLI for reviewing and curating candidate terms.

Usage examples:
  python examples/academic/candidate_terms_cli.py list
  python examples/academic/candidate_terms_cli.py list --status pending --domain legal
  python examples/academic/candidate_terms_cli.py approve --term "data residency" --category compliance --weight 1.8 --curated-by analyst
  python examples/academic/candidate_terms_cli.py reject --term "foo" --reason "too generic"
  python examples/academic/candidate_terms_cli.py export --format csv --out candidate_terms.csv
"""

import argparse
import csv
import json
import sys
from typing import Optional, List, Dict

from scripts.rag.domain_terms import get_domain_term_manager, DomainType, resolve_domain_type


def _parse_domain(value: Optional[str]) -> Optional[DomainType]:
    if not value:
        return None
    resolved_value, display_name = resolve_domain_type(value)
    if resolved_value:
        try:
            return DomainType(resolved_value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid domain: {value}") from None
    raise argparse.ArgumentTypeError(f"Invalid domain: {value}")


def _write_csv(path: Optional[str], rows: List[Dict]) -> None:
    fieldnames = [
        "term",
        "domain",
        "source_doc_id",
        "frequency",
        "context",
        "suggested_weight",
        "status",
        "curated_by",
        "notes",
        "created_at",
    ]

    if path:
        with open(path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Optional[str], rows: List[Dict]) -> None:
    if path:
        with open(path, "w") as handle:
            json.dump(rows, handle, indent=2)
    else:
        print(json.dumps(rows, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Candidate term curation CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List candidate terms")
    list_parser.add_argument("--status", type=str, default="pending")
    list_parser.add_argument("--domain", type=_parse_domain, default=None)

    approve_parser = subparsers.add_parser("approve", help="Approve a candidate term")
    approve_parser.add_argument("--term", required=True)
    approve_parser.add_argument("--category", required=True)
    approve_parser.add_argument("--weight", type=float, default=1.5)
    approve_parser.add_argument("--curated-by", default="system")

    reject_parser = subparsers.add_parser("reject", help="Reject a candidate term")
    reject_parser.add_argument("--term", required=True)
    reject_parser.add_argument("--reason", default="")
    reject_parser.add_argument("--curated-by", default="system")

    export_parser = subparsers.add_parser("export", help="Export candidate terms")
    export_parser.add_argument("--status", type=str, default="pending")
    export_parser.add_argument("--domain", type=_parse_domain, default=None)
    export_parser.add_argument("--format", choices=["json", "csv"], default="json")
    export_parser.add_argument("--out", default=None, help="Output path (defaults to stdout)")

    args = parser.parse_args()
    manager = get_domain_term_manager()

    if args.command == "list":
        status = None if args.status == "all" else args.status
        terms = manager.get_candidate_terms(status=status, domain=args.domain)
        rows = [t.to_dict() for t in terms]
        print(json.dumps(rows, indent=2))
        return

    if args.command == "approve":
        ok = manager.approve_candidate_term(
            args.term,
            category=args.category,
            weight=args.weight,
            curated_by=args.curated_by,
        )
        if not ok:
            raise SystemExit(1)
        print(f"Approved: {args.term}")
        return

    if args.command == "reject":
        ok = manager.reject_candidate_term(
            args.term,
            reason=args.reason,
            curated_by=args.curated_by,
        )
        if not ok:
            raise SystemExit(1)
        print(f"Rejected: {args.term}")
        return

    if args.command == "export":
        status = None if args.status == "all" else args.status
        terms = manager.get_candidate_terms(status=status, domain=args.domain)
        rows = [t.to_dict() for t in terms]
        if args.format == "csv":
            _write_csv(args.out, rows)
        else:
            _write_json(args.out, rows)
        return


if __name__ == "__main__":
    main()
