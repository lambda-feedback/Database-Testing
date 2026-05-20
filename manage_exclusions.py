import argparse
import csv
import sys
from datetime import datetime, timezone

from google.cloud import firestore

from config import logger
from firestore_client import get_firestore_client


def _get_doc_ref(db: firestore.Client, eval_function_name: str):
    return db.collection("excluded-submissions").document(eval_function_name)


def _handle_add(db: firestore.Client, args: argparse.Namespace) -> None:
    ids_to_add = set(args.ids or [])

    if args.from_csv:
        try:
            with open(args.from_csv, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    sid = (row.get("submission_id") or "").strip()
                    if sid:
                        ids_to_add.add(sid)
        except FileNotFoundError:
            logger.error(f"CSV file not found: {args.from_csv}")
            sys.exit(1)

    if not ids_to_add:
        print("No IDs provided. Use --ids or --from_csv.")
        sys.exit(1)

    ids_list = list(ids_to_add)
    doc_ref = _get_doc_ref(db, args.eval_function_name)
    doc_ref.set(
        {"ids": firestore.ArrayUnion(ids_list), "updated_at": datetime.now(timezone.utc)},
        merge=True,
    )
    print(f"Added {len(ids_list)} ID(s) to excluded-submissions/{args.eval_function_name}.")


def _handle_remove(db: firestore.Client, args: argparse.Namespace) -> None:
    ids_to_remove = list(set(args.ids))
    doc_ref = _get_doc_ref(db, args.eval_function_name)
    doc_ref.set(
        {"ids": firestore.ArrayRemove(ids_to_remove), "updated_at": datetime.now(timezone.utc)},
        merge=True,
    )
    print(f"Removed {len(ids_to_remove)} ID(s) from excluded-submissions/{args.eval_function_name}.")


def _handle_list(db: firestore.Client, args: argparse.Namespace) -> None:
    snapshot = _get_doc_ref(db, args.eval_function_name).get()
    if not snapshot.exists:
        print(f"No exclusions configured for {args.eval_function_name}.")
        return
    ids = snapshot.get("ids") or []
    updated_at = snapshot.get("updated_at")
    print(f"{len(ids)} excluded submission ID(s) for {args.eval_function_name}:")
    for sid in ids:
        print(f"  {sid}")
    if updated_at:
        print(f"Last updated: {updated_at}")


def main() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description="Manage excluded submission IDs in Firestore.")
    parser.add_argument("--eval_function_name", required=True, help="Evaluation function name")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    add_parser = subparsers.add_parser("add", help="Add submission IDs to the exclusion list")
    add_parser.add_argument("--ids", nargs="*", default=[], metavar="UUID")
    add_parser.add_argument("--from_csv", default=None, metavar="PATH",
                            help="CSV file with a 'submission_id' column")

    remove_parser = subparsers.add_parser("remove", help="Remove submission IDs from the exclusion list")
    remove_parser.add_argument("--ids", nargs="+", required=True, metavar="UUID")

    subparsers.add_parser("list", help="List currently excluded submission IDs")

    args = parser.parse_args()

    if args.subcommand == "add" and not args.ids and not args.from_csv:
        parser.error("add requires at least one of --ids or --from_csv")

    db, _ = get_firestore_client()

    if args.subcommand == "add":
        _handle_add(db, args)
    elif args.subcommand == "remove":
        _handle_remove(db, args)
    elif args.subcommand == "list":
        _handle_list(db, args)


if __name__ == "__main__":
    main()
