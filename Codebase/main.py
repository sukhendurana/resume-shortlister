"""CV Sorting using LLMs - entry point.

Pipeline: read CVs + job description -> Model A structures them (extractor.py) ->
Model B scores every requirement with verified evidence (matcher.py) -> weighted
aggregation, ranking, export and interactive refinement (ranker.py).

Example:
    export HF_TOKEN=hf_xxx
    python main.py --cv-dir ./cvs --jd ./job_description.txt --interactive
"""
import argparse
import os
import sys
import textwrap

from document_loader import load_resumes, read_document
from extractor import parse_job_description
from llm_client import ROUTER_URL, HFChatClient, LLMError
from ranker import RankingSession

DEFAULT_EXTRACT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"   # Model A: fast, strong JSON extraction
DEFAULT_SCORE_MODEL = "microsoft/phi-4"              # Model B: stronger reasoning for scoring

HELP = """Commands:
  list [n]               show ranking (top n)
  show <rank|file>       per-requirement scores, evidence and explanation
  reqs                   list current requirements
  weight <Rn> <1-5>      change importance of a requirement
  must <Rn> | optional <Rn>   mark requirement mandatory / optional
  add <text> [;weight=N] [;must]   add a new requirement (scores each CV once)
  remove <Rn>            drop a requirement
  require <skill>        keep only CVs that mention <skill>   (unrequire clears)
  export                 write ranking.csv / ranking.json
  help | quit"""


def parse_args(argv=None):
    """Define and parse the command-line interface."""
    p = argparse.ArgumentParser(description="Rank CVs against a job description with LLMs.")
    p.add_argument("--cv-dir", help="folder with CVs (.pdf/.docx/.txt/.md)")
    p.add_argument("--jd", help="job description file (.pdf/.docx/.txt/.md)")
    p.add_argument("--list-models", nargs="?", const="", metavar="FILTER",
                   help="print chat models available to your token (optional name filter) and exit")
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"),
                   help="Hugging Face token (default: HF_TOKEN environment variable)")
    p.add_argument("--extract-model", default=DEFAULT_EXTRACT_MODEL,
                   help=f"Model A for parsing (default: {DEFAULT_EXTRACT_MODEL})")
    p.add_argument("--score-model", default=DEFAULT_SCORE_MODEL,
                   help=f"Model B for matching (default: {DEFAULT_SCORE_MODEL})")
    p.add_argument("--api-url", default=ROUTER_URL,
                   help="OpenAI-compatible chat endpoint (default: Hugging Face router)")
    p.add_argument("--no-cache", action="store_true",
                   help="disable the on-disk LLM response cache (<output-dir>/llm_cache.jsonl)")
    p.add_argument("--workers", type=int, default=4, help="CVs processed in parallel")
    p.add_argument("--top-k", type=int, default=None, help="only display the best k CVs")
    p.add_argument("--require-skills", default="",
                   help="comma-separated skills every shortlisted CV must mention")
    p.add_argument("--output-dir", default="./output", help="where ranking.csv/json are written")
    p.add_argument("--interactive", action="store_true", help="refine criteria after ranking")
    args = p.parse_args(argv)
    if args.list_models is None and not (args.cv_dir and args.jd):
        p.error("--cv-dir and --jd are required (unless --list-models is used)")
    return args


def print_table(rows):
    """Print the ranking as a fixed-width table."""
    if not rows:
        print("No candidates match the current filters.")
        return
    width = min(31, max(len(r["file"]) for r in rows)) + 2  # CV column fits the longest name
    print(f"\n{'Rank':<5}{'Score':>5} {'Reqs':>5} {'LLM':>4} {'KW%':>4}  {'CV':<{width}}Missed must-haves")
    for r in rows:
        if r["final"] is None:
            status = "BLOCKED" if (r["error"] or "").startswith("BLOCKED") else "FAILED"
            print(f"{r['rank']:<5}{'-':>5}  {status}: {r['file']} ({r['error'] or 'no assessment'})")
            continue
        kw = "-" if r["coverage"] is None else f"{100 * r['coverage']:.0f}"
        llm = "-" if r["overall"] is None else f"{r['overall']:.0f}"
        print(f"{r['rank']:<5}{r['final']:>5.1f} {r['req_score']:>5.1f} {llm:>4} {kw:>4}  "
              f"{r['file'][:31]:<{width}}{' '.join(r['missed']) or '-'}")


def print_detail(row):
    """Print the full explanation for one candidate (transparency for recruiters)."""
    print(f"\n#{row['rank']} {row['file']}  {row['name']}  final={row['final']}")
    print(textwrap.fill(row["summary"] or "(no summary)", 100))
    for req in row["requirements"]:
        flag = "*" if req["must_have"] else " "
        score = "n/a" if req["score"] is None else f"{req['score']:.0f}/10"
        verified = "" if req["evidence_verified"] else " (unverified)"
        print(f" {req['id']:<4}{flag}w{req['weight']} {score:>6}  {req['text']}")
        print(f"        reason: {req['reason']}")
        if req["evidence"]:
            print(f"        evidence{verified}: \"{req['evidence']}\"")
    print("  strengths:", "; ".join(row["strengths"]) or "-")
    print("  gaps:     ", "; ".join(row["gaps"]) or "-")


def print_requirements(session):
    """List the current requirements with weights ('*' = must-have)."""
    for r in session.requirements:
        print(f" {r['id']:<4}{'*' if r['must_have'] else ' '}w{r['weight']}  {r['text']}")


def _find_row(session, key):
    """Resolve a rank number or (partial) file name to a ranking row."""
    rows = session.ranked()
    if key.isdigit():
        return next((r for r in rows if r["rank"] == int(key)), None)
    return next((r for r in rows if key.lower() in r["file"].lower()), None)


def _parse_add(arg):
    """Split 'text ;weight=4 ;must' into (text, weight, must_have)."""
    parts = [p.strip() for p in arg.split(";")]
    weight, must = 3, False
    for opt in parts[1:]:
        if opt.startswith("weight="):
            weight = int(opt[7:]) if opt[7:].isdigit() else 3
        elif opt == "must":
            must = True
    return parts[0], weight, must


def handle_command(session, line, top_k, output_dir):
    """Execute one interactive command; return False when the user quits."""
    cmd, _, arg = line.strip().partition(" ")
    arg = arg.strip()
    cmd = cmd.lower()
    if cmd in ("quit", "exit", "q"):
        return False
    if cmd == "help":
        print(HELP)
    elif cmd == "list":
        print_table(session.ranked(int(arg) if arg.isdigit() else top_k))
    elif cmd == "show":
        row = _find_row(session, arg)
        print_detail(row) if row else print("No such candidate.")
    elif cmd == "reqs":
        print_requirements(session)
    elif cmd in ("weight", "must", "optional"):
        parts = arg.split()
        req = session.get_requirement(parts[0]) if parts else None
        if not req:
            print("Unknown requirement id (see 'reqs').")
        else:
            if cmd == "weight" and len(parts) == 2 and parts[1].isdigit():
                req["weight"] = max(1, min(5, int(parts[1])))
            elif cmd != "weight":
                req["must_have"] = cmd == "must"
            print_table(session.ranked(top_k))
    elif cmd == "add" and arg:
        text, weight, must = _parse_add(arg)
        print(f"Scoring all CVs on new requirement {session.add_requirement(text, weight, must)} ...")
        print_table(session.ranked(top_k))
    elif cmd == "remove":
        print("Removed." if session.remove_requirement(arg) else "Cannot remove (unknown id / last one).")
        print_table(session.ranked(top_k))
    elif cmd == "require" and arg:
        session.filters.append(arg.lower())
        print_table(session.ranked(top_k))
    elif cmd == "unrequire":
        session.filters.clear()
        print_table(session.ranked(top_k))
    elif cmd == "export":
        print("Wrote:", *session.export(output_dir))
    else:
        print("Unknown command. Type 'help'.")
    return True


def main():
    """Run the full ranking pipeline and (optionally) the interactive refinement loop."""
    args = parse_args()
    try:
        cache_path = None
        if not args.no_cache and args.list_models is None:
            os.makedirs(args.output_dir, exist_ok=True)
            cache_path = os.path.join(args.output_dir, "llm_cache.jsonl")
        client = HFChatClient(args.hf_token, api_url=args.api_url, cache_path=cache_path)
        if args.list_models is not None:
            print("\n".join(client.list_models(args.list_models)) or "No matching models.")
            return
        resumes = load_resumes(args.cv_dir)
        print(f"Parsing job description with {args.extract_model} ...", file=sys.stderr)
        job = parse_job_description(client, args.extract_model, read_document(args.jd))
        session = RankingSession(client, args.extract_model, args.score_model, job, resumes,
                                 args.workers)
        session.filters = [s.strip().lower() for s in args.require_skills.split(",") if s.strip()]
        print(f"Scoring {len(resumes)} CVs for '{job['job_title']}' with "
              f"{args.score_model} ({len(job['requirements'])} requirements) ...", file=sys.stderr)
        session.evaluate_all()
    except (LLMError, ValueError) as err:
        sys.exit(f"Error: {err}")

    print_requirements(session)
    print_table(session.ranked(args.top_k))
    print("Wrote:", *session.export(args.output_dir))

    if args.interactive:
        print("\nInteractive mode. " + HELP)
        while True:
            try:
                line = input("\n> ")
            except (EOFError, KeyboardInterrupt):
                break
            try:
                if line.strip() and not handle_command(session, line, args.top_k, args.output_dir):
                    break
            except (LLMError, ValueError) as err:  # e.g. credits exhausted / blocked text; keep going
                print(f"Error: {err}")


if __name__ == "__main__":
    main()
