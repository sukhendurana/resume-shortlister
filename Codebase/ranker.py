"""Stage 3: candidate pool, ranking, interactive criteria refinement and export.

`RankingSession` caches every per-requirement LLM assessment. Re-weighting,
removing requirements or filtering by skill therefore re-sorts instantly with no
new LLM calls; only a *newly added* requirement triggers one scoring call per CV.
"""
import csv
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from extractor import parse_resume
from guardrails import InjectionDetected, ensure_clean
from llm_client import CreditsExhaustedError
from matcher import aggregate, assess_candidate, keyword_coverage


@dataclass
class Candidate:
    """One resume plus everything the pipeline learned about it."""

    file: str
    text: str
    profile: dict = field(default_factory=dict)
    assessments: dict = field(default_factory=dict)  # requirement id -> assessment
    overall: float = None       # LLM holistic fit (0-100) for the original JD
    summary: str = ""
    strengths: list = field(default_factory=list)
    gaps: list = field(default_factory=list)
    coverage: float = None      # keyword coverage (0-1)
    error: str = None           # set when the LLM pipeline failed for this CV


class RankingSession:
    """Runs the two-model pipeline over all CVs and serves (re-)rankings."""

    def __init__(self, client, extract_model, score_model, job, resumes, workers=4):
        """Create candidates from {file: text} and remember the structured job.

        Args:
            client: object exposing chat_json(model, system, user, max_tokens).
            extract_model: Model A id (parsing); score_model: Model B id (matching).
            job: normalised job dict from extractor.parse_job_description.
            workers: number of CVs processed concurrently.
        """
        self.client, self.extract_model, self.score_model = client, extract_model, score_model
        self.job = job
        self.requirements = [dict(r) for r in job["requirements"]]
        self._next_id = len(self.requirements) + 1
        self.workers = max(1, workers)
        self.filters = []
        self._fatal = None  # set once the account runs out of credits: stop spending calls
        self.candidates = {
            name: Candidate(file=name, text=text,
                            coverage=keyword_coverage(job["keywords"], text))
            for name, text in resumes.items()
        }
        for cand in self.candidates.values():  # block suspicious CVs before any model call
            try:
                ensure_clean(cand.text, f"{cand.file}")
            except InjectionDetected as err:
                cand.error = str(err)

    # ---------------------------------------------------------------- pipeline
    def _evaluate(self, cand):
        """Parse one resume (Model A) then score it against all requirements (Model B)."""
        if self._fatal or cand.error:  # blocked CVs never reach a model
            return
        try:
            cand.profile = parse_resume(self.client, self.extract_model, cand.text)
            result = assess_candidate(self.client, self.score_model, self.job["job_title"],
                                      self.requirements, cand.profile, cand.text)
        except CreditsExhaustedError as err:
            self._fatal = err
            return
        except Exception as err:  # keep the batch alive; surface the failure in the table
            cand.error = str(err)[:200]
            return
        cand.assessments = result["requirements"]
        cand.overall, cand.summary = result["overall"], result["summary"]
        cand.strengths, cand.gaps = result["strengths"], result["gaps"]
        print(f"  [done] {cand.file}", file=sys.stderr)

    def _map(self, func, items):
        """Run `func` over `items` on a thread pool (LLM calls are I/O bound)."""
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            list(pool.map(func, items))

    def evaluate_all(self):
        """Process every candidate concurrently.

        Raises CreditsExhaustedError (after in-flight calls finish) if credits run out;
        completed responses are already cached, so a re-run only pays for the rest.
        """
        self._map(self._evaluate, self.candidates.values())
        if self._fatal:
            raise self._fatal

    # --------------------------------------------------------- criteria changes
    def add_requirement(self, text, weight=3, must_have=False):
        """Add a requirement and score only that requirement for every CV; return its id."""
        ensure_clean(text, "the new requirement")
        req = {"id": f"R{self._next_id}", "text": text, "category": "skill",
               "weight": max(1, min(5, weight)), "must_have": must_have}
        self._next_id += 1

        def score_new(cand):
            """Score the new requirement for one candidate (skips failed CVs)."""
            if cand.error:
                return
            try:
                res = assess_candidate(self.client, self.score_model, self.job["job_title"],
                                       [req], cand.profile, cand.text)
                cand.assessments.update(res["requirements"])
            except CreditsExhaustedError as err:
                self._fatal = err
            except Exception as err:
                print(f"  [warn] {cand.file}: {err}", file=sys.stderr)

        self._fatal = None
        self._map(score_new, self.candidates.values())
        if self._fatal:
            raise self._fatal
        self.requirements.append(req)
        return req["id"]

    def get_requirement(self, req_id):
        """Return the requirement dict for `req_id` (case-insensitive) or None."""
        return next((r for r in self.requirements if r["id"].lower() == req_id.lower()), None)

    def remove_requirement(self, req_id):
        """Drop a requirement from the ranking; returns True if it existed."""
        req = self.get_requirement(req_id)
        if req and len(self.requirements) > 1:
            self.requirements.remove(req)
            return True
        return False

    # ----------------------------------------------------------------- filters
    @staticmethod
    def _mentions(cand, term):
        """True if `term` appears as a whole word in the CV text or parsed skills."""
        pattern = rf"(?<![\w+#]){re.escape(term.lower())}(?![\w+#])"
        return bool(re.search(pattern, cand.text.lower())) or term.lower() in cand.profile.get(
            "skills", [])

    # ----------------------------------------------------------------- ranking
    def ranked(self, top_k=None):
        """Return rows sorted by final score (best first), honouring skill filters."""
        rows = []
        for cand in self.candidates.values():
            if not all(self._mentions(cand, term) for term in self.filters):
                continue
            rows.append(self._row(cand))
        rows.sort(key=lambda r: (r["final"] is None, -(r["final"] or 0), -(r["req_score"] or 0),
                                 r["file"]))
        for position, row in enumerate(rows, 1):
            row["rank"] = position
        return rows[:top_k] if top_k else rows

    def _row(self, cand):
        """Build the display/export record for one candidate under current criteria."""
        row = {"file": cand.file, "name": cand.profile.get("name") or "", "error": cand.error,
               "final": None, "req_score": None, "overall": cand.overall,
               "coverage": cand.coverage, "missed": [], "summary": cand.summary,
               "strengths": cand.strengths, "gaps": cand.gaps, "requirements": []}
        if cand.error or not cand.assessments:
            return row
        row["final"], row["req_score"], row["missed"] = aggregate(
            self.requirements, cand.assessments, cand.overall, cand.coverage)
        for req in self.requirements:
            item = cand.assessments.get(req["id"])
            row["requirements"].append({
                **{k: req[k] for k in ("id", "text", "weight", "must_have")},
                "score": item["score"] if item else None,
                "evidence": item["evidence"] if item else "",
                "evidence_verified": item["evidence_verified"] if item else False,
                "reason": item["reason"] if item else "not assessed",
            })
        return row

    # ------------------------------------------------------------------ export
    def export(self, output_dir):
        """Write ranking.csv (summary) and ranking.json (full evidence); return both paths."""
        folder = Path(output_dir)
        folder.mkdir(parents=True, exist_ok=True)
        rows = self.ranked()
        csv_path, json_path = folder / "ranking.csv", folder / "ranking.json"
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["rank", "file", "name", "final_score", "requirement_score",
                             "llm_overall", "keyword_coverage_pct", "missed_must_haves",
                             "summary", "error"])
            for r in rows:
                writer.writerow([
                    r["rank"], r["file"], r["name"], r["final"], r["req_score"],
                    r["overall"], None if r["coverage"] is None else round(100 * r["coverage"]),
                    " ".join(r["missed"]), r["summary"], r["error"] or ""])
        payload = {"job": {**self.job, "requirements": self.requirements},
                   "filters": self.filters, "ranking": rows}
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return csv_path, json_path
