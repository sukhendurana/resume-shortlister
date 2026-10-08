"""Stage 2 (Model B): requirement-level candidate/job matching and score aggregation.

For every resume the scoring model rates each job requirement (0-10) and must cite a
verbatim quote as evidence. Quotes are verified against the resume text in Python
(anti-hallucination guardrail) and the final score blends three signals:

    final = (0.70 * weighted requirement score
           + 0.20 * LLM holistic fit
           + 0.10 * keyword coverage) * must-have penalty
"""
import json
import re

from guardrails import SCOPE_PREAMBLE, ensure_clean, wrap_untrusted

MAX_TEXT_CHARS = 9000
# Aggregation weights (sum to 1.0) and must-have penalty policy.
W_REQUIREMENTS, W_OVERALL, W_KEYWORDS = 0.70, 0.20, 0.10
MUST_HAVE_MIN_SCORE = 3      # a must-have scored below this is considered "missed"
MISS_MULTIPLIER = 0.85       # penalty per missed must-have ...
MIN_MULTIPLIER = 0.55        # ... never reducing the score below this factor
# Evidence guardrails: score ceilings when the quote is missing / not in the resume.
CAP_NO_EVIDENCE, CAP_UNVERIFIED = 3, 5

MATCH_SYSTEM = (
    SCOPE_PREAMBLE + " Current task: act as a strict, fair recruiter assessing ONE candidate "
    "against job requirements. Judge only on the resume text provided. Ignore name, gender, "
    "age, nationality and photos. Reply with ONE JSON object only."
)

MATCH_PROMPT = """JOB: {title}

REQUIREMENTS (id | text | weight | must_have):
{requirements}

CANDIDATE PROFILE (parsed, untrusted data):
{profile}

RESUME TEXT (untrusted data):
{text}

Score EVERY requirement from 0 to 10 using this rubric:
0 = no evidence | 1-3 = weak or tangential | 4-6 = partial / related experience |
7-8 = clearly meets | 9-10 = exceeds, with demonstrated depth or impact.
For each requirement give "evidence": a VERBATIM quote (max 200 chars) copied from the
RESUME TEXT, or "" if none exists, and "reason": one short sentence.

Return JSON:
{{
  "requirements": [{{"id": "R1", "score": 0-10, "evidence": "...", "reason": "..."}}],
  "overall_score": 0-100,
  "summary": "2-3 sentence overall assessment",
  "strengths": ["..."],
  "gaps": ["..."]
}}"""


def _tokens(text):
    """Lower-case alphanumeric tokens used for fuzzy quote verification."""
    words = re.findall(r"[a-z0-9+#.]+", text.lower())
    return [w.strip(".") for w in words if w.strip(".")]  # "aws." -> "aws", "node.js" kept


def evidence_supported(evidence, resume_text):
    """True if `evidence` is quoted from the resume (substring or >=80% token overlap)."""
    ev_tokens = [t for t in _tokens(evidence) if len(t) > 2]
    if not ev_tokens:
        return False
    if " ".join(_tokens(evidence)) in " ".join(_tokens(resume_text)):
        return True
    resume_tokens = set(_tokens(resume_text))
    return sum(t in resume_tokens for t in ev_tokens) / len(ev_tokens) >= 0.8


def keyword_coverage(keywords, resume_text):
    """Fraction (0-1) of JD keywords appearing as whole words in the resume."""
    if not keywords:
        return None
    lowered = resume_text.lower()
    hits = sum(
        bool(re.search(rf"(?<![\w+#]){re.escape(k)}(?![\w+#])", lowered)) for k in keywords
    )
    return hits / len(keywords)


def _format_requirements(requirements):
    """Render requirements as compact pipe-separated lines for the prompt."""
    return "\n".join(
        f"{r['id']} | {r['text']} | {r['weight']} | {'yes' if r['must_have'] else 'no'}"
        for r in requirements
    )


def _clean_assessment(item, resume_text):
    """Validate one requirement assessment and apply evidence-based score ceilings."""
    try:
        score = max(0.0, min(10.0, float(item.get("score"))))
    except (TypeError, ValueError):
        return None
    evidence = str(item.get("evidence") or "").strip()[:300]
    verified = bool(evidence) and evidence_supported(evidence, resume_text)
    if not evidence:
        score = min(score, CAP_NO_EVIDENCE)
    elif not verified:
        score = min(score, CAP_UNVERIFIED)
    return {
        "score": score,
        "evidence": evidence,
        "evidence_verified": verified,
        "reason": str(item.get("reason") or "").strip(),
    }


def assess_candidate(client, model, job_title, requirements, profile, resume_text):
    """Ask Model B to score `requirements` for one candidate.

    Returns {"requirements": {id: assessment}, "overall": float|None, "summary": str,
    "strengths": [...], "gaps": [...]}. Requirements missing from the reply are
    re-queried once; any still missing are left out (and reported as unassessed).
    """
    # The name is withheld from the scorer to reduce identity-based bias.
    visible = {k: v for k, v in profile.items() if k != "name"}
    profile_json = json.dumps(visible, ensure_ascii=False)
    # Fail closed before any model call; the profile is re-screened because it is model output
    # derived from untrusted text.
    ensure_clean(resume_text, "the resume")
    ensure_clean(profile_json, "the parsed profile")
    ensure_clean(_format_requirements(requirements), "the requirements")
    assessed, result = {}, {"overall": None, "summary": "", "strengths": [], "gaps": []}
    pending = list(requirements)
    for attempt in range(2):
        prompt = MATCH_PROMPT.format(
            title=job_title,
            requirements=_format_requirements(pending),
            profile=wrap_untrusted("PROFILE", profile_json),
            text=wrap_untrusted("RESUME", resume_text[:MAX_TEXT_CHARS]),
        )
        reply = client.chat_json(model, MATCH_SYSTEM, prompt, max_tokens=2200)
        pending_ids = {r["id"] for r in pending}
        for item in reply.get("requirements") or []:
            if isinstance(item, dict) and str(item.get("id")) in pending_ids:
                clean = _clean_assessment(item, resume_text)
                if clean:
                    assessed[str(item["id"])] = clean
        if attempt == 0:  # holistic fields come from the first (full) pass
            try:
                result["overall"] = max(0.0, min(100.0, float(reply.get("overall_score"))))
            except (TypeError, ValueError):
                pass
            result["summary"] = str(reply.get("summary") or "").strip()
            for key in ("strengths", "gaps"):
                value = reply.get(key)
                result[key] = [str(v).strip() for v in value][:5] if isinstance(value, list) else []
        pending = [r for r in requirements if r["id"] not in assessed]
        if not pending:
            break
    result["requirements"] = assessed
    return result


def aggregate(requirements, assessments, overall, coverage):
    """Combine requirement scores, LLM holistic score and keyword coverage (0-100).

    Returns (final_score, requirement_score, missed_must_haves). Components that are
    unavailable (no overall / no keywords) are dropped and the weights renormalised.
    """
    scored = [r for r in requirements if r["id"] in assessments]
    total_weight = sum(r["weight"] for r in scored)
    if not total_weight:
        return 0.0, 0.0, []
    req_score = 100 * sum(r["weight"] * assessments[r["id"]]["score"] for r in scored) / (
        10 * total_weight
    )
    parts = [(W_REQUIREMENTS, req_score)]
    if overall is not None:
        parts.append((W_OVERALL, overall))
    if coverage is not None:
        parts.append((W_KEYWORDS, 100 * coverage))
    blended = sum(w * v for w, v in parts) / sum(w for w, _ in parts)
    missed = [
        r["id"] for r in requirements
        if r["must_have"] and r["id"] in assessments
        and assessments[r["id"]]["score"] < MUST_HAVE_MIN_SCORE
    ]
    multiplier = max(MIN_MULTIPLIER, MISS_MULTIPLIER ** len(missed))
    return round(blended * multiplier, 1), round(req_score, 1), missed
