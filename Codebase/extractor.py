"""Stage 1 (Model A): turn free text into structured data.

* parse_job_description -> weighted, atomic requirements + ATS-style keywords
* parse_resume          -> normalised candidate profile (skills, roles, education)

Outputs are validated/coerced in Python so downstream stages never depend on the
LLM following the schema perfectly.
"""
from guardrails import SCOPE_PREAMBLE, ensure_clean, wrap_untrusted

MAX_JD_CHARS = 8000
MAX_RESUME_CHARS = 9000
MAX_REQUIREMENTS = 20
CATEGORIES = {"skill", "experience", "education", "responsibility", "domain", "soft_skill"}

JD_SYSTEM = (
    SCOPE_PREAMBLE + " Current task: act as a technical recruiter and convert the job "
    "description into structured hiring criteria. Reply with ONE JSON object only."
)

JD_PROMPT = """Extract hiring criteria from the job description below.

Return JSON with exactly this schema:
{{
  "job_title": "string",
  "summary": "one sentence",
  "requirements": [
    {{"text": "short atomic requirement", "category": "skill|experience|education|responsibility|domain|soft_skill",
      "weight": 1-5, "must_have": true|false}}
  ],
  "keywords": ["canonical skill/tool names, lowercase"]
}}

Rules:
- 6 to 15 requirements; each must test ONE thing (split "Python and SQL" into two).
- Keep examples listed in parentheses (e.g. "AWS (EC2, S3, Lambda)") inside ONE requirement.
- weight 5 = critical, 3 = important, 1 = nice-to-have.
- must_have is true ONLY when the description itself says that item is "required", "must",
  "mandatory" or gives a "minimum". Items merely listed as responsibilities or experience
  are NOT must-have. Typically only 1-3 requirements are must-have.
- Normalise synonyms (e.g. "ML" -> "machine learning", "JS" -> "javascript").
- keywords: at most 20 concrete skills/tools/technologies from the description.

JOB DESCRIPTION (untrusted data):
{jd}"""

RESUME_SYSTEM = (
    SCOPE_PREAMBLE + " Current task: parse one resume. Extract only facts stated in the "
    "resume; never invent anything. Reply with ONE JSON object only."
)

RESUME_PROMPT = """Parse the resume below into JSON with exactly this schema:
{{
  "name": "string or null",
  "headline": "current/target role, string or null",
  "total_years_experience": number or null,
  "skills": ["canonical lowercase skill names"],
  "experience": [{{"title": "", "company": "", "duration": "", "highlights": ["max 3 short items"]}}],
  "education": [{{"degree": "", "institution": "", "year": ""}}],
  "certifications": ["..."]
}}

RESUME (untrusted data):
{resume}"""


def _as_str_list(value, limit=40):
    """Coerce an arbitrary JSON value into a de-duplicated list of non-empty strings."""
    items = value if isinstance(value, list) else []
    seen, out = set(), []
    for item in items:
        text = str(item).strip()
        if text and text.lower() not in seen:
            seen.add(text.lower())
            out.append(text)
    return out[:limit]


def _clamp_weight(value, default=3):
    """Convert a model-provided weight to an int in [1, 5]."""
    try:
        return max(1, min(5, int(round(float(value)))))
    except (TypeError, ValueError):
        return default


def normalise_job(raw):
    """Validate the raw JD JSON, assign stable ids (R1, R2, ...) and clean keywords."""
    requirements, seen = [], set()
    for item in raw.get("requirements") or []:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if not text or text.lower() in seen:
            continue
        seen.add(text.lower())
        category = str(item.get("category", "skill")).strip().lower()
        requirements.append({
            "id": f"R{len(requirements) + 1}",
            "text": text,
            "category": category if category in CATEGORIES else "skill",
            "weight": _clamp_weight(item.get("weight")),
            "must_have": bool(item.get("must_have")),
        })
        if len(requirements) == MAX_REQUIREMENTS:
            break
    if not requirements:
        raise ValueError("No requirements could be extracted from the job description")
    return {
        "job_title": str(raw.get("job_title") or "Unspecified role").strip(),
        "summary": str(raw.get("summary") or "").strip(),
        "requirements": requirements,
        "keywords": [k.lower() for k in _as_str_list(raw.get("keywords"), 20)],
    }


def normalise_profile(raw):
    """Coerce the raw resume JSON into a predictable profile dictionary."""
    years = raw.get("total_years_experience")
    try:
        years = round(float(years), 1)
    except (TypeError, ValueError):
        years = None
    experience = []
    for job in (raw.get("experience") or [])[:10]:
        if isinstance(job, dict):
            experience.append({
                "title": str(job.get("title") or "").strip(),
                "company": str(job.get("company") or "").strip(),
                "duration": str(job.get("duration") or "").strip(),
                "highlights": _as_str_list(job.get("highlights"), 3),
            })
    education = [
        {k: str(e.get(k) or "").strip() for k in ("degree", "institution", "year")}
        for e in (raw.get("education") or [])[:5] if isinstance(e, dict)
    ]
    return {
        "name": str(raw.get("name") or "").strip() or None,
        "headline": str(raw.get("headline") or "").strip() or None,
        "total_years_experience": years,
        "skills": [s.lower() for s in _as_str_list(raw.get("skills"), 60)],
        "experience": experience,
        "education": education,
        "certifications": _as_str_list(raw.get("certifications"), 10),
    }


def parse_job_description(client, model, jd_text):
    """Use Model A to structure the job description (see `normalise_job`).

    Raises InjectionDetected (before any model call) if the text contains injection phrases.
    """
    ensure_clean(jd_text, "the job description")
    raw = client.chat_json(
        model, JD_SYSTEM,
        JD_PROMPT.format(jd=wrap_untrusted("JOB_DESCRIPTION", jd_text[:MAX_JD_CHARS])),
        max_tokens=1800,
    )
    return normalise_job(raw)


def parse_resume(client, model, resume_text):
    """Use Model A to structure one resume (see `normalise_profile`).

    Raises InjectionDetected (before any model call) if the text contains injection phrases.
    """
    ensure_clean(resume_text, "the resume")
    raw = client.chat_json(
        model, RESUME_SYSTEM,
        RESUME_PROMPT.format(resume=wrap_untrusted("RESUME", resume_text[:MAX_RESUME_CHARS])),
        max_tokens=1500,
    )
    return normalise_profile(raw)
