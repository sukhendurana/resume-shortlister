"""Minimal Hugging Face Inference Providers chat client (stdlib only).

Talks to the OpenAI-compatible endpoint exposed by the Hugging Face router, so a
single HF access token is enough to call any hosted open-weight chat model.
Includes retry/backoff and tolerant JSON extraction for small instruction models.
"""
import hashlib
import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request

from guardrails import require_scope

ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"

# Reasoning models (e.g. Qwen3, DeepSeek-R1) may prepend <think>...</think>.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json)?", re.IGNORECASE)
# HTTP codes worth retrying: rate limit and transient server/provider errors.
_RETRYABLE = {408, 429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    """Raised when a model call fails permanently or returns unusable output."""


class CreditsExhaustedError(LLMError):
    """Raised on HTTP 402: the account's inference credits are used up (not retryable)."""


def extract_json(text):
    """Return the first valid JSON object found in `text`, or None.

    Strips <think> blocks and markdown fences, then tries to decode an object
    starting at each '{' so leading/trailing chatter from the model is ignored.
    """
    cleaned = _FENCE_RE.sub("", _THINK_RE.sub("", text or ""))
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", cleaned):
        try:
            obj, _ = decoder.raw_decode(cleaned[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


class HFChatClient:
    """Chat-completion client for Hugging Face hosted models."""

    def __init__(self, token, timeout=120, max_retries=4, api_url=ROUTER_URL, cache_path=None):
        """Store credentials, network policy and the optional response cache.

        Args:
            token: Hugging Face access token (never hard-coded; supplied at runtime).
            timeout: per-request timeout in seconds.
            max_retries: attempts for transient failures before giving up.
            api_url: OpenAI-compatible chat endpoint (default: Hugging Face router;
                any compatible server, e.g. a local Ollama, also works).
            cache_path: JSON-lines file storing parsed replies keyed by a hash of the
                prompt, so re-runs and runs resumed after an error cost no credits.
        """
        if not token:
            raise LLMError("Missing Hugging Face token (use --hf-token or HF_TOKEN).")
        self._token = token
        self._timeout = timeout
        self._max_retries = max_retries
        self._api_url = api_url
        self._cache, self._cache_path = {}, cache_path
        self._lock = threading.Lock()
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as handle:
                for line in handle:
                    try:
                        entry = json.loads(line)
                        self._cache[entry["k"]] = entry["v"]
                    except (json.JSONDecodeError, KeyError):
                        continue  # ignore a truncated last line

    def _post(self, payload):
        """POST one request, retrying transient failures with exponential backoff."""
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }
        last_error = "unknown error"
        for attempt in range(self._max_retries):
            request = urllib.request.Request(self._api_url, data=body, headers=headers)
            delay = 2 ** attempt + random.random()
            try:
                with urllib.request.urlopen(request, timeout=self._timeout) as resp:
                    return json.load(resp)
            except urllib.error.HTTPError as err:
                detail = err.read().decode("utf-8", "replace")[:300]
                last_error = f"HTTP {err.code}: {detail}"
                if err.code == 402:
                    raise CreditsExhaustedError(
                        "Hugging Face inference credits are exhausted (HTTP 402). Add "
                        "pre-paid credits / PRO, wait for the monthly reset, or point "
                        "--api-url at a local OpenAI-compatible server. Finished work is "
                        "cached, so re-running resumes where it stopped.") from err
                if err.code not in _RETRYABLE:
                    hint = (" -> run `python main.py --list-models` to see models available "
                            "to your token" if "model_not_supported" in detail else "")
                    raise LLMError(f"{payload['model']}: {last_error}{hint}") from err
                retry_after = err.headers.get("Retry-After", "")
                if retry_after.isdigit():
                    delay = max(delay, int(retry_after))
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as err:
                last_error = str(err)
            if attempt < self._max_retries - 1:
                time.sleep(delay)
        raise LLMError(f"{payload['model']}: failed after retries ({last_error})")

    def list_models(self, needle=""):
        """Return sorted ids of chat models served to this token (optionally filtered)."""
        request = urllib.request.Request(
            self._api_url.replace("/chat/completions", "/models"), headers={"Authorization": f"Bearer {self._token}"})
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as resp:
                data = json.load(resp).get("data", [])
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as err:
            raise LLMError(f"Could not list models: {err}") from err
        return sorted(m["id"] for m in data if needle.lower() in m["id"].lower())

    def chat(self, model, system, user, max_tokens=1500, temperature=0.0):
        """Return the assistant text for a system+user prompt (think-blocks removed).

        Refuses any system prompt that is not the CV-review scope prompt.
        """
        require_scope(system)
        data = self._post({
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        })
        try:
            content = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as err:
            raise LLMError(f"{model}: malformed response {str(data)[:200]}") from err
        return _THINK_RE.sub("", content).strip()

    def chat_json(self, model, system, user, max_tokens=1500):
        """Call the model and parse a JSON object (cached), re-asking once on bad JSON."""
        key = hashlib.sha256(
            json.dumps([model, system, user, max_tokens]).encode("utf-8")).hexdigest()
        with self._lock:
            if key in self._cache:
                return self._cache[key]
        reply = self.chat(model, system, user, max_tokens)
        parsed = extract_json(reply)
        if parsed is None:
            reply = self.chat(
                model, system,
                user + "\n\nYour previous reply was not valid JSON. "
                       "Reply with ONE valid JSON object and nothing else.",
                max_tokens,
            )
            parsed = extract_json(reply)
        if parsed is None:
            raise LLMError(f"{model}: could not obtain valid JSON")
        with self._lock:
            self._cache[key] = parsed
            if self._cache_path:
                with open(self._cache_path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"k": key, "v": parsed}, ensure_ascii=False) + "\n")
        return parsed
