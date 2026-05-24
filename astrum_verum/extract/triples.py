"""
Извлечение структурированных триплетов (subject, relation, object) из текста.

Портировано с паттерна mavka-bot/deep_parse.py (fallback-цепочка провайдеров,
загрузка ключей из env/.env, жадный JSON-парс, Jaccard-дедуп), но вместо
ПЛОСКИХ фактов извлекает РОЛЕВУЮ структуру — то, что нужно VSA-слою (Phase 2).

Провайдеры (по убыванию приоритета): DeepSeek → xAI → Groq. Ключи берутся из
окружения, иначе из astrum-verum/.env (строки `export KEY=...` или `KEY=...`).
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from pathlib import Path

_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"

_PROVIDERS = [
    ("deepseek", "DEEPSEEK_API_KEY", "https://api.deepseek.com/chat/completions", "deepseek-chat"),
    ("xai", "XAI_API_KEY", "https://api.x.ai/v1/chat/completions", "grok-3-mini"),
    ("groq", "GROQ_API_KEY", "https://api.groq.com/openai/v1/chat/completions", "llama-3.3-70b-versatile"),
]


def _env(key: str) -> str:
    val = os.environ.get(key, "")
    if val:
        return val
    if _ENV_FILE.exists():
        for line in _ENV_FILE.read_text(errors="replace").splitlines():
            line = line.strip()
            if line.startswith("export "):
                line = line[7:].strip()
            if line.startswith(f"{key}="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def _call(url: str, key: str, model: str, prompt: str, timeout: int = 60) -> str | None:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 1500,
    }).encode()
    req = urllib.request.Request(
        url, data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
            "User-Agent": "python-urllib/3.11",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"].strip()


def _call_llm(prompt: str) -> tuple[str | None, str | None]:
    """Возвращает (ответ, имя_провайдера) или (None, None)."""
    for name, env_key, url, model in _PROVIDERS:
        key = _env(env_key)
        if not key:
            continue
        try:
            out = _call(url, key, model, prompt)
            if out:
                return out, name
        except Exception as e:  # noqa: BLE001
            print(f"  [{name}] error: {e}")
    return None, None


_PROMPT = """\
You extract STRUCTURED RELATIONAL FACTS from text as (subject, relation, object) triples.

Rules:
- subject / object: a SHORT canonical noun phrase (entity, person, place, thing, concept).
- relation: a SHORT lemma/verb phrase describing how subject relates to object
  (e.g. "lives in", "founded", "mentors", "uses", "works as").
- Keep direction meaningful: who does what to whom. "A mentors B" ≠ "B mentors A".
- Extract only durable, factual relations. Skip small-talk, questions, emotions.
- Normalize entity names consistently (same entity → same string everywhere).

Return ONLY a JSON array, nothing else:
[
  {{"subject": "...", "relation": "...", "object": "..."}},
  ...
]
If there are no facts → return []

TEXT:
{text}
"""


def extract_triples(text: str) -> tuple[list[dict], str | None]:
    """text → список {'subject','relation','object'} + имя сработавшего провайдера.

    Дедуп — по ТОЧНОМУ упорядоченному триплету (роли важны: A→B ≠ B→A; дедуп по
    множеству слов схлопнул бы role-swap и убил бы ролевую структуру)."""
    raw, provider = _call_llm(_PROMPT.format(text=text))
    if not raw:
        return [], None
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        return [], provider
    try:
        items = json.loads(match.group())
    except Exception:  # noqa: BLE001
        return [], provider

    out: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for it in items:
        if not isinstance(it, dict):
            continue
        s, r, o = (str(it.get(k, "")).strip() for k in ("subject", "relation", "object"))
        if not (s and r and o):
            continue
        key = (s.lower(), r.lower(), o.lower())  # упорядоченный → role-swap сохраняется
        if key in seen:
            continue
        seen.add(key)
        out.append({"subject": s, "relation": r, "object": o})
    return out, provider
