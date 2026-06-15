"""Shared agent machinery: LLM dispatch, JSON discipline, and timeouts.

Every specialist agent subclasses `BaseAgent`, sets a `name`, a retrieval `query`,
and a `system_prompt`, and inherits one `run()` that: formats its retrieved chunks
into a prompt, calls the LLM (Groq `llama-3.3-70b-versatile` → Gemini 2.5 Pro
fallback), parses strict JSON (one retry with a reminder if the model rambles), and
enforces a hard 30-second ceiling. Agents never throw across their boundary — every
outcome is a `Result[dict, AgentError]`.
"""

from __future__ import annotations

import json
import os
import re
from abc import ABC
from dataclasses import dataclass
from functools import lru_cache

import tiktoken
from rich.console import Console

from ingestion.types import Err, Ok, RankedChunk, Result

GROQ_MODEL = "llama-3.3-70b-versatile"
GEMINI_MODEL = "gemini-2.5-pro"
AGENT_TIMEOUT = 30.0  # seconds, per agent (patchable in tests)
MAX_CONTEXT_TOKENS = 9000  # hard cap on the chunks context sent to the LLM
CONTEXT_ENCODING = "cl100k_base"

_JSON_REMINDER = (
    "Your previous response was not valid JSON. Return ONLY a single valid JSON "
    "object matching the schema. No preamble, no explanation, no markdown fences."
)

console = Console(stderr=True)

_FENCE_RE = re.compile(r"^```[a-zA-Z]*\n|\n```$")


@lru_cache(maxsize=1)
def _encoder():  # noqa: ANN202
    return tiktoken.get_encoding(CONTEXT_ENCODING)


def _truncate_context(text: str, agent_name: str) -> str:
    """Cap the chunks context at MAX_CONTEXT_TOKENS tokens; the system prompt is
    never touched. Logs via rich when truncation actually happens."""
    tokens = _encoder().encode(text)
    if len(tokens) <= MAX_CONTEXT_TOKENS:
        return text
    truncated = _encoder().decode(tokens[:MAX_CONTEXT_TOKENS])
    console.log(
        f"[yellow]Truncated context from {len(tokens)} to "
        f"{MAX_CONTEXT_TOKENS} tokens for {agent_name}[/]"
    )
    return truncated


@dataclass(frozen=True, slots=True)
class AgentError:
    agent: str
    reason: str


def _extract_json(text: str) -> dict | None:
    """Parse a JSON object out of an LLM response, tolerating fences and chatter."""
    if not text:
        return None
    candidate = _FENCE_RE.sub("", text.strip()).strip()
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else None
    except (json.JSONDecodeError, ValueError):
        pass
    # Last resort: grab the outermost {...} span.
    start, end = candidate.find("{"), candidate.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(candidate[start : end + 1])
            return parsed if isinstance(parsed, dict) else None
        except (json.JSONDecodeError, ValueError):
            return None
    return None


async def _call_groq(
    system_prompt: str, user_message: str, api_key: str, *, json_mode: bool = True
) -> str:
    from groq import AsyncGroq

    client = AsyncGroq(api_key=api_key)
    # json_object mode requires the word "json" in the messages, so it's only valid
    # for the agents/synthesis (whose schema prompts contain it) — not prose Q&A.
    response_format = {"type": "json_object"} if json_mode else None
    resp = await client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,
        response_format=response_format,
    )
    return resp.choices[0].message.content or ""


async def _call_gemini(
    system_prompt: str, user_message: str, api_key: str, *, json_mode: bool = True
) -> str:
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(GEMINI_MODEL, system_instruction=system_prompt)
    generation_config = {"response_mime_type": "application/json"} if json_mode else None
    resp = await model.generate_content_async(
        user_message, generation_config=generation_config
    )
    return resp.text or ""


async def _complete_text(
    agent_name: str, system_prompt: str, user_message: str, *, json_mode: bool = True
) -> Result[str, AgentError]:
    """Raw text completion with Groq → Gemini fallback. No JSON parsing here.

    `json_mode` constrains the model to a JSON object (used by agents/synthesis).
    Set it False for free-form prose, e.g. the Q&A endpoint.
    """
    groq_key = os.environ.get("GROQ_API_KEY")
    if groq_key:
        try:
            return Ok(await _call_groq(system_prompt, user_message, groq_key, json_mode=json_mode))
        except Exception as exc:  # quota, auth, network — fall through to Gemini
            console.log(
                f"[yellow]Groq failed for {agent_name} ({exc}); falling back to Gemini[/]"
            )

    gemini_key = os.environ.get("GEMINI_API_KEY")
    if gemini_key:
        try:
            return Ok(await _call_gemini(system_prompt, user_message, gemini_key, json_mode=json_mode))
        except Exception as exc:
            return Err(AgentError(agent=agent_name, reason=f"Gemini call failed: {exc}"))

    return Err(
        AgentError(
            agent=agent_name,
            reason="no LLM backend configured (set GROQ_API_KEY or GEMINI_API_KEY)",
        )
    )


class BaseAgent(ABC):  # noqa: B024 — abstract via its class-attribute contract, not methods
    # Subclasses MUST override these three; the base is never instantiated directly.
    name: str = "base"
    query: str = ""
    system_prompt: str = ""

    def _format_context(self, chunks: list[RankedChunk]) -> str:
        """Render retrieved chunks as the LLM's grounding context, importance included."""
        if not chunks:
            return "No code chunks were retrieved for this query."
        parts: list[str] = []
        for rc in chunks:
            c = rc.chunk
            parts.append(
                f"### {c.file_path} :: {c.unit_type} {c.name} "
                f"(lines {c.start_line}-{c.end_line}, importance={rc.importance_score:.4f})\n"
                f"{c.context_header}\n{c.content}"
            )
        return "\n\n".join(parts)

    def _postprocess(self, data: dict, chunks: list[RankedChunk]) -> dict:
        """Hook for agents to inject grounded data (overridden by dep_agent)."""
        return data

    async def _call_llm(self, user_message: str) -> Result[dict, AgentError]:
        first = await _complete_text(self.name, self.system_prompt, user_message)
        if not first.is_ok():
            return first
        parsed = _extract_json(first.unwrap())
        if parsed is not None:
            return Ok(parsed)

        # One retry with an explicit reminder before giving up.
        retry_message = f"{user_message}\n\n{_JSON_REMINDER}"
        second = await _complete_text(self.name, self.system_prompt, retry_message)
        if not second.is_ok():
            return second
        parsed = _extract_json(second.unwrap())
        if parsed is not None:
            return Ok(parsed)

        return Err(
            AgentError(agent=self.name, reason="LLM did not return valid JSON after one retry")
        )

    async def run(self, retriever_results: list[RankedChunk]) -> Result[dict, AgentError]:
        import asyncio

        user_message = _truncate_context(
            self._format_context(retriever_results), self.name
        )
        try:
            result = await asyncio.wait_for(
                self._call_llm(user_message), timeout=AGENT_TIMEOUT
            )
        except TimeoutError:
            return Err(
                AgentError(agent=self.name, reason=f"LLM call timed out after {AGENT_TIMEOUT}s")
            )
        if not result.is_ok():
            return result
        return Ok(self._postprocess(result.unwrap(), retriever_results))
