"""
Agent mode: lets the LLM decide which searches to run before answering.

The normal pipeline does exactly one search and one answer. That is enough for
most questions, but not for ones that need facts from several places
("compare nPMS and nVMS"), a date calculation ("years from graduation to his
current role"), or a better phrasing than the user's. Here the model gets
tools and a small step budget:

  search_documents(query)            -> top chunks for a focused query
  list_documents()                   -> what is uploaded
  calculate_duration(start, end)     -> exact years/months between two dates

Every step is one LLM call, and the step budget is a hard cap: on the last
step tools are disabled, so the model must answer with what it has. Total LLM
calls per question <= AGENT_MAX_STEPS.

Only GeminiService implements generate_with_tools; with other providers the
caller falls back to the normal pipeline.
"""

import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import date

from fastapi import HTTPException, status

from app.core.config import get_settings
from app.core.logging import logger
from app.models.chat import SourceReference
from app.services.rerank_service import rerank

_MAX_SOURCES = 8
_MAX_TOOL_CALLS_PER_STEP = 4

TOOLS = [
    {
        "name": "search_documents",
        "description": (
            "Semantic search over the user's uploaded documents. Returns the most relevant "
            "passages. Use short, focused queries; call it several times for multi-part questions."
        ),
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "What to look for"}},
            "required": ["query"],
        },
    },
    {
        "name": "list_documents",
        "description": "List the uploaded documents (file names and sizes in chunks).",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "calculate_duration",
        "description": (
            "Exact time between two dates, e.g. for years of experience. Dates may be like "
            "'June 2022', 'Jun 2022', '2022-06', '2021'. end defaults to today ('present')."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "start": {"type": "string"},
                "end": {"type": "string", "description": "Defaults to 'present'"},
            },
            "required": ["start"],
        },
    },
]

AGENT_INSTRUCTIONS = (
    "\n\nYou can use tools. The 'Initial search results' in the user's message come from one "
    "automatic search and may be incomplete. If the question has several parts, compares "
    "things, needs facts from different sections or documents, or the initial results don't "
    "contain the answer, call search_documents with focused queries (one per part) before "
    "answering. Use calculate_duration instead of doing date arithmetic yourself. As soon as "
    "you have enough information, answer. Base the answer only on search results; never "
    "mention tools or 'search results' in the answer itself."
)

# Phrases that usually mean the answer needs more than one retrieved passage.
_MULTI_PART_RE = re.compile(
    r"\b(compare|comparison|difference|differences|differ|versus|vs\.?|both|each of|all (?:of )?(?:his |the )?"
    r"(?:projects|companies|roles|documents)|summari[sz]e (?:all|every)|how many years (?:between|from|since))\b",
    re.IGNORECASE,
)

_MONTHS = {
    m: i
    for i, names in enumerate(
        [
            ("jan", "january"), ("feb", "february"), ("mar", "march"), ("apr", "april"),
            ("may",), ("jun", "june"), ("jul", "july"), ("aug", "august"),
            ("sep", "sept", "september"), ("oct", "october"), ("nov", "november"), ("dec", "december"),
        ],
        start=1,
    )
    for m in names
}


def looks_multi_part(question: str) -> bool:
    return bool(_MULTI_PART_RE.search(question)) or question.count("?") >= 2


def parse_month(value: str, today: date | None = None) -> tuple[int, int] | None:
    """Parse 'June 2022' / 'Jun 2022' / '2022-06' / '06/2022' / '2022' / 'present' -> (year, month)."""
    today = today or date.today()
    text = value.strip().lower()
    if text in {"present", "now", "today", "current", ""}:
        return today.year, today.month
    if match := re.fullmatch(r"(\d{4})[-/.](\d{1,2})(?:[-/.]\d{1,2})?", text):
        return int(match.group(1)), int(match.group(2))
    if match := re.fullmatch(r"(\d{1,2})[-/.](\d{4})", text):
        return int(match.group(2)), int(match.group(1))
    if match := re.fullmatch(r"([a-z]+)\.?,?\s+(\d{4})", text):
        month = _MONTHS.get(match.group(1))
        return (int(match.group(2)), month) if month else None
    if match := re.fullmatch(r"(\d{4})", text):
        return int(match.group(1)), 1
    return None


def calculate_duration(start: str, end: str = "present", today: date | None = None) -> dict:
    begin, finish = parse_month(start, today), parse_month(end or "present", today)
    if not begin or not finish:
        return {"error": f"Could not understand the dates {start!r} / {end!r}. Use e.g. 'June 2022'."}
    months = (finish[0] - begin[0]) * 12 + (finish[1] - begin[1])
    if months < 0:
        return {"error": "The end date is before the start date."}
    years, rest = divmod(months, 12)
    return {
        "start": f"{begin[0]}-{begin[1]:02d}",
        "end": f"{finish[0]}-{finish[1]:02d}",
        "total_months": months,
        "years": years,
        "months": rest,
        "as_text": f"{years} years {rest} months",
    }


@dataclass
class AgentResult:
    answer: str
    sources: list[SourceReference]
    steps: list[str] = field(default_factory=list)  # human-readable tool calls, for logs/UI
    llm_calls: int = 0


class AgentService:
    """Runs the tool-using loop. Needs an LLM with generate_with_tools (Gemini)."""

    def __init__(self, llm_service, embedding_service, qdrant_service, memory_service) -> None:
        self.settings = get_settings()
        self.llm = llm_service
        self.embedding = embedding_service
        self.qdrant = qdrant_service
        self.memory = memory_service

    @property
    def available(self) -> bool:
        return hasattr(self.llm, "generate_with_tools")

    async def run(self, **kwargs) -> AgentResult:
        result: AgentResult | None = None
        async for event in self.run_with_events(**kwargs):
            if isinstance(event, AgentResult):
                result = event
        assert result is not None
        return result

    async def run_with_events(
        self,
        question: str,
        system_prompt: str,
        initial_context: str,
        initial_sources: list[SourceReference],
        history: list[tuple[str, str]],
    ) -> AsyncIterator["str | AgentResult"]:
        """Yield a status string before each tool call, and finally the AgentResult."""
        max_steps = max(1, self.settings.agent_max_steps)
        sources: dict[str, SourceReference] = {s.chunk_id: s for s in initial_sources}
        steps: list[str] = []

        contents: list[dict] = []
        for past_question, past_answer in history:
            contents.append({"role": "user", "parts": [{"text": past_question}]})
            contents.append({"role": "model", "parts": [{"text": past_answer}]})
        contents.append({
            "role": "user",
            "parts": [{
                "text": f"Question: {question}\n\nInitial search results:\n"
                f"{initial_context or '(nothing relevant found)'}"
            }],
        })

        for step in range(1, max_steps + 1):
            last_step = step == max_steps
            content = await self.llm.generate_with_tools(
                system_prompt=system_prompt + AGENT_INSTRUCTIONS,
                contents=contents,
                tools=TOOLS,
                allow_tools=not last_step,
            )
            calls = [p["functionCall"] for p in content["parts"] if "functionCall" in p]
            text = "".join(p.get("text", "") for p in content["parts"] if not p.get("thought")).strip()

            if not calls:
                if text or last_step:
                    yield AgentResult(
                        answer=text or "I couldn't find enough information in the documents to answer that.",
                        sources=self._rank_sources(sources),
                        steps=steps,
                        llm_calls=step,
                    )
                    return
                # Empty reply without tool calls: nudge once, the next step will answer.
                contents.append(content)
                contents.append({"role": "user", "parts": [{"text": "Please answer the question now."}]})
                continue

            contents.append(content)  # verbatim, incl. thoughtSignature
            responses = []
            for call in calls[:_MAX_TOOL_CALLS_PER_STEP]:
                label = self._describe(call)
                steps.append(label)
                yield label
                result = await self._execute(call, sources)
                response = {"name": call["name"], "response": result}
                if call.get("id"):
                    response["id"] = call["id"]
                responses.append({"functionResponse": response})
            contents.append({"role": "user", "parts": responses})

        raise HTTPException(  # unreachable: the last step has tools disabled
            status_code=status.HTTP_502_BAD_GATEWAY, detail="The agent did not produce an answer."
        )

    # ------------------------------------------------------------------ #
    # Tools
    # ------------------------------------------------------------------ #

    @staticmethod
    def _describe(call: dict) -> str:
        args = call.get("args") or {}
        if call["name"] == "search_documents":
            return f"Searching: {args.get('query', '')}"
        if call["name"] == "calculate_duration":
            return f"Calculating time from {args.get('start', '?')} to {args.get('end') or 'present'}"
        if call["name"] == "list_documents":
            return "Listing documents"
        return f"Running {call['name']}"

    async def _execute(self, call: dict, sources: dict[str, SourceReference]) -> dict:
        name, args = call.get("name"), call.get("args") or {}
        try:
            if name == "search_documents":
                return await self._search(str(args.get("query", "")), sources)
            if name == "list_documents":
                documents = await self.memory.list_documents()
                return {"documents": [{"filename": d.filename, "chunks": d.chunk_count} for d in documents]}
            if name == "calculate_duration":
                return calculate_duration(str(args.get("start", "")), str(args.get("end") or "present"))
            return {"error": f"Unknown tool {name!r}"}
        except HTTPException as exc:
            logger.warning(f"Agent tool {name} failed: {exc.detail}")
            return {"error": str(exc.detail)}

    async def _search(self, query: str, sources: dict[str, SourceReference]) -> dict:
        if not query.strip():
            return {"error": "Empty query"}
        k = self.settings.rag_top_k
        vector = await self.embedding.embed_text(query)
        candidates = await self.qdrant.search(
            query_vector=vector, top_k=min(k * 4, 40), score_threshold=self.settings.effective_score_threshold
        )
        results = rerank(query, candidates, top_k=k, mode=self.settings.rag_rerank_mode)
        passages = []
        for r in results:
            metadata = r.get("metadata") or {}
            source = SourceReference(
                document_id=r["document_id"],
                filename=r["filename"],
                chunk_id=r["chunk_id"],
                chunk_text=r["chunk_text"],
                score=r["score"],
                page_start=metadata.get("page_start"),
                page_end=metadata.get("page_end"),
                section=metadata.get("section"),
            )
            sources.setdefault(source.chunk_id, source)
            passages.append({
                "document": source.filename,
                "page": source.page_start,
                "section": source.section,
                "text": source.chunk_text,
            })
        return {"results": passages} if passages else {"results": [], "note": "No relevant passages found."}

    @staticmethod
    def _rank_sources(sources: dict[str, SourceReference]) -> list[SourceReference]:
        seen_text: set[str] = set()
        ranked: list[SourceReference] = []
        for source in sorted(sources.values(), key=lambda s: s.score, reverse=True):
            if source.chunk_text in seen_text:
                continue
            seen_text.add(source.chunk_text)
            ranked.append(source)
        return ranked[:_MAX_SOURCES]
