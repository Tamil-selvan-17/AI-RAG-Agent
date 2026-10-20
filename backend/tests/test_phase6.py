"""Tests for Phase 6: section headings, contextual headers, and agent mode."""

import sys
from datetime import date
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.models.chat import SourceReference  # noqa: E402
from app.services.agent_service import (  # noqa: E402
    AgentService,
    calculate_duration,
    looks_multi_part,
    parse_month,
)
from app.services.chunking_service import ChunkingService, _detect_heading, contextual_text  # noqa: E402
from app.services.memory_store_service import MemoryStoreService  # noqa: E402
from app.services.rag_service import RagService  # noqa: E402

# ---- Section headings ---------------------------------------------------------- #


@pytest.mark.parametrize(
    "line, previous, expected",
    [
        ("KEY PROJECTS", "", (1, "KEY PROJECTS")),
        ("## 4. Qdrant setup", "---", (1, "4. Qdrant setup")),
        ("### The Q&A cache", "", (2, "The Q&A cache")),
        ("2. nVMS - Vendor Management System", "", (2, "2. nVMS - Vendor Management System")),
        ("S k i l l s", "", (1, "SKILLS")),
        ("XML", "Technologies: Angular, SQL Server,", None),  # acronym continuing a wrapped line
        ("AWS", "", None),  # short acronym alone
        ("3. Register it in the extractors dict", "", None),  # numbered step, not a title
        ("Built REST APIs with FastAPI.", "", None),
        ("[Page 2]", "", None),
    ],
)
def test_detect_heading(line, previous, expected):
    assert _detect_heading(line, previous) == expected


def test_sections_follow_document_structure_and_ignore_code_blocks():
    text = (
        "PROFESSIONAL EXPERIENCE\n" + "Built enterprise apps with Angular and .NET. " * 12 + "\n"
        "KEY PROJECTS\n1. nPMS - Email Monitoring System\n" + "Tracks project emails for managers. " * 12 + "\n"
        "```bash\n# not a heading\n```\n" + "More nPMS detail about dashboards. " * 12
    )
    chunks = ChunkingService(chunk_size_tokens=60, chunk_overlap_tokens=10).chunk_document(text, "d", "cv.txt")
    sections = [c.metadata.get("section") for c in chunks]
    assert sections[0] == "PROFESSIONAL EXPERIENCE"
    assert "KEY PROJECTS > 1. nPMS - Email Monitoring System" in sections
    assert not any("not a heading" in (s or "") for s in sections)


def test_contextual_text_header():
    assert contextual_text("cv.pdf", "KEY PROJECTS", "text") == "Document: cv.pdf\nSection: KEY PROJECTS\n\ntext"
    assert contextual_text("cv.pdf", None, "text") == "Document: cv.pdf\n\ntext"


# ---- Duration tool / multi-part detection --------------------------------------- #

TODAY = date(2026, 10, 20)


@pytest.mark.parametrize(
    "value, expected",
    [("June 2022", (2022, 6)), ("Jun 2022", (2022, 6)), ("Sept 2024", (2024, 9)), ("2022-06", (2022, 6)),
     ("06/2022", (2022, 6)), ("2021", (2021, 1)), ("Present", (2026, 10)), ("someday", None)],
)
def test_parse_month(value, expected):
    assert parse_month(value, TODAY) == expected


def test_calculate_duration():
    result = calculate_duration("Jun 2022", "present", today=TODAY)
    assert (result["years"], result["months"], result["total_months"]) == (4, 4, 52)
    assert "error" in calculate_duration("June 2024", "June 2022", today=TODAY)
    assert "error" in calculate_duration("whenever", today=TODAY)


@pytest.mark.parametrize(
    "question, expected",
    [
        ("Compare his nPMS and nVMS projects", True),
        ("Which projects use both MongoDB and .NET 10?", True),
        ("Summarize all of his projects", True),
        ("Where did he study? What was his CGPA?", True),
        ("What is his CGPA?", False),
        ("Which company does he work for?", False),
    ],
)
def test_looks_multi_part(question, expected):
    assert looks_multi_part(question) is expected


# ---- Agent loop ----------------------------------------------------------------- #


def _hit(text, score=0.7, chunk_id=None):
    return {
        "document_id": "doc-1", "filename": "cv.pdf", "chunk_id": chunk_id or f"c-{abs(hash(text))}",
        "chunk_text": text, "score": score, "metadata": {"page_start": 2, "section": "KEY PROJECTS"},
    }


def _call(name, args, call_id="call_1"):
    return {"role": "model", "parts": [{"functionCall": {"name": name, "args": args, "id": call_id},
                                        "thoughtSignature": "sig-abc"}]}


def _text(text):
    return {"role": "model", "parts": [{"text": text}]}


def _agent(llm, search_results=None):
    embedding = AsyncMock()
    embedding.embed_text.return_value = [0.1] * 4
    qdrant = AsyncMock()
    qdrant.search.return_value = search_results or []
    return AgentService(llm, embedding, qdrant, MemoryStoreService())


@pytest.mark.asyncio
async def test_agent_searches_then_answers_and_echoes_call_verbatim():
    llm = AsyncMock()
    llm.generate_with_tools.side_effect = [
        _call("search_documents", {"query": "nVMS technologies"}),
        _text("nVMS uses Angular and .NET 6."),
    ]
    agent = _agent(llm, [_hit("nVMS: Angular, Tailwind CSS, ASP.NET Core (.NET 6)")])

    result = await agent.run(question="nVMS stack?", system_prompt="sys", initial_context="",
                             initial_sources=[], history=[])

    assert result.answer == "nVMS uses Angular and .NET 6."
    assert result.llm_calls == 2
    assert result.steps == ["Searching: nVMS technologies"]
    assert result.sources[0].section == "KEY PROJECTS"
    second_contents = llm.generate_with_tools.call_args_list[1].kwargs["contents"]
    assert second_contents[-2]["parts"][0]["thoughtSignature"] == "sig-abc"  # echoed unchanged
    response = second_contents[-1]["parts"][0]["functionResponse"]
    assert response["id"] == "call_1" and response["name"] == "search_documents"


@pytest.mark.asyncio
async def test_agent_last_step_has_tools_disabled(monkeypatch):
    llm = AsyncMock()
    llm.generate_with_tools.side_effect = [
        _call("search_documents", {"query": "a"}, "c1"),
        _call("search_documents", {"query": "b"}, "c2"),
        _text("Final answer."),
    ]
    agent = _agent(llm, [_hit("some text")])
    monkeypatch.setattr(agent.settings, "agent_max_steps", 3)

    result = await agent.run(question="q", system_prompt="s", initial_context="", initial_sources=[], history=[])

    allow = [c.kwargs["allow_tools"] for c in llm.generate_with_tools.call_args_list]
    assert allow == [True, True, False]
    assert result.answer == "Final answer." and result.llm_calls == 3


@pytest.mark.asyncio
async def test_agent_duration_tool_uses_exact_math():
    llm = AsyncMock()
    llm.generate_with_tools.side_effect = [
        _call("calculate_duration", {"start": "Jun 2022", "end": "Sep 2024"}),
        _text("2 years 3 months."),
    ]
    agent = _agent(llm)
    await agent.run(question="q", system_prompt="s", initial_context="", initial_sources=[], history=[])
    sent = llm.generate_with_tools.call_args_list[1].kwargs["contents"][-1]["parts"][0]
    assert sent["functionResponse"]["response"]["as_text"] == "2 years 3 months"


# ---- RagService trigger rules --------------------------------------------------- #


@pytest.fixture
def rag(tmp_path):
    llm = AsyncMock()
    llm.generate_chat_response.return_value = "Pipeline answer."
    llm.generate_with_tools.return_value = _text("Agent answer.")
    embedding = AsyncMock()
    embedding.embed_text.return_value = [0.1] * 4
    qdrant = AsyncMock()
    service = RagService(llm_service=llm, embedding_service=embedding, qdrant_service=qdrant,
                         redis_service=MemoryStoreService())
    service.settings.upload_dir = str(tmp_path)
    return service


@pytest.fixture
def agent_mode(rag, monkeypatch):
    def set_mode(mode, score_cutoff=0.6):
        monkeypatch.setattr(rag.settings, "agent_mode", mode)
        monkeypatch.setattr(rag.settings, "agent_fallback_score", score_cutoff)
    return set_mode


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode, question, score, expect_agent",
    [
        ("fallback", "What is his CGPA?", 0.72, False),  # strong search -> pipeline
        ("fallback", "What is his CGPA?", 0.55, True),   # weak search -> agent
        ("fallback", "Compare nPMS and nVMS", 0.80, True),  # multi-part -> agent
        ("off", "Compare nPMS and nVMS", 0.55, False),
        ("always", "What is his CGPA?", 0.90, True),
    ],
)
async def test_agent_trigger_rules(rag, agent_mode, mode, question, score, expect_agent):
    agent_mode(mode)
    rag.qdrant_service.search.return_value = [_hit("CGPA: 7.45 / 10", score=score)]
    _, answer, _ = await rag.answer_question(question, None)
    assert answer == ("Agent answer." if expect_agent else "Pipeline answer.")


@pytest.mark.asyncio
async def test_no_match_costs_no_llm_call_in_fallback(rag, agent_mode):
    agent_mode("fallback")
    rag.qdrant_service.search.return_value = []
    _, answer, _ = await rag.answer_question("Compare apples and oranges", None)
    assert "couldn't find" in answer
    rag.llm_service.generate_with_tools.assert_not_awaited()
    rag.llm_service.generate_chat_response.assert_not_awaited()


@pytest.mark.asyncio
async def test_agent_failure_falls_back_to_pipeline(rag, agent_mode):
    agent_mode("always")
    rag.qdrant_service.search.return_value = [_hit("CGPA: 7.45")]
    rag.llm_service.generate_with_tools.side_effect = HTTPException(status_code=502, detail="bad tool reply")
    _, answer, _ = await rag.answer_question("What is his CGPA?", None)
    assert answer == "Pipeline answer."


@pytest.mark.asyncio
async def test_agent_quota_error_is_not_retried_with_pipeline(rag, agent_mode):
    agent_mode("always")
    rag.qdrant_service.search.return_value = [_hit("CGPA: 7.45")]
    rag.llm_service.generate_with_tools.side_effect = HTTPException(status_code=429, detail="daily limit")
    with pytest.raises(HTTPException) as exc_info:
        await rag.answer_question("What is his CGPA?", None)
    assert exc_info.value.status_code == 429
    rag.llm_service.generate_chat_response.assert_not_awaited()


@pytest.mark.asyncio
async def test_stream_reports_agent_status_and_steps(rag, agent_mode):
    agent_mode("fallback")
    rag.qdrant_service.search.return_value = [_hit("nPMS uses MongoDB", score=0.8)]
    rag.llm_service.generate_with_tools.side_effect = [
        _call("search_documents", {"query": "nVMS database"}),
        _text("nPMS uses MongoDB; nVMS uses SQL Server."),
    ]
    events = [e async for e in rag.stream_answer("Compare the databases of nPMS and nVMS", None)]

    assert {"type": "status", "content": "Searching: nVMS database"} in events
    assert events[-1]["type"] == "done"
    assert events[-1]["agent_steps"] == ["Searching: nVMS database"]
    tokens = "".join(e["content"] for e in events if e["type"] == "token")
    assert tokens == "nPMS uses MongoDB; nVMS uses SQL Server."


@pytest.mark.asyncio
async def test_agent_unavailable_for_providers_without_tools(rag, agent_mode):
    agent_mode("always")

    class NoTools:  # e.g. OllamaService: no generate_with_tools
        async def generate_chat_response(self, **kwargs):
            return "Pipeline answer."

    rag.llm_service = NoTools()
    rag.agent_service.llm = rag.llm_service
    rag.qdrant_service.search.return_value = [_hit("CGPA: 7.45")]
    _, answer, _ = await rag.answer_question("What is his CGPA?", None)
    assert answer == "Pipeline answer."


def test_source_reference_section_field():
    assert SourceReference(document_id="d", filename="f", chunk_id="c", chunk_text="t", score=0.5).section is None
