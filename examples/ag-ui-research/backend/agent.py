import json
from dataclasses import dataclass

from ag_ui.core import EventType, StateSnapshotEvent
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.ag_ui import StateDeps

from haiku.rag.client import HaikuRAG
from haiku.rag.config import Config
from haiku.rag.graph.common import get_model


class ResearchState(BaseModel):
    """Shared state between research agent and frontend."""

    question: str = ""
    phase: str = "idle"
    status: str = ""
    plan: list[dict] = []
    current_question_index: int = 0
    insights: list[dict] = []
    document_registry: dict[str, dict] = {}
    current_document: dict | None = None
    confidence: float = 0.0
    final_report: dict | None = None


@dataclass
class ResearchDeps(StateDeps[ResearchState]):
    """Dependencies for the research agent with HaikuRAG client."""

    client: HaikuRAG


def _as_state_snapshot(ctx: RunContext[ResearchDeps]) -> StateSnapshotEvent:
    return StateSnapshotEvent(type=EventType.STATE_SNAPSHOT, snapshot=ctx.deps.state)


def create_agent(
    qa_provider: str = Config.QA_PROVIDER, qa_model: str = Config.QA_MODEL
) -> Agent[ResearchDeps, str]:
    """Create and configure the research agent.

    Args:
        qa_provider: QA provider for the agent (default: from Config.QA_PROVIDER)
        qa_model: Model name to use (default: from Config.QA_MODEL)
    """
    print(f"[AGENT SETUP] Creating agent with provider={qa_provider}, model={qa_model}")
    agent = Agent(
        model=get_model(qa_provider, qa_model),
        deps_type=ResearchDeps,
        instructions="""You are a research co-pilot powered by haiku.rag.

Your workflow MUST follow these exact steps in order:
1. Call propose_research_plan with the user's question
2. After propose_research_plan completes, IMMEDIATELY call approve_research_plan (with no arguments)
3. WAIT for approve_research_plan to return:
   - If it returns "APPROVED", proceed to step 4
   - If it returns "REVISE", ask the user "How would you like me to revise the research plan?" and wait for their response
   - Once you receive their revision feedback, revise the plan and go back to step 1
4. Once approved, process questions ONE AT A TIME:
   - Call search_question(question_id=0) and WAIT for it to complete
   - Then call extract_insights_from_results(question_id=0) and WAIT for it to complete
   - Then call search_question(question_id=1) and WAIT for it to complete
   - Then call extract_insights_from_results(question_id=1) and WAIT for it to complete
   - Then call search_question(question_id=2) and WAIT for it to complete
   - Then call extract_insights_from_results(question_id=2) and WAIT for it to complete
5. After all questions are processed, call evaluate_research_confidence
6. Ask user if they want to finalize or continue researching
7. When user approves, call synthesize_final_report

CRITICAL RULES:
- MANDATORY: Call approve_research_plan immediately after propose_research_plan - NO EXCEPTIONS
- If approve_research_plan returns "REVISE", ask the user for revision feedback naturally in chat
- Call ONE tool at a time - wait for each tool to return before calling the next
- NEVER call extract_insights_from_results until search_question has completed and returned results
- DO NOT explain what you're about to do - just call the tool
- The state updates will show the user what's happening - you don't need to narrate
- Process all 3 questions automatically without asking for approval between them

Document Viewing:
- When user asks to "show document X", call get_full_document with the document_uri

Remember: Call tools ONE AT A TIME in sequence. Each tool must complete before calling the next.
""",
    )

    @agent.tool
    async def propose_research_plan(
        ctx: RunContext[ResearchDeps], question: str
    ) -> StateSnapshotEvent:
        """Propose a research plan by decomposing the question into sub-questions."""
        ctx.deps.state.question = question
        ctx.deps.state.phase = "planning"
        ctx.deps.state.status = "Decomposing question into sub-questions..."

        decompose_prompt = f"""Break down this research question into exactly 3 specific sub-questions that would help answer it comprehensively.

Research Question: {question}

Return ONLY a JSON array of sub-questions, like: ["Question 1?", "Question 2?", "Question 3?"]"""

        response = await ctx.deps.client.ask(decompose_prompt)

        try:
            sub_questions = json.loads(response)
        except json.JSONDecodeError:
            sub_questions = [
                q.strip().lstrip("0123456789.-) ")
                for q in response.split("\n")
                if q.strip()
            ][:3]

        plan = [
            {"id": i, "question": q, "status": "pending"}
            for i, q in enumerate(sub_questions)
        ]

        ctx.deps.state.plan = plan
        ctx.deps.state.current_question_index = 0
        ctx.deps.state.status = f"Proposed plan with {len(plan)} sub-questions"

        return _as_state_snapshot(ctx)

    @agent.tool
    async def search_question(
        ctx: RunContext[ResearchDeps],
        question_id: int,
        search_type: str = "hybrid",
    ) -> StateSnapshotEvent:
        """Execute search for a specific sub-question."""
        plan = ctx.deps.state.plan
        if question_id >= len(plan):
            raise ValueError(f"Question ID {question_id} not found in plan")

        question = plan[question_id]["question"]
        ctx.deps.state.phase = "searching"
        ctx.deps.state.current_question_index = question_id
        ctx.deps.state.status = f"Searching: {question}"
        plan[question_id]["status"] = "searching"

        search_results = await ctx.deps.client.search(
            question, limit=5, search_type=search_type
        )

        expanded_map = {}
        if search_results:
            expanded_results = await ctx.deps.client.expand_context(
                search_results[:3], radius=2
            )
            expanded_map = {
                chunk.id: (chunk, score) for chunk, score in expanded_results
            }

        results = []
        for chunk, score in search_results:
            doc_uri = chunk.document_uri or "unknown"
            doc_title = chunk.document_title or chunk.document_uri or "Unknown"

            if doc_uri not in ctx.deps.state.document_registry:
                ctx.deps.state.document_registry[doc_uri] = {
                    "title": doc_title,
                    "chunks_referenced": [],
                }

            if (
                chunk.id
                not in ctx.deps.state.document_registry[doc_uri]["chunks_referenced"]
            ):
                ctx.deps.state.document_registry[doc_uri]["chunks_referenced"].append(
                    chunk.id
                )

            expanded_chunk, _ = (
                expanded_map[chunk.id] if chunk.id in expanded_map else (chunk, score)
            )
            result_data = {
                "chunk": expanded_chunk.content[:500],
                "chunk_id": chunk.id,
                "document_uri": doc_uri,
                "document_title": doc_title,
                "chunk_position": chunk.order,
                "full_chunk_content": expanded_chunk.content,
                "score": round(score, 3),
                "expanded": chunk.id in expanded_map,
            }
            results.append(result_data)

        plan[question_id]["search_results"] = {
            "type": search_type,
            "results": results,
        }
        plan[question_id]["status"] = "searched"
        ctx.deps.state.status = f"Found {len(results)} results"

        return _as_state_snapshot(ctx)

    @agent.tool
    async def extract_insights_from_results(
        ctx: RunContext[ResearchDeps],
        question_id: int,
    ) -> StateSnapshotEvent:
        """Extract key insights from search results for a specific question."""
        plan = ctx.deps.state.plan
        if question_id >= len(plan):
            raise ValueError(f"Question ID {question_id} not found in plan")

        question_item = plan[question_id]
        if "search_results" not in question_item:
            raise ValueError(
                f"No search results found for question ID {question_id}. "
                f"You must call search_question(question_id={question_id}) first."
            )

        search_results = question_item["search_results"]
        ctx.deps.state.phase = "analyzing"
        ctx.deps.state.status = "Extracting insights from results..."

        context_parts = [
            f"[Result {idx}] [Source: {r['document_title']}] {r['full_chunk_content']}"
            for idx, r in enumerate(search_results["results"])
        ]
        context = "\n\n".join(context_parts)

        class InsightResult(BaseModel):
            summary: str
            confidence: float
            result_indices: list[int]

        class InsightsList(BaseModel):
            insights: list[InsightResult]

        question_text = question_item["question"]
        extract_prompt = f"""Analyze these search results and extract 1-3 key insights that help answer the question: "{question_text}"

Search Results:
{context}

For each insight, reference which result numbers (0, 1, 2, etc.) support it."""

        insight_agent: Agent[None, InsightsList] = Agent(
            ctx.model,
            output_type=InsightsList,
        )

        result = await insight_agent.run(extract_prompt)
        raw_insights = [
            {
                "summary": insight.summary,
                "confidence": insight.confidence,
                "result_indices": insight.result_indices,
            }
            for insight in result.output.insights
        ]

        new_insights = []
        for insight in raw_insights:
            source_refs = []
            for idx in insight.get("result_indices", []):
                if 0 <= idx < len(search_results["results"]):
                    result = search_results["results"][idx]
                    source_refs.append(
                        {
                            "chunk_id": result["chunk_id"],
                            "document_uri": result["document_uri"],
                            "document_title": result["document_title"],
                            "chunk_position": result["chunk_position"],
                        }
                    )

            new_insights.append(
                {
                    "summary": insight["summary"],
                    "confidence": insight.get("confidence", 0.7),
                    "source_refs": source_refs,
                }
            )

        ctx.deps.state.insights.extend(new_insights)
        plan[question_id]["status"] = "done"
        ctx.deps.state.status = f"Extracted {len(new_insights)} insights"

        return _as_state_snapshot(ctx)

    @agent.tool
    async def evaluate_research_confidence(
        ctx: RunContext[ResearchDeps],
    ) -> StateSnapshotEvent:
        """Evaluate overall confidence in the research findings."""
        insights = ctx.deps.state.insights
        if not insights:
            raise ValueError("No insights collected yet")

        ctx.deps.state.phase = "evaluating"
        ctx.deps.state.status = "Evaluating research confidence..."

        confidences = [i.get("confidence", 0.5) for i in insights]
        overall_confidence = sum(confidences) / len(confidences) if confidences else 0

        eval_prompt = f"""Evaluate if these insights provide a confident answer to: "{ctx.deps.state.question}"

Insights collected:
{chr(10).join([f"- {i['summary']}" for i in insights])}

Assess:
1. Do we have enough information to answer the question?
2. What gaps remain?
3. Overall confidence (0.0-1.0)

Return JSON: {{"confidence": 0.0-1.0, "gaps": ["gap1", "gap2"], "recommendation": "continue" or "finalize"}}"""

        response = await ctx.deps.client.ask(eval_prompt)

        try:
            evaluation = json.loads(response)
            overall_confidence = evaluation.get("confidence", overall_confidence)
        except json.JSONDecodeError:
            pass

        ctx.deps.state.confidence = overall_confidence
        ctx.deps.state.status = f"Confidence: {overall_confidence:.0%}"

        return _as_state_snapshot(ctx)

    @agent.tool
    async def synthesize_final_report(
        ctx: RunContext[ResearchDeps],
    ) -> StateSnapshotEvent:
        """Generate final research report with citations."""
        insights = ctx.deps.state.insights
        if not insights:
            raise ValueError("No insights to synthesize")

        ctx.deps.state.phase = "synthesizing"
        ctx.deps.state.status = "Generating final report..."

        insights_summary = []
        for i in insights:
            source_titles = [ref["document_title"] for ref in i.get("source_refs", [])]
            unique_sources = list(dict.fromkeys(source_titles))
            insights_summary.append(
                f"- {i['summary']} (sources: {', '.join(unique_sources[:2])})"
            )

        report_prompt = f"""Generate a comprehensive research report answering: "{ctx.deps.state.question}"

Based on these insights:
{chr(10).join(insights_summary)}

Create a structured report with:
- Executive Summary (2-3 sentences)
- Main Findings (bullet points)
- Conclusions
- Sources (list the document titles mentioned above)

Return JSON with format:
{{
    "title": "...",
    "summary": "...",
    "findings": ["finding1", "finding2", ...],
    "conclusions": ["conclusion1", ...],
    "sources": ["source1", "source2", ...]
}}"""

        response = await ctx.deps.client.ask(report_prompt)

        try:
            report = json.loads(response)
        except json.JSONDecodeError:
            report = {
                "title": ctx.deps.state.question,
                "summary": response[:300],
                "findings": [i["summary"] for i in insights],
                "conclusions": ["See findings above"],
                "sources": [],
            }

        citations = [
            {
                "document_uri": doc_uri,
                "document_title": doc_info["title"],
                "chunk_ids": doc_info["chunks_referenced"],
            }
            for doc_uri, doc_info in ctx.deps.state.document_registry.items()
        ]
        report["citations"] = citations

        ctx.deps.state.final_report = report
        ctx.deps.state.phase = "done"
        ctx.deps.state.status = "Research complete"

        return _as_state_snapshot(ctx)

    @agent.tool
    async def get_full_document(
        ctx: RunContext[ResearchDeps],
        document_uri: str,
    ) -> StateSnapshotEvent:
        """Retrieve and display the full content of a document by its URI."""
        ctx.deps.state.status = f"Retrieving document: {document_uri}"
        document = await ctx.deps.client.get_document_by_uri(document_uri)

        if document is None:
            ctx.deps.state.status = f"Document not found: {document_uri}"
            ctx.deps.state.current_document = {
                "uri": document_uri,
                "title": "Not Found",
                "content": f"Document with URI '{document_uri}' was not found.",
                "total_chunks": 0,
            }
        else:
            all_chunks = await ctx.deps.client.search(
                query="", limit=1000, search_type="fts"
            )
            chunks_for_doc = [
                c for c, _ in all_chunks if c.document_uri == document_uri
            ]

            ctx.deps.state.current_document = {
                "uri": document.uri or document_uri,
                "title": document.title or "Untitled",
                "content": document.content,
                "total_chunks": len(chunks_for_doc),
                "metadata": document.metadata,
            }
            ctx.deps.state.status = f"Retrieved: {document.title or document_uri}"

        return _as_state_snapshot(ctx)

    return agent
