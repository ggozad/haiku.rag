"""Pydantic AI research agent for haiku.rag with AG-UI protocol."""

from __future__ import annotations

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
    phase: str = "idle"  # idle|planning|searching|analyzing|evaluating|done
    status: str = ""  # Human-readable message

    # Research plan with embedded search results
    plan: list[
        dict
    ] = []  # [{id, question, status: pending|searching|done, search_results: {type, results: [...]}}]
    current_question_index: int = 0

    # Accumulated findings
    insights: list[
        dict
    ] = []  # [{summary, confidence, source_refs: [{chunk_id, document_uri, document_title, chunk_position}]}]

    # Document registry - tracks all referenced documents
    document_registry: dict[
        str, dict
    ] = {}  # {doc_uri: {title, chunks_referenced: [chunk_id]}}

    # Document viewer state
    current_document: dict | None = None  # {uri, title, content, total_chunks}

    # Final output
    confidence: float = 0.0
    final_report: dict | None = (
        None  # {title, summary, findings, conclusions, citations: [{document_uri, document_title, chunk_ids}]}
    )


@dataclass
class ResearchDeps(StateDeps[ResearchState]):
    """Dependencies for the research agent with HaikuRAG client."""

    client: HaikuRAG


def _as_state_snapshot(ctx: RunContext[ResearchDeps]) -> StateSnapshotEvent:
    """Helper to create state snapshot event for AG-UI synchronization."""
    return StateSnapshotEvent(type=EventType.STATE_SNAPSHOT, snapshot=ctx.deps.state)


def create_agent(
    qa_provider: str = Config.QA_PROVIDER, qa_model: str = Config.QA_MODEL
) -> Agent[ResearchDeps, str]:
    """Create and configure the research agent.

    Args:
        qa_provider: QA provider for the agent (default: from Config.QA_PROVIDER)
        qa_model: Model name to use (default: from Config.QA_MODEL)
    """
    agent = Agent(
        model=get_model(qa_provider, qa_model),
        deps_type=ResearchDeps,
        instructions="""You are a research co-pilot powered by haiku.rag.

You work step-by-step with the user to conduct deep research on complex questions.

Your workflow:
1. When user asks a question, propose a research plan (3-5 sub-questions)
2. Wait for user approval before proceeding to the search phase
3. Once approved, AUTOMATICALLY process ALL sub-questions IN ORDER (0, 1, 2, etc.) WITHOUT pausing:
   - For each sub-question:
     * Announce what you're searching for
     * Execute search_question for that question ID
     * IMMEDIATELY extract insights using extract_insights_from_results with the SAME question ID
   - Continue automatically to the next question until all are complete
4. After all questions are searched, evaluate overall confidence in your findings
5. Ask user if confident enough or should search more
6. Synthesize final report with complete citations

CRITICAL RULES:
- ALWAYS call search_question BEFORE extract_insights_from_results for each question
- Process questions in sequence: search Q0 → extract Q0 → search Q1 → extract Q1, etc.
- NEVER skip ahead to extract insights for a question you haven't searched yet
- In the search phase, DO NOT pause between questions - process all questions automatically

Document Viewing:
- Users can request to view the full content of any cited document
- When a user asks to "show document X" or "view source Y", use the get_full_document tool
- Document URIs are tracked automatically as you search
- The final report includes structured citations linking back to source documents

Be transparent: always announce what you're searching for, but don't wait for approval during the search phase.
Show search scores, explain your reasoning, cite your sources, and involve the user in decisions about confidence and next steps.
""",
    )

    @agent.tool
    async def propose_research_plan(
        ctx: RunContext[ResearchDeps], question: str
    ) -> StateSnapshotEvent:
        """Propose a research plan by decomposing the question into sub-questions.

        Args:
            question: The main research question to decompose
        """
        # Update state with the question
        ctx.deps.state.question = question
        ctx.deps.state.phase = "planning"
        ctx.deps.state.status = "Decomposing question into sub-questions..."
        print(
            f"[AGENT] Updated state: phase={ctx.deps.state.phase}, question={question}"
        )

        # Use LLM to decompose the question
        decompose_prompt = f"""Break down this research question into 3-5 specific sub-questions that would help answer it comprehensively.

Research Question: {question}

Return ONLY a JSON array of sub-questions, like: ["Question 1?", "Question 2?", ...]"""

        response = await ctx.deps.client.ask(decompose_prompt)

        # Parse the response (simplified - assume it returns reasonable sub-questions)
        import json

        try:
            sub_questions = json.loads(response)
        except json.JSONDecodeError:
            # Fallback: split by newlines and clean up
            sub_questions = [
                q.strip().lstrip("0123456789.-) ")
                for q in response.split("\n")
                if q.strip()
            ][:5]

        # Create plan
        plan = [
            {"id": i, "question": q, "status": "pending"}
            for i, q in enumerate(sub_questions)
        ]

        ctx.deps.state.plan = plan
        ctx.deps.state.current_question_index = 0
        ctx.deps.state.status = f"Proposed plan with {len(plan)} sub-questions"
        print(f"[AGENT] Plan created with {len(plan)} sub-questions")
        print("[AGENT] Sending state snapshot to frontend")

        return _as_state_snapshot(ctx)

    @agent.tool
    async def search_question(
        ctx: RunContext[ResearchDeps],
        question_id: int,
        search_type: str = "hybrid",
    ) -> StateSnapshotEvent:
        """Execute search for a specific sub-question.

        Args:
            question_id: ID of the sub-question from the plan
            search_type: Type of search (hybrid, vector, or fts)
        """
        # Get the question from plan
        plan = ctx.deps.state.plan
        if question_id >= len(plan):
            raise ValueError(f"Question ID {question_id} not found in plan")

        question = plan[question_id]["question"]

        # Update state
        ctx.deps.state.phase = "searching"
        ctx.deps.state.current_question_index = question_id
        ctx.deps.state.status = f"Searching: {question}"
        plan[question_id]["status"] = "searching"

        # Execute search
        search_results = await ctx.deps.client.search(
            question, limit=5, search_type=search_type
        )

        # Expand context for top 3 results
        if len(search_results) > 0:
            # Get top 3 for context expansion
            top_results = search_results[:3]
            expanded_results = await ctx.deps.client.expand_context(
                top_results, radius=2
            )

            # Create a map of expanded chunks
            expanded_map = {
                chunk.id: (chunk, score) for chunk, score in expanded_results
            }
        else:
            expanded_map = {}

        # Process results and update document registry
        results = []
        for chunk, score in search_results:
            # Update document registry
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

            # Check if this chunk was expanded
            if chunk.id in expanded_map:
                expanded_chunk, _ = expanded_map[chunk.id]
                result_data = {
                    "chunk": expanded_chunk.content[:500],  # Truncate for display
                    "chunk_id": chunk.id,
                    "document_uri": doc_uri,
                    "document_title": doc_title,
                    "chunk_position": chunk.order,
                    "full_chunk_content": expanded_chunk.content,
                    "score": round(score, 3),
                    "expanded": True,
                }
            else:
                result_data = {
                    "chunk": chunk.content[:500],  # Truncate for display
                    "chunk_id": chunk.id,
                    "document_uri": doc_uri,
                    "document_title": doc_title,
                    "chunk_position": chunk.order,
                    "full_chunk_content": chunk.content,
                    "score": round(score, 3),
                    "expanded": False,
                }

            results.append(result_data)

        # Store search results in the plan item
        plan[question_id]["search_results"] = {
            "type": search_type,
            "results": results,
        }
        plan[question_id]["status"] = "done"
        ctx.deps.state.status = f"Found {len(results)} results"
        print("[AGENT] Search complete, sending state snapshot")

        return _as_state_snapshot(ctx)

    @agent.tool
    async def extract_insights_from_results(
        ctx: RunContext[ResearchDeps],
        question_id: int,
    ) -> StateSnapshotEvent:
        """Extract key insights from search results for a specific question.

        IMPORTANT: You must call search_question for this question_id BEFORE calling this tool.
        This tool requires that search results already exist for the given question.

        Args:
            question_id: ID of the question whose results to analyze
        """
        plan = ctx.deps.state.plan
        if question_id >= len(plan):
            raise ValueError(f"Question ID {question_id} not found in plan")

        question_item = plan[question_id]
        if "search_results" not in question_item:
            raise ValueError(
                f"No search results found for question ID {question_id}. "
                f"You must call search_question(question_id={question_id}) first before extracting insights."
            )

        search_results = question_item["search_results"]

        # Update state
        ctx.deps.state.phase = "analyzing"
        ctx.deps.state.status = "Extracting insights from results..."

        # Build context from results with chunk IDs for reference
        context_parts = []
        for idx, r in enumerate(search_results["results"]):
            context_parts.append(
                f"[Result {idx}] [Source: {r['document_title']}] {r['full_chunk_content']}"
            )
        context = "\n\n".join(context_parts)

        # Use LLM to extract insights
        question_text = question_item["question"]
        extract_prompt = f"""Analyze these search results and extract 1-3 key insights that help answer the question: "{question_text}"

Search Results:
{context}

For each insight, reference which result numbers (0, 1, 2, etc.) support it.

Return a JSON array of insights with format:
[{{"summary": "brief insight", "confidence": 0.0-1.0, "result_indices": [0, 1, ...]}}]"""

        response = await ctx.deps.client.ask(extract_prompt)

        # Parse insights
        import json

        try:
            raw_insights = json.loads(response)
        except json.JSONDecodeError:
            # Fallback: create simple insight referencing all results
            raw_insights = [
                {
                    "summary": response[:200],
                    "confidence": 0.7,
                    "result_indices": list(
                        range(min(3, len(search_results["results"])))
                    ),
                }
            ]

        # Convert result indices to structured source references
        new_insights = []
        for insight in raw_insights:
            result_indices = insight.get("result_indices", [])
            source_refs = []

            for idx in result_indices:
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

        # Add to accumulated insights
        ctx.deps.state.insights.extend(new_insights)

        ctx.deps.state.status = f"Extracted {len(new_insights)} insights"
        print("[AGENT] Insights extracted, sending state snapshot")

        return _as_state_snapshot(ctx)

    @agent.tool
    async def evaluate_research_confidence(
        ctx: RunContext[ResearchDeps],
    ) -> StateSnapshotEvent:
        """Evaluate overall confidence in the research findings."""
        insights = ctx.deps.state.insights
        if not insights:
            raise ValueError("No insights collected yet")

        # Update state
        ctx.deps.state.phase = "evaluating"
        ctx.deps.state.status = "Evaluating research confidence..."

        # Calculate confidence (simple average of insight confidences)
        confidences = [i.get("confidence", 0.5) for i in insights]
        overall_confidence = sum(confidences) / len(confidences) if confidences else 0

        # Use LLM to evaluate completeness
        eval_prompt = f"""Evaluate if these insights provide a confident answer to: "{ctx.deps.state.question}"

Insights collected:
{chr(10).join([f"- {i['summary']}" for i in insights])}

Assess:
1. Do we have enough information to answer the question?
2. What gaps remain?
3. Overall confidence (0.0-1.0)

Return JSON: {{"confidence": 0.0-1.0, "gaps": ["gap1", "gap2"], "recommendation": "continue" or "finalize"}}"""

        response = await ctx.deps.client.ask(eval_prompt)

        # Parse evaluation
        import json

        try:
            evaluation = json.loads(response)
            overall_confidence = evaluation.get("confidence", overall_confidence)
        except json.JSONDecodeError:
            evaluation = {
                "confidence": overall_confidence,
                "gaps": [],
                "recommendation": "finalize"
                if overall_confidence > 0.7
                else "continue",
            }

        # Update state
        ctx.deps.state.confidence = overall_confidence
        ctx.deps.state.status = f"Confidence: {overall_confidence:.0%}"
        print("[AGENT] Confidence evaluated, sending state snapshot")

        return _as_state_snapshot(ctx)

    @agent.tool
    async def synthesize_final_report(
        ctx: RunContext[ResearchDeps],
    ) -> StateSnapshotEvent:
        """Generate final research report with citations."""
        insights = ctx.deps.state.insights
        if not insights:
            raise ValueError("No insights to synthesize")

        # Update state
        ctx.deps.state.phase = "synthesizing"
        ctx.deps.state.status = "Generating final report..."

        # Build summary of insights with source information
        insights_summary = []
        for i in insights:
            source_titles = [ref["document_title"] for ref in i.get("source_refs", [])]
            unique_sources = list(
                dict.fromkeys(source_titles)
            )  # Preserve order, remove duplicates
            insights_summary.append(
                f"- {i['summary']} (sources: {', '.join(unique_sources[:2])})"
            )

        # Build report prompt
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

        # Parse report
        import json

        try:
            report = json.loads(response)
        except json.JSONDecodeError:
            # Fallback report
            report = {
                "title": ctx.deps.state.question,
                "summary": response[:300],
                "findings": [i["summary"] for i in insights],
                "conclusions": ["See findings above"],
                "sources": [],
            }

        # Build structured citations from document registry
        citations = []
        for doc_uri, doc_info in ctx.deps.state.document_registry.items():
            citations.append(
                {
                    "document_uri": doc_uri,
                    "document_title": doc_info["title"],
                    "chunk_ids": doc_info["chunks_referenced"],
                }
            )

        # Add citations to report
        report["citations"] = citations

        # Update state
        ctx.deps.state.final_report = report
        ctx.deps.state.phase = "done"
        ctx.deps.state.status = "Research complete"
        print("[AGENT] Report complete, sending state snapshot")

        return _as_state_snapshot(ctx)

    @agent.tool
    async def get_full_document(
        ctx: RunContext[ResearchDeps],
        document_uri: str,
    ) -> StateSnapshotEvent:
        """Retrieve and display the full content of a document by its URI.

        Args:
            document_uri: The URI identifier of the document to retrieve
        """
        # Update state
        ctx.deps.state.status = f"Retrieving document: {document_uri}"

        # Get document from haiku.rag
        document = await ctx.deps.client.get_document_by_uri(document_uri)

        if document is None:
            ctx.deps.state.status = f"Document not found: {document_uri}"
            ctx.deps.state.current_document = {
                "uri": document_uri,
                "title": "Not Found",
                "content": f"Document with URI '{document_uri}' was not found in the database.",
                "total_chunks": 0,
            }
        else:
            # Get all chunks for this document to count them
            all_chunks = await ctx.deps.client.search(
                query="",  # Empty query to get all chunks
                limit=1000,
                search_type="fts",
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
            ctx.deps.state.status = (
                f"Retrieved document: {document.title or document_uri}"
            )

        print(f"[AGENT] Document retrieved: {document_uri}")

        return _as_state_snapshot(ctx)

    return agent
