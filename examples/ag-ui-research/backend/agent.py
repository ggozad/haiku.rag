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

    # Research plan
    plan: list[dict] = []  # [{id, question, status: pending|searching|done}]
    current_question_index: int = 0

    # Search results (live updates)
    current_search: dict | None = (
        None  # {query, type, results: [{chunk, score, expanded}]}
    )

    # Accumulated findings
    insights: list[dict] = []  # [{summary, confidence, sources}]

    # Final output
    confidence: float = 0.0
    final_report: dict | None = None


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
2. Wait for user approval before proceeding
3. For each sub-question:
   - Announce what you're searching for
   - Execute search and show results with scores
   - Extract insights from the results
   - Ask user if they want to continue to next question
4. Evaluate overall confidence in your findings
5. Ask user if confident enough or should search more
6. Synthesize final report with citations

Be transparent: always announce what you're doing before you do it.
Show search scores, explain your reasoning, and involve the user in decisions.
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

        # Process results
        results = []
        for chunk, score in search_results:
            # Check if this chunk was expanded
            if chunk.id in expanded_map:
                expanded_chunk, _ = expanded_map[chunk.id]
                result_data = {
                    "chunk": expanded_chunk.content[:500],  # Truncate for display
                    "score": round(score, 3),
                    "source": chunk.document_title or chunk.document_uri or "Unknown",
                    "expanded": True,
                }
            else:
                result_data = {
                    "chunk": chunk.content[:500],  # Truncate for display
                    "score": round(score, 3),
                    "source": chunk.document_title or chunk.document_uri or "Unknown",
                    "expanded": False,
                }

            results.append(result_data)

        # Update state
        ctx.deps.state.current_search = {
            "query": question,
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
    ) -> StateSnapshotEvent:
        """Extract key insights from current search results."""
        current_search = ctx.deps.state.current_search
        if not current_search:
            raise ValueError("No current search results to analyze")

        # Update state
        ctx.deps.state.phase = "analyzing"
        ctx.deps.state.status = "Extracting insights from results..."

        # Build context from results
        context = "\n\n".join(
            [f"[Source: {r['source']}] {r['chunk']}" for r in current_search["results"]]
        )

        # Use LLM to extract insights
        extract_prompt = f"""Analyze these search results and extract 1-3 key insights that help answer the question: "{current_search["query"]}"

Search Results:
{context}

Return a JSON array of insights with format:
[{{"summary": "brief insight", "confidence": 0.0-1.0, "sources": ["source1", "source2"]}}]"""

        response = await ctx.deps.client.ask(extract_prompt)

        # Parse insights
        import json

        try:
            new_insights = json.loads(response)
        except json.JSONDecodeError:
            # Fallback: create simple insight
            new_insights = [
                {
                    "summary": response[:200],
                    "confidence": 0.7,
                    "sources": [r["source"] for r in current_search["results"][:3]],
                }
            ]

        # Add to accumulated insights
        ctx.deps.state.insights.extend(new_insights)

        # Clear current search
        ctx.deps.state.current_search = None
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

        # Build report prompt
        report_prompt = f"""Generate a comprehensive research report answering: "{ctx.deps.state.question}"

Based on these insights:
{chr(10).join([f"- {i['summary']} (sources: {', '.join(i.get('sources', [])[:2])})" for i in insights])}

Create a structured report with:
- Executive Summary (2-3 sentences)
- Main Findings (bullet points)
- Conclusions
- Sources

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
                "sources": list(
                    set([s for i in insights for s in i.get("sources", [])])
                ),
            }

        # Update state
        ctx.deps.state.final_report = report
        ctx.deps.state.phase = "done"
        ctx.deps.state.status = "Research complete"
        print("[AGENT] Report complete, sending state snapshot")

        return _as_state_snapshot(ctx)

    return agent
