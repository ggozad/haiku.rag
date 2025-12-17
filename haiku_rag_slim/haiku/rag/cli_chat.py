"""Interactive CLI chat loop for research graph with human-in-the-loop."""

import asyncio
import json

from pydantic_ai import Agent
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from haiku.rag.client import HaikuRAG
from haiku.rag.config import get_config
from haiku.rag.config.models import AppConfig
from haiku.rag.graph.agui.emitter import AGUIEmitter
from haiku.rag.graph.research.dependencies import ResearchContext
from haiku.rag.graph.research.graph import build_research_graph
from haiku.rag.graph.research.models import ResearchReport
from haiku.rag.graph.research.state import HumanDecision, ResearchDeps, ResearchState
from haiku.rag.utils import get_model

INITIAL_CHAT_PROMPT = """You are a research assistant. The user hasn't started a research task yet.

You can:
1. Chat with the user - greet them, answer questions about what you can do
2. Detect when they want to research something

## Actions:
- "chat": User is chatting, greeting, or asking questions (set message with your response)
- "research": User wants to research a topic (extract the research question into research_question)

## Guidelines:
- If the user provides a clear research question or topic, set action="research" and extract the question
- If the user is just chatting or asking what you can do, set action="chat" and respond helpfully
- Be friendly and explain you can help them research topics by searching a knowledge base

Examples:
- "hi" → action="chat", message="Hello! I'm a research assistant. I can help you research topics by searching through documents and synthesizing findings. What would you like to explore?"
- "what can you do?" → action="chat", message="I help you conduct research! Give me a question or topic, and I'll break it into sub-questions, search for answers, and synthesize a report. What are you curious about?"
- "tell me about Python's memory management" → action="research", research_question="How does Python's memory management work?"
- "I want to understand how RAG systems work" → action="research", research_question="How do RAG (Retrieval-Augmented Generation) systems work?"
"""

RESEARCH_ASSISTANT_PROMPT = """You are a research assistant helping the user conduct research on a topic.

You are at a decision point in the research workflow. You can:
1. Chat with the user - answer questions, discuss the research, make suggestions
2. Take workflow actions when the user requests them

## Workflow Actions (set in the action field):
- "search": Search the pending questions (user says: "go", "search", "yes", "continue", "looks good")
- "synthesize": Generate final report (user says: "done", "finish", "synthesize", "generate report")
- "add_questions": Add NEW research questions to the existing list
- "modify_questions": REPLACE all pending questions with a new list (use when user wants to remove, keep only certain questions, or change the questions)
- "chat": Have a conversation without modifying questions

## IMPORTANT - Modifying Questions:
- "use only the first question" → action="modify_questions", questions=[first question from the list]
- "drop questions 2 and 3" → action="modify_questions", questions=[remaining questions]
- "keep only questions about X" → action="modify_questions", questions=[filtered list]
- "remove the duplicate" → action="modify_questions", questions=[deduplicated list]
- When user wants to reduce/filter/keep-only, use "modify_questions" NOT "chat"

## Guidelines:
- If the user wants to modify the question list in ANY way (remove, keep only, filter), use "modify_questions"
- For "modify_questions", include ALL questions that should remain in the questions field
- You can combine "chat" with a message to explain what you're doing
- If just chatting without changes, set action="chat" and provide helpful response in message
"""


async def initial_chat(
    user_message: str,
    config: AppConfig,
) -> HumanDecision:
    """Handle initial conversation before research starts.

    Args:
        user_message: The user's message
        config: Application configuration

    Returns:
        HumanDecision with chat response or research question
    """
    agent: Agent[None, HumanDecision] = Agent(
        model=get_model(config.research.model, config),
        output_type=HumanDecision,
        instructions=INITIAL_CHAT_PROMPT,
        retries=2,
    )

    result = await agent.run(user_message)
    return result.output


async def interpret_user_decision(
    user_message: str,
    sub_questions: list[str],
    qa_responses: list[dict],
    config: AppConfig,
) -> HumanDecision:
    """Interpret a natural language user message into a HumanDecision.

    Args:
        user_message: The user's natural language input
        sub_questions: Current sub-questions pending search
        qa_responses: Answers already collected
        config: Application configuration

    Returns:
        HumanDecision with the interpreted action, questions, and/or message
    """
    agent: Agent[None, HumanDecision] = Agent(
        model=get_model(config.research.model, config),
        output_type=HumanDecision,
        instructions=RESEARCH_ASSISTANT_PROMPT,
        retries=2,
    )

    # Build context with full research state
    answers_summary = ""
    if qa_responses:
        answers_parts = []
        for qa in qa_responses:
            conf = f"{qa['confidence']:.0%}" if qa.get("confidence") else "N/A"
            answers_parts.append(
                f"Q: {qa['query']}\nA: {qa['answer'][:300]}... (confidence: {conf})"
            )
        answers_summary = "\n\n".join(answers_parts)

    context = f"""Current research state:
- Answers collected: {len(qa_responses)}
- Pending questions to search: {len(sub_questions)}

Pending questions:
{chr(10).join(f"- {q}" for q in sub_questions) if sub_questions else "(none)"}

{f"Collected answers:{chr(10)}{answers_summary}" if answers_summary else ""}

User message: {user_message}"""

    result = await agent.run(context)
    return result.output


async def run_interactive_research(
    question: str,
    client: HaikuRAG,
    config: AppConfig | None = None,
    search_filter: str | None = None,
) -> ResearchReport:
    """Run interactive research with human-in-the-loop decision points.

    Args:
        question: The research question
        client: HaikuRAG client for document operations
        config: Application configuration (uses global config if None)
        search_filter: Optional SQL WHERE clause to filter documents

    Returns:
        ResearchReport with the final synthesis
    """
    config = config or get_config()
    console = Console()

    # Build interactive graph
    graph = build_research_graph(config=config, include_plan=True, interactive=True)

    # Create async queue for human input
    human_input_queue: asyncio.Queue[HumanDecision] = asyncio.Queue()

    # Create emitter
    emitter: AGUIEmitter[ResearchState, ResearchReport] = AGUIEmitter()

    # Create deps with queue
    deps = ResearchDeps(
        client=client,
        agui_emitter=emitter,
        human_input_queue=human_input_queue,
        interactive=True,
    )

    # Create initial state
    context = ResearchContext(original_question=question)
    state = ResearchState.from_config(context=context, config=config)
    state.search_filter = search_filter

    # Start the run
    emitter.start_run(state)

    # Run graph in background task
    async def run_graph() -> ResearchReport:
        try:
            result = await graph.run(state=state, deps=deps)
            emitter.finish_run(result)
            return result
        except Exception as e:
            emitter.error(e)
            raise

    graph_task = asyncio.create_task(run_graph())

    # Process events and handle human decision points
    try:
        async for event in emitter:
            event_type = event.get("type")

            if event_type == "STEP_STARTED":
                step_name = event.get("stepName", "")
                if step_name == "plan":
                    console.print("[dim]Planning research...[/dim]")
                elif step_name.startswith("search:"):
                    query = step_name.replace("search: ", "")
                    console.print(f"[dim]Searching: {query}[/dim]")
                elif step_name == "synthesize":
                    console.print("[dim]Synthesizing report...[/dim]")

            elif event_type == "STATE_SNAPSHOT" or event_type == "STATE_DELTA":
                # State updated, could show progress
                pass

            elif event_type == "TOOL_CALL_START":
                tool_name = event.get("toolCallName")
                if tool_name == "human_decision":
                    # Will get args in next event
                    pass

            elif event_type == "TOOL_CALL_ARGS":
                delta = event.get("delta", "{}")
                args = json.loads(delta) if isinstance(delta, str) else delta
                original_question = args.get("original_question", "")
                sub_questions = list(args.get("sub_questions", []))
                qa_responses = args.get("qa_responses", [])
                iterations = args.get("iterations", 0)

                # Loop for modifications until user wants to proceed
                while True:
                    # Show research state
                    console.print()
                    console.print(
                        Panel(
                            f"[bold]{original_question}[/bold]",
                            title="Research Question",
                            border_style="blue",
                        )
                    )

                    # Show collected answers
                    if qa_responses:
                        answers_text = []
                        for i, qa in enumerate(qa_responses, 1):
                            conf = (
                                f"{qa['confidence']:.0%}"
                                if qa.get("confidence")
                                else "N/A"
                            )
                            answer_preview = (
                                qa["answer"][:200] + "..."
                                if len(qa["answer"]) > 200
                                else qa["answer"]
                            )
                            answers_text.append(
                                f"[cyan]{i}. {qa['query']}[/cyan]\n"
                                f"   [dim]Confidence: {conf} | Citations: {qa.get('citations_count', 0)}[/dim]\n"
                                f"   {answer_preview}"
                            )
                        console.print(
                            Panel(
                                "\n\n".join(answers_text),
                                title=f"Answers Collected ({len(qa_responses)})",
                                border_style="green",
                            )
                        )

                    # Show pending questions
                    if sub_questions:
                        console.print(
                            Panel(
                                "\n".join(
                                    f"{i + 1}. {q}" for i, q in enumerate(sub_questions)
                                ),
                                title="Pending Questions to Search",
                                border_style="cyan",
                            )
                        )
                    else:
                        console.print("[dim]No pending questions.[/dim]")

                    if iterations > 0:
                        console.print(f"[dim]Iteration: {iterations}[/dim]")

                    # Prompt user with context-aware hints
                    console.print()
                    hints = []
                    if sub_questions:
                        hints.append("search questions")
                        hints.append("modify questions")
                    if qa_responses:
                        hints.append("generate report")
                    hint_text = f" [dim]({', '.join(hints)})[/dim]" if hints else ""
                    user_input = Prompt.ask(
                        f"[bold]What would you like to do?[/bold]{hint_text}"
                    )

                    # Chat with research assistant
                    console.print("[dim]Thinking...[/dim]")
                    decision = await interpret_user_decision(
                        user_message=user_input,
                        sub_questions=sub_questions,
                        qa_responses=qa_responses,
                        config=config,
                    )

                    # Handle modifications and chat locally, continue loop
                    if decision.action == "chat":
                        if decision.message:
                            console.print(
                                f"\n[bold cyan]Assistant:[/bold cyan] {decision.message}"
                            )
                        continue
                    elif decision.action == "add_questions" and decision.questions:
                        sub_questions.extend(decision.questions)
                        console.print(
                            f"[green]Added {len(decision.questions)} question(s)[/green]"
                        )
                        continue
                    elif decision.action == "modify_questions" and decision.questions:
                        sub_questions = list(decision.questions)
                        console.print(
                            f"[green]Replaced with {len(decision.questions)} question(s)[/green]"
                        )
                        continue

                    # User wants to proceed - send final decision
                    action_display = {
                        "search": "Searching questions",
                        "synthesize": "Generating report",
                    }
                    console.print(
                        f"[dim]→ {action_display.get(decision.action, decision.action)}[/dim]"
                    )

                    # Include any accumulated question changes
                    if decision.action == "search":
                        decision = HumanDecision(
                            action="modify_questions", questions=sub_questions
                        )

                    await human_input_queue.put(decision)
                    break

            elif event_type == "TEXT_MESSAGE_CHUNK":
                # Log message from graph
                message = event.get("delta", "")
                if message:
                    console.print(f"[dim]{message}[/dim]")

            elif event_type == "RUN_FINISHED":
                break

            elif event_type == "RUN_ERROR":
                error_msg = event.get("message", "Unknown error")
                console.print(f"[red]Error: {error_msg}[/red]")
                break

        # Wait for graph to complete
        report = await graph_task
        return report

    except Exception as e:
        graph_task.cancel()
        raise e
    finally:
        await emitter.close()


async def run_chat_loop(
    client: HaikuRAG,
    config: AppConfig | None = None,
    search_filter: str | None = None,
    question: str | None = None,
) -> None:
    """Run an interactive chat loop for research.

    Args:
        client: HaikuRAG client for document operations
        config: Application configuration (uses global config if None)
        search_filter: Optional SQL WHERE clause to filter documents
        question: Optional initial research question (skips initial chat if provided)
    """
    config = config or get_config()
    console = Console()

    console.print(
        Panel(
            "[bold cyan]Interactive Research Mode[/bold cyan]\n\n"
            "Chat with me or tell me what you'd like to research.\n"
            "Type [green]exit[/green] or [green]quit[/green] to end the session.",
            title="haiku.rag Research Assistant",
            border_style="cyan",
        )
    )

    while True:
        try:
            # Use provided question or get one through conversation
            if question:
                research_question = question
                console.print(f"[dim]Starting research: {research_question}[/dim]")
                question = None  # Clear so subsequent loops go through chat
            else:
                # Initial conversation loop - chat until user wants to research
                research_question = None
                while research_question is None:
                    user_input = Prompt.ask("\n[bold blue]You[/bold blue]")

                    if not user_input.strip():
                        continue

                    if user_input.lower().strip() in ("exit", "quit", "q"):
                        console.print("[dim]Goodbye![/dim]")
                        return

                    console.print("[dim]Thinking...[/dim]")
                    decision = await initial_chat(user_input, config)

                    if decision.action == "research" and decision.research_question:
                        research_question = decision.research_question
                        console.print(
                            f"[dim]Starting research: {research_question}[/dim]"
                        )
                    elif decision.action == "chat" and decision.message:
                        console.print(
                            f"\n[bold cyan]Assistant:[/bold cyan] {decision.message}"
                        )
                    else:
                        # Fallback - treat as research question
                        research_question = user_input

            console.print()
            report = await run_interactive_research(
                question=research_question,
                client=client,
                config=config,
                search_filter=search_filter,
            )

            # Display final report
            console.print()
            console.print(
                Panel(
                    Markdown(f"## {report.title}\n\n{report.executive_summary}"),
                    title="Research Report",
                    border_style="green",
                )
            )

            if report.main_findings:
                findings = "\n".join(f"- {f}" for f in report.main_findings[:5])
                console.print(Markdown(f"**Key Findings:**\n{findings}"))

            if report.conclusions:
                conclusions = "\n".join(f"- {c}" for c in report.conclusions[:3])
                console.print(Markdown(f"**Conclusions:**\n{conclusions}"))

            console.print(Markdown(f"**Sources:** {report.sources_summary}"))

        except KeyboardInterrupt:
            console.print("\n[dim]Goodbye![/dim]")
            return
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


def interactive_research(
    client: HaikuRAG,
    config: AppConfig | None = None,
    search_filter: str | None = None,
    question: str | None = None,
) -> None:
    """Entry point for interactive research mode.

    Args:
        client: HaikuRAG client for document operations
        config: Application configuration (uses global config if None)
        search_filter: Optional SQL WHERE clause to filter documents
        question: Optional initial research question (skips initial chat if provided)
    """
    asyncio.run(run_chat_loop(client, config, search_filter, question))
