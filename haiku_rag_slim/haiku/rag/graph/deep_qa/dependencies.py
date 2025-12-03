from pydantic import BaseModel, Field

from haiku.rag.client import HaikuRAG
from haiku.rag.graph.common.models import SearchAnswer
from haiku.rag.store.models import SearchResult


class DeepQAContext(BaseModel):
    original_question: str = Field(description="The original question")
    sub_questions: list[str] = Field(
        default_factory=list, description="Decomposed sub-questions"
    )
    qa_responses: list[SearchAnswer] = Field(
        default_factory=list, description="QA pairs collected during answering"
    )

    def add_qa_response(self, qa: SearchAnswer) -> None:
        """Add a QA response (citations already resolved)."""
        self.qa_responses.append(qa)


class DeepQADependencies(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    client: HaikuRAG = Field(description="RAG client for document operations")
    context: DeepQAContext = Field(description="Shared QA context")
    search_results: list[SearchResult] = Field(
        default_factory=list, description="Search results for citation resolution"
    )
