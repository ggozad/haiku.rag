import dspy


class QASignature(dspy.Signature):
    """Answer a question based on a document knowledge base."""

    question: str = dspy.InputField(desc="The user's question to answer")
    answer: str = dspy.OutputField(desc="The answer based on the knowledge base")


class QAModule(dspy.Module):
    """
    DSPy module for QA prompt optimization.

    This module wraps the QA task signature with customizable instructions.
    MIPROv2 optimizes these instructions to improve QA performance.
    """

    def __init__(self, instructions: str):
        super().__init__()
        self.predict = dspy.Predict(QASignature)
        self.predict.signature = self.predict.signature.with_instructions(instructions)

    def forward(self, question: str) -> dspy.Prediction:
        return self.predict(question=question)

    def get_instructions(self) -> str:
        """Extract current instructions from the signature."""
        return str(getattr(self.predict.signature, "instructions", ""))
