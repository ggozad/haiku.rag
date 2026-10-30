import pytest

from evaluations.submission import build_submission_rows, extract_prediction


@pytest.mark.parametrize(
    ("output", "prediction"),
    [
        ("...\n\nANSWER: 18.6", "18.6"),
        # T² gold stores percentages as decimals.
        ("ANSWER: 93.5%", "0.935"),
        # The scale word is not expanded, the gold is the bare number.
        ("ANSWER: $688 million", "688"),
        ("ANSWER: $1,234.5", "1234.5"),
        ("ANSWER: −1.9", "-1.9"),
        ("We saw 72.8 in the table but recomputed.\nANSWER: 82.8", "82.8"),
        ("", ""),
        (None, ""),
        ("ANSWER: not reported", ""),
    ],
    ids=[
        "answer-line",
        "percent-becomes-fraction",
        "strips-currency-and-scale-word",
        "strips-currency-and-commas",
        "unicode-minus",
        "uses-answer-line-not-reasoning",
        "empty-output",
        "no-output",
        "no-number",
    ],
)
def test_extract_prediction(output: str | None, prediction: str) -> None:
    assert extract_prediction(output) == prediction


class TestBuildSubmissionRows:
    def _preds(self) -> list[dict[str, str]]:
        return [
            {"id": "finqa_dev_0", "question": "Q1?", "output": "ANSWER: 127.4"},
            {"id": "finqa_dev_1", "question": "Q2?", "output": ""},  # null
        ]

    def _retrieval(self) -> dict[str, list[str]]:
        return {
            "Q1?": ["ctx_a", "ctx_b", "ctx_c", "ctx_d"],
            "Q2?": ["ctx_e", "ctx_f"],
        }

    def test_topk_list_and_fields(self) -> None:
        rows = build_submission_rows(
            self._preds(), self._retrieval(), subset="FinQA", topk=3
        )
        assert rows[0] == {
            "id": "finqa_dev_0",
            "subset": "FinQA",
            "context_id": ["ctx_a", "ctx_b", "ctx_c"],
            "prediction": "127.4",
        }
        # null prediction -> blank string (counted wrong); ranking still attached
        assert rows[1]["prediction"] == ""
        assert rows[1]["context_id"] == ["ctx_e", "ctx_f"]

    def test_topk_one_emits_single_string(self) -> None:
        rows = build_submission_rows(
            self._preds(), self._retrieval(), subset="FinQA", topk=1
        )
        assert rows[0]["context_id"] == "ctx_a"

    def test_missing_retrieval_is_empty(self) -> None:
        rows = build_submission_rows(
            [{"id": "x", "question": "unseen?", "output": "ANSWER: 1"}],
            self._retrieval(),
            subset="FinQA",
            topk=3,
        )
        assert rows[0]["context_id"] == []


class TestEnterpriseRagAnswers:
    def test_row_per_case_with_deduped_cited_documents(self) -> None:
        from evaluations.submission import enterprise_rag_answers

        rows = [
            {
                "key": "qst_0001",
                "answer": "Ten MiB.",
                "attributes": {"cited_uris": ["dsid_b", "dsid_a", "dsid_b"]},
            }
        ]
        assert enterprise_rag_answers(rows) == [
            {
                "question_id": "qst_0001",
                "answer": "Ten MiB.",
                "document_ids": ["dsid_b", "dsid_a"],
            }
        ]

    def test_aborted_case_keeps_an_empty_row(self) -> None:
        from evaluations.submission import enterprise_rag_answers

        rows = [{"key": "qst_0002", "answer": None, "attributes": {}}]
        assert enterprise_rag_answers(rows) == [
            {"question_id": "qst_0002", "answer": "", "document_ids": []}
        ]
