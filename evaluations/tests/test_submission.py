from evaluations.submission import build_submission_rows, extract_prediction


class TestExtractPrediction:
    def test_answer_line_integer(self) -> None:
        assert extract_prediction("...\n\nANSWER: 18.6") == "18.6"

    def test_percent_becomes_fraction(self) -> None:
        # T² gold stores percentages as decimals.
        assert extract_prediction("ANSWER: 93.5%") == "0.935"

    def test_strips_currency_commas_and_scale_word(self) -> None:
        # "$688 million" -> 688 (scale word not expanded; gold is the bare number)
        assert extract_prediction("ANSWER: $688 million") == "688"
        assert extract_prediction("ANSWER: $1,234.5") == "1234.5"

    def test_unicode_minus(self) -> None:
        assert extract_prediction("ANSWER: −1.9") == "-1.9"

    def test_uses_answer_line_not_reasoning(self) -> None:
        out = "We saw 72.8 in the table but recomputed.\nANSWER: 82.8"
        assert extract_prediction(out) == "82.8"

    def test_empty_output_is_blank(self) -> None:
        assert extract_prediction("") == ""
        assert extract_prediction(None) == ""

    def test_no_number_is_blank(self) -> None:
        assert extract_prediction("ANSWER: not reported") == ""


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
