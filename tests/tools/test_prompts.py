from haiku.rag.tools.prompts import build_tools_prompt


def test_empty_features():
    result = build_tools_prompt([])
    assert result == ""


def test_single_feature_search():
    result = build_tools_prompt(["search"])
    assert "search" in result
    assert "document_name" in result.lower()


def test_single_feature_documents():
    result = build_tools_prompt(["documents"])
    assert "list_documents" in result
    assert "summarize_document" in result
    assert "get_document" in result


def test_single_feature_qa():
    result = build_tools_prompt(["qa"])
    assert "ask" in result
    assert "document_name" in result.lower()


def test_single_feature_analysis():
    result = build_tools_prompt(["analysis"])
    assert "analyze" in result


def test_multiple_features():
    result = build_tools_prompt(["search", "qa", "documents"])
    assert "search" in result
    assert "ask" in result
    assert "list_documents" in result


def test_search_and_qa_both_add_document_name_examples():
    result = build_tools_prompt(["search", "qa"])
    assert "search for embeddings" in result.lower() or "embeddings" in result
    assert "what does the ML paper say" in result or "ML paper" in result


def test_unknown_features_ignored():
    result = build_tools_prompt(["nonexistent", "also_fake"])
    assert result == ""


def test_unknown_mixed_with_known():
    result = build_tools_prompt(["nonexistent", "search"])
    assert "search" in result
