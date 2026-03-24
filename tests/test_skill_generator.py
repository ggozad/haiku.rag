import pytest

from haiku.rag.skill_generator import (
    AVAILABLE_TOOLS,
    generate_skill,
    render_templates,
    validate_db_path,
    validate_metadata,
    validate_output_dir,
    validate_tools,
)


class TestAvailableTools:
    def test_all_tools_present(self):
        assert AVAILABLE_TOOLS == {
            "list_documents",
            "get_document",
            "search",
            "ask",
            "research",
            "analyze",
        }


class TestValidateMetadata:
    def test_valid(self):
        validate_metadata("my-recipes", "A skill.")

    def test_rejects_invalid_name(self):
        with pytest.raises(ValueError):
            validate_metadata("Bad_Name!", "A skill.")


class TestValidateTools:
    def test_valid_single_tool(self):
        validate_tools(["search"])

    def test_valid_multiple_tools(self):
        validate_tools(["list_documents", "get_document", "search", "ask"])

    def test_valid_all_tools(self):
        validate_tools(list(AVAILABLE_TOOLS))

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="at least one"):
            validate_tools([])

    def test_rejects_unknown_tool(self):
        with pytest.raises(ValueError, match="Unknown"):
            validate_tools(["search", "bogus"])


class TestValidateDbPath:
    def test_valid_path(self, tmp_path):
        db_path = tmp_path / "test.lancedb"
        db_path.mkdir()
        validate_db_path(db_path)

    def test_rejects_nonexistent(self, tmp_path):
        db_path = tmp_path / "nonexistent.lancedb"
        with pytest.raises(ValueError, match="does not exist"):
            validate_db_path(db_path)

    def test_rejects_file(self, tmp_path):
        db_path = tmp_path / "test.lancedb"
        db_path.touch()
        with pytest.raises(ValueError, match="not a directory"):
            validate_db_path(db_path)


class TestValidateOutputDir:
    def test_valid_output_dir(self, tmp_path):
        validate_output_dir(tmp_path, "recipes")

    def test_rejects_nonexistent(self, tmp_path):
        output_dir = tmp_path / "nonexistent"
        with pytest.raises(ValueError, match="does not exist"):
            validate_output_dir(output_dir, "recipes")

    def test_rejects_existing_target(self, tmp_path):
        target = tmp_path / "recipes-skill"
        target.mkdir()
        with pytest.raises(ValueError, match="already exists"):
            validate_output_dir(tmp_path, "recipes")


class TestRenderTemplates:
    def test_output_structure(self, tmp_path):
        result = render_templates(
            output_dir=tmp_path,
            name="recipes",
            description="A recipe skill.",
            tool_names=["list_documents", "get_document", "search", "ask"],
        )
        assert result == tmp_path / "recipes-skill"
        assert result.is_dir()
        pkg = result / "recipes_skill"
        assert (pkg / "__init__.py").is_file()
        assert (pkg / "SKILL.md").is_file()
        assert (pkg / "assets").is_dir()

    def test_dashed_name_uses_underscores_for_python(self, tmp_path):
        result = render_templates(
            output_dir=tmp_path,
            name="my-recipes",
            description="A recipe skill.",
            tool_names=["search", "ask"],
        )
        assert result == tmp_path / "my-recipes-skill"
        pkg = result / "my_recipes_skill"
        assert (pkg / "__init__.py").is_file()
        init = (pkg / "__init__.py").read_text()
        assert '"my-recipes.lancedb"' in init
        assert 'state_namespace="my-recipes"' in init
        toml = (result / "pyproject.toml").read_text()
        assert 'name = "my-recipes-skill"' in toml
        assert 'my-recipes = "my_recipes_skill:create_skill"' in toml

    def test_tool_names_list_matches_selection(self, tmp_path):
        render_templates(
            output_dir=tmp_path,
            name="docs",
            description="A docs skill.",
            tool_names=["search", "ask"],
        )
        init = tmp_path / "docs-skill" / "docs_skill" / "__init__.py"
        content = init.read_text()
        assert '["search", "ask"]' in content

    def test_create_skill_tools_called(self, tmp_path):
        render_templates(
            output_dir=tmp_path,
            name="docs",
            description="A docs skill.",
            tool_names=list(AVAILABLE_TOOLS),
        )
        init = tmp_path / "docs-skill" / "docs_skill" / "__init__.py"
        content = init.read_text()
        assert (
            "create_skill_tools(_DB_PATH, config, SkillState, _TOOL_NAMES)" in content
        )

    def test_tool_names_in_init(self, tmp_path):
        render_templates(
            output_dir=tmp_path,
            name="recipes",
            description="A recipe skill.",
            tool_names=["search", "ask"],
        )
        init = tmp_path / "recipes-skill" / "recipes_skill" / "__init__.py"
        content = init.read_text()
        assert '"search"' in content
        assert '"ask"' in content

    def test_pyproject_toml(self, tmp_path):
        render_templates(
            output_dir=tmp_path,
            name="recipes",
            description="A recipe skill.",
            tool_names=["search"],
        )
        toml = tmp_path / "recipes-skill" / "pyproject.toml"
        content = toml.read_text()
        assert 'name = "recipes-skill"' in content
        assert 'description = "A recipe skill."' in content
        assert 'recipes = "recipes_skill:create_skill"' in content
        assert "haiku.rag-slim >= " in content

    def test_skill_md_conditionals(self, tmp_path):
        render_templates(
            output_dir=tmp_path,
            name="docs",
            description="A docs skill.",
            tool_names=["search"],
        )
        skill_md = tmp_path / "docs-skill" / "docs_skill" / "SKILL.md"
        content = skill_md.read_text()
        assert "**search**" in content
        assert "**ask**" not in content
        assert "**list_documents**" not in content
        assert "**research**" not in content
        assert "**analyze**" not in content

    def test_skill_md_includes_all_selected(self, tmp_path):
        render_templates(
            output_dir=tmp_path,
            name="docs",
            description="A docs skill.",
            tool_names=["search", "ask", "analyze"],
        )
        skill_md = tmp_path / "docs-skill" / "docs_skill" / "SKILL.md"
        content = skill_md.read_text()
        assert "**search**" in content
        assert "**ask**" in content
        assert "**analyze**" in content

    def test_custom_preamble(self, tmp_path):
        render_templates(
            output_dir=tmp_path,
            name="docs",
            description="A docs skill.",
            tool_names=["search"],
            preamble="You are a docs expert.",
        )
        skill_md = tmp_path / "docs-skill" / "docs_skill" / "SKILL.md"
        content = skill_md.read_text()
        assert "You are a docs expert." in content

    def test_state_namespace_is_skill_name(self, tmp_path):
        render_templates(
            output_dir=tmp_path,
            name="recipes",
            description="A recipe skill.",
            tool_names=["search"],
        )
        init = tmp_path / "recipes-skill" / "recipes_skill" / "__init__.py"
        content = init.read_text()
        assert 'state_namespace="recipes"' in content

    def test_analyze_state_fields(self, tmp_path):
        render_templates(
            output_dir=tmp_path,
            name="docs",
            description="A docs skill.",
            tool_names=["search", "analyze"],
        )
        init = tmp_path / "docs-skill" / "docs_skill" / "__init__.py"
        content = init.read_text()
        assert "analyses" in content

    def test_imports_from_shared_tools(self, tmp_path):
        render_templates(
            output_dir=tmp_path,
            name="recipes",
            description="A recipe skill.",
            tool_names=["search", "ask", "analyze"],
        )
        init = tmp_path / "recipes-skill" / "recipes_skill" / "__init__.py"
        content = init.read_text()
        assert "from haiku.rag.skills._tools import create_skill_tools" in content

    def test_generated_python_is_valid(self, tmp_path):
        render_templates(
            output_dir=tmp_path,
            name="recipes",
            description="A recipe skill.",
            tool_names=list(AVAILABLE_TOOLS),
        )
        pkg = tmp_path / "recipes-skill" / "recipes_skill"
        for py_file in pkg.glob("*.py"):
            source = py_file.read_text()
            compile(source, str(py_file), "exec")


def _make_fake_lancedb(path):
    path.mkdir()
    (path / "data.lance").touch()
    return path


class TestGenerateSkill:
    def test_end_to_end(self, tmp_path):
        db_path = _make_fake_lancedb(tmp_path / "test.lancedb")
        result = generate_skill(
            db_path=db_path,
            output_dir=tmp_path,
            name="recipes",
            description="A recipe skill.",
            tool_names=["search", "ask"],
        )
        assert result == tmp_path / "recipes-skill"
        assets = result / "recipes_skill" / "assets"
        assert (assets / "recipes.lancedb").is_dir()
        assert (assets / "recipes.lancedb" / "data.lance").is_file()
        assert not (assets / "haiku.rag.yaml").exists()

    def test_with_config(self, tmp_path):
        db_path = _make_fake_lancedb(tmp_path / "test.lancedb")
        config_file = tmp_path / "haiku.rag.yaml"
        config_file.write_text("storage:\n  data_dir: /tmp\n")
        result = generate_skill(
            db_path=db_path,
            output_dir=tmp_path,
            name="recipes",
            description="A recipe skill.",
            tool_names=["search"],
            config_path=config_file,
        )
        assets = result / "recipes_skill" / "assets"
        assert (assets / "haiku.rag.yaml").is_file()
        assert (assets / "haiku.rag.yaml").read_text() == (
            "storage:\n  data_dir: /tmp\n"
        )

    def test_rejects_invalid_name(self, tmp_path):
        db_path = _make_fake_lancedb(tmp_path / "test.lancedb")
        with pytest.raises(ValueError):
            generate_skill(
                db_path=db_path,
                output_dir=tmp_path,
                name="Bad_Name!",
                description="A skill.",
                tool_names=["search"],
            )

    def test_rejects_invalid_tools(self, tmp_path):
        db_path = _make_fake_lancedb(tmp_path / "test.lancedb")
        with pytest.raises(ValueError, match="Unknown"):
            generate_skill(
                db_path=db_path,
                output_dir=tmp_path,
                name="recipes",
                description="A skill.",
                tool_names=["bogus"],
            )

    def test_rejects_nonexistent_db(self, tmp_path):
        with pytest.raises(ValueError, match="does not exist"):
            generate_skill(
                db_path=tmp_path / "nope.lancedb",
                output_dir=tmp_path,
                name="recipes",
                description="A skill.",
                tool_names=["search"],
            )

    def test_rejects_existing_target(self, tmp_path):
        db_path = _make_fake_lancedb(tmp_path / "test.lancedb")
        (tmp_path / "recipes-skill").mkdir()
        with pytest.raises(ValueError, match="already exists"):
            generate_skill(
                db_path=db_path,
                output_dir=tmp_path,
                name="recipes",
                description="A skill.",
                tool_names=["search"],
            )

    def test_with_preamble(self, tmp_path):
        db_path = _make_fake_lancedb(tmp_path / "test.lancedb")
        result = generate_skill(
            db_path=db_path,
            output_dir=tmp_path,
            name="recipes",
            description="A recipe skill.",
            tool_names=["search"],
            preamble="You are a recipe expert.",
        )
        skill_md = result / "recipes_skill" / "SKILL.md"
        content = skill_md.read_text()
        assert "You are a recipe expert." in content
