# Contributing to haiku.rag

Thanks for your interest. Bug reports, fixes, documentation and features are all welcome.

## Reporting issues

Open an issue at https://github.com/ggozad/haiku.rag/issues. Describe what you did, what happened and what you expected. Include the `haiku-rag` version and the relevant configuration.

## Before you start

For anything beyond a small fix, open an issue first so we can agree on the approach before you invest time in it.

## Development setup

```bash
git clone https://github.com/ggozad/haiku.rag.git
cd haiku.rag
uv sync
uv run pre-commit install
```

## Making changes

1. Create a branch from `main`.
2. Make your change, with tests. Coverage is enforced at 100%.
3. Run the checks:

   ```bash
   uv run pytest
   uv run ruff check && uv run ruff format
   uv run ty check
   ```

4. Add an entry under `[Unreleased]` in `CHANGELOG.md`.
5. Update `docs/` and `README.md` if you changed user-facing behaviour.
6. Open a pull request against `main`. Describe what changed and why.

Tests that need external services are marked `integration` and are skipped in CI. Run them locally with `docker compose -f tests/docker/docker-compose.yml up -d`. The [Development guide](https://ggozad.github.io/haiku.rag/development/) covers fixtures, test markers and recording HTTP cassettes.

## AI-assisted contributions

When haiku.rag started, I was not using AI assistance for it. Since the beginning of 2026, I use AI tools myself and most of the code is now written with them. I never vibe-code, maintain the general design and aim to review and monitor everything they produce.

Using AI tools to write code, issues or pull requests is fine. What matters is that a human (you) read the result before I do. I review everything myself, so I need to understand the problem and the solution from what you send.

- Open an issue before the pull request. If you plan to use AI for the implementation, describe your plan in the issue first.
- Read, check and trim whatever the tool produced. Remove anything you cannot explain or that does not serve the change.
- Keep issues and pull request descriptions short and concrete. Say what is wrong, how you fixed it and how you verified it.
- Say that AI was used. It is not held against the contribution.

## License

By contributing you agree that your contributions are licensed under the MIT License.
