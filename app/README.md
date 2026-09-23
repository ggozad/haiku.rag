# haiku.rag Chat App

A browser chat over a haiku.rag database: a Starlette backend adapting the RAG capability with Pydantic AI's `AGUIAdapter`, and a Next.js / CopilotKit frontend.

> **Note:** An illustrative example meant as a starting point, with no authentication. The compose files bind the frontend and backend to `127.0.0.1`. The frontend proxies every backend route, so don't expose either to an untrusted network.

```bash
cp .env.example .env                      # API keys and the database path
cp haiku.rag.yaml.example haiku.rag.yaml  # models; the database is at /data
docker compose up -d --build              # or docker-compose.dev.yml for hot reload
```

Open http://localhost:3000. Configuration, endpoints and development are covered in the [web application docs](https://ggozad.github.io/haiku.rag/apps/).
