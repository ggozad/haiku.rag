# haiku.rag app

A browser application over haiku.rag: chat over every configured database, the ingester's queue and jobs, an inspector for documents and chunks, and the doctor with a map of the corpus. A Starlette backend reads LanceDB, `haiku-ingester serve` writes it, and a Next.js frontend proxies both.

> **Note:** The application has no authentication. The compose files bind every service to `127.0.0.1`, and the frontend proxies the backend and the ingester, so don't expose any of them to an untrusted network.

```bash
cd app
cp .env.example .env                      # API keys, host paths, the ingester token
cp haiku.rag.yaml.example haiku.rag.yaml  # models, database at /data, ingester source /docs
docker compose up -d --build              # or docker-compose.dev.yml for hot reload
```

Open http://localhost:3000. Configuration, endpoints and development are covered in the [web application docs](https://ggozad.github.io/haiku.rag/apps/).
