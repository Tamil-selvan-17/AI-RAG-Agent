# Free Cloud Deployment Guide

This guide walks you through hosting the entire AI RAG Agent **for free**, with nothing running on your own PC. It replaces the three local pieces (Ollama, Qdrant, Redis) with free cloud equivalents, then deploys the backend (which also serves the frontend) to a free web host.

| Local piece | Free cloud replacement | Free tier limits |
|---|---|---|
| Ollama (`qwen2.5:7b` + `bge-m3`) | **Google Gemini API** | Generous free daily request quota, no credit card |
| Qdrant (Docker) | **Qdrant Cloud** | 1GB cluster, free forever |
| Redis (Docker) | **Upstash Redis** | Free tier, generous daily command limit |
| Your machine running `uvicorn` | **Render.com** free web service | Sleeps after 15 min idle, wakes on next request (~30-50s cold start) |

No code changes are needed from you — this project already has a built-in `AI_PROVIDER` switch that flips between local Ollama and Gemini, and both Qdrant/Redis services auto-detect whether to connect locally or to a cloud URL. You just need to sign up for the free accounts and set environment variables.

---

## Step 1 — Get a free Gemini API key

1. Go to **[aistudio.google.com/apikey](https://aistudio.google.com/apikey)**
2. Sign in with any Google account
3. Click **"Create API key"**
4. Copy the key — you'll paste it into Render's environment variables in Step 5

This one key covers both the chat model (`gemini-2.0-flash`) and the embedding model (`text-embedding-004`).

---

## Step 2 — Create a free Qdrant Cloud cluster

1. Go to **[cloud.qdrant.io](https://cloud.qdrant.io)** and sign up (free)
2. Click **"Create Cluster"** → choose the **Free tier** (1GB)
3. Once it's created, open the cluster and copy two things:
   - The **Cluster URL** (looks like `https://xxxxxxx.us-east.aws.cloud.qdrant.io:6333`)
   - An **API Key** (create one from the cluster's "API Keys" tab)

Keep both — you'll need them in Step 5.

---

## Step 3 — Create a free Upstash Redis database (optional — skip if you'd rather not)

You can skip this step entirely if you'd rather not sign up for one more service. Just set `MEMORY_BACKEND=memory` in Render's environment variables instead (Step 5) — everything will run with zero Redis dependency. The trade-off: conversation history, the document list, and the answer cache all live only in the app's RAM, so they reset whenever the free instance restarts or wakes from sleep. Your uploaded documents' actual *knowledge* isn't affected, since that lives in Qdrant Cloud regardless.

If you'd rather have that memory persist across restarts, set it up:

1. Go to **[upstash.com](https://upstash.com)** and sign up (free)
2. Click **"Create Database"** → choose the **Free** plan → pick any region
3. Once created, open the database and copy the **Redis connection URL** — it looks like:
   ```
   rediss://default:AbCdEf123...@abc-def-12345.upstash.io:6379
   ```
   (Note the `rediss://` with two S's — that means TLS/SSL, which Upstash requires.)

---

## Step 4 — Push your project to GitHub

Render deploys directly from a GitHub repo.

1. Create a new **empty** repository on [github.com](https://github.com) (e.g. `ai-rag-agent`)
2. From your project folder:
   ```bash
   cd "AI-RAG-Agent"
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/ai-rag-agent.git
   git push -u origin main
   ```

Your `.gitignore` already excludes `venv/`, `.env.local`, logs, and uploaded documents, so nothing sensitive or bulky gets pushed. Note: your real `backend/.env` file with actual secrets **will** get pushed as-is right now since it only has placeholders — you'll set the real secrets as Render environment variables instead (Step 5), not by editing `.env` in the repo.

---

## Step 5 — Deploy the backend on Render

1. Go to **[render.com](https://render.com)** and sign up (free, no credit card needed for the free tier)
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub account and select the `ai-rag-agent` repo
4. Fill in the settings:
   | Field | Value |
   |---|---|
   | **Name** | anything, e.g. `ai-rag-agent` |
   | **Root Directory** | `backend` |
   | **Runtime** | `Python 3` |
   | **Build Command** | `pip install -r requirements.txt` |
   | **Start Command** | `uvicorn app.main:app --host 0.0.0.0 --port $PORT` |
   | **Instance Type** | `Free` |

5. Before clicking "Create Web Service," scroll to **Environment Variables** and add these (this is where your real secrets go — never commit them to `.env` in the repo):

   ```
   AI_PROVIDER=gemini
   GEMINI_API_KEY=<your key from Step 1>
   GEMINI_CHAT_MODEL=gemini-2.0-flash
   GEMINI_EMBED_MODEL=text-embedding-004
   GEMINI_EMBED_DIMENSIONS=768

   QDRANT_URL=<your cluster URL from Step 2>
   QDRANT_API_KEY=<your API key from Step 2>
   QDRANT_COLLECTION_NAME=documents

   REDIS_URL=<your connection URL from Step 3, or leave this OUT entirely if you skipped Step 3>
   MEMORY_BACKEND=redis
   # ^ If you skipped Step 3, delete REDIS_URL above and instead set:
   # MEMORY_BACKEND=memory

   UPLOAD_DIR=/opt/render/project/src/documents
   CORS_ORIGINS=*
   LOG_LEVEL=INFO
   ```

   Leave `QDRANT_HOST`/`QDRANT_PORT`/`REDIS_HOST`/`REDIS_PORT` unset — they're ignored automatically once `QDRANT_URL`/`REDIS_URL` are set.

6. Click **"Create Web Service"**. Render will build and deploy — this takes a few minutes the first time.

7. Once it says **"Live"**, your app is running at a URL like:
   ```
   https://ai-rag-agent.onrender.com
   ```
   Open `https://ai-rag-agent.onrender.com/app` in your browser — that's your hosted assistant.

---

## Step 6 — Verify it's working

Check the health endpoint:
```
https://ai-rag-agent.onrender.com/health
```
You want (if using Upstash):
```json
{"status":"healthy","services":{"api":"healthy","gemini":"healthy","qdrant":"healthy","redis":"healthy"}}
```
Or, if you set `MEMORY_BACKEND=memory` instead of using Upstash:
```json
{"status":"healthy","services":{"api":"healthy","gemini":"healthy","qdrant":"healthy","memory (in-process, non-persistent)":"healthy"}}
```

If something shows `unreachable`, double check the matching environment variable in Render's dashboard (Settings → Environment) and redeploy.

Then open `/app`, upload a document, and ask it a question — same as running locally, just hosted for free in the cloud.

---

## Important free-tier caveats

- **Render free tier sleeps after ~15 minutes of no traffic.** The first request after it wakes up takes 30-50 seconds while it restarts — this is normal, not a bug.
- **Render's free-tier disk is ephemeral.** Every time the service restarts or redeploys, anything written to `/opt/render/project/src/documents` is wiped. This doesn't break the assistant's ability to *answer questions* about previously uploaded documents (that data lives safely in Qdrant Cloud + Upstash, which persist), but you won't be able to re-download the original uploaded files after a restart. If you need the raw files to survive restarts too, you'd need paid persistent disk (Render) or an object storage bucket — not required for the assistant to keep working.
- **Gemini, Qdrant Cloud, and Upstash free tiers all have daily/monthly limits.** For personal or small-team use they're generous enough that you're unlikely to hit them, but very heavy usage could require upgrading one of them later.
- **Switching back to local Ollama at any time:** just set `AI_PROVIDER=ollama` again (and unset `QDRANT_URL`/`REDIS_URL`) — nothing else in the code needs to change, since the project supports both side by side.
