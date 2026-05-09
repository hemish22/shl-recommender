# SHL Assessment Recommender — Approach Document

**Candidate:** Hemish Jain  
**Live API:** https://shl-recommender-ghed.onrender.com  
**Repository:** https://github.com/hemish22/shl-recommender

---

## 1. Problem Understanding

The task requires a conversational agent that takes vague hiring intent from a user and, through dialogue, recommends relevant SHL Individual Test Solutions assessments. The system must expose a stateless REST API (`POST /chat`) that accepts full conversation history and returns structured JSON with recommendations, a conversational reply, and a conversation-end signal. Scoring is on three axes: Recall@10 (are the right assessments surfaced?), conversational behavior (clarify, refine, compare, refuse), and API correctness.

---

## 2. Data Collection

**Scraper (`scraper.py`):** Paginated the SHL product catalog at `https://www.shl.com/solutions/products/product-catalog/?start=N&type=1` (12 items/page, 32 pages) using `requests` + `BeautifulSoup`. Filtered to the "Individual Test Solutions" table only. For each listing item, concurrently fetched detail pages (8 threads via `ThreadPoolExecutor`) to extract: full description, job levels, remote testing flag, and adaptive/IRT flag. Output: `catalog.json` with 377 unique assessments.

Each item captures:
- `name`, `url` (canonical `/solutions/products/product-catalog/view/...`)
- `test_types`: list of SHL type codes (A, B, C, D, E, K, P, S)
- `job_levels`, `remote_testing`, `adaptive`, `description`

---

## 3. Vector Index

**Embedder (`embedder.py`):** Custom ONNX-based embedder using `onnxruntime` + `tokenizers`. Loads the `all-MiniLM-L6-v2` model in ONNX format (86MB, committed to repo). Performs attention-mask-weighted mean pooling and L2 normalization — producing 384-dimensional vectors compatible with cosine similarity. No PyTorch dependency; total runtime memory ~150MB (fits Render's 512MB free tier).

**Index (`build_index.py`):** Encodes all 377 catalog documents into dense vectors. Each document concatenates: name, description, test types, job levels, remote/adaptive flags. Stored in a FAISS `IndexFlatIP` (inner product = cosine similarity on normalized vectors). Outputs `faiss.index` and `index_meta.json`.

---

## 4. Retrieval Strategy

**Multi-query retrieval (`retriever.py`):** At query time, two searches run in parallel — one over the last three user messages (recent context), one over the first user message (original role definition). Results are deduplicated by URL, sorted by score, and merged into a candidate pool of 20. The LLM then selects the top 1–10 from this pool.

**Test-type filtering:** Keyword-to-code mapping (e.g., "personality" → P, "coding" → K) detects when the user specifies an assessment type. Post-search filter applied; falls back to unfiltered if no results pass. This improves precision without sacrificing recall.

**Compare mode:** If the conversation contains compare triggers ("vs", "difference between"), a dedicated search over the full conversation text fetches named assessments specifically, ensuring both sides of a comparison are surfaced.

---

## 5. LLM Agent

**Model:** `llama-3.3-70b-versatile` via Groq API (low latency, generous free quota). Structured output enforced with `response_format={"type": "json_object"}`.

**System prompt design:** The prompt encodes four behavioral modes — CLARIFY, RECOMMEND, REFINE, COMPARE — with explicit rules:
- Clarify only on turn 1 if the query has zero role/domain context.
- Recommend as soon as a role or domain is known — even on partial info.
- After 2+ user messages, always recommend regardless of missing details.
- Max one clarifying question across the entire conversation.

This aggressive recommendation posture directly optimizes Recall@10: the model surfaces assessments early rather than over-clarifying.

**Catalog injection:** At each turn, the top 20 retrieved assessments are injected into the prompt as structured context. The model is constrained to recommend only from this context; all URLs are validated against a whitelist of scraped catalog URLs before being returned.

**Turn cap:** Conversations hard-cap at 8 turns. If the LLM fails to recommend by the final turn, the top 5 retrieval results are injected directly as a fallback.

---

## 6. API Design

`POST /chat` accepts a full stateless message history. The server holds no session state — clients own the conversation. Response schema:

```json
{
  "reply": "string",
  "recommendations": [
    {"name": "string", "url": "string", "test_type": "single letter"}
  ],
  "end_of_conversation": false
}
```

`test_type` is normalized to a single SHL type code (first valid letter from A/B/C/D/E/K/P/S) regardless of what the LLM returns. `GET /health` returns `{"status": "ok"}` and is used for uptime monitoring.

---

## 7. Security

- **Prompt injection detection:** 15 regex patterns block injection attempts ("ignore previous instructions", "act as", "reveal your prompt", etc.) before the LLM is called. Returns HTTP 400.
- **Input sanitization:** Unicode control characters stripped from all messages before processing.
- **Rate limiting:** `slowapi` enforces 10 requests/minute per IP (HTTP 429 on breach).
- **API key auth:** Optional `X-API-Key` header enforced when `API_KEY` env var is set.
- **Message validation:** Max 20 messages per request, max 2000 characters per message, blank messages rejected.
- **URL whitelist:** All recommendation URLs validated against the scraped catalog before returning to client.

---

## 8. Deployment

Deployed on Render (free tier) at `https://shl-recommender-ghed.onrender.com`.

- **Build command:** `pip install -r requirements.txt`
- **Start command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
- **Startup:** Embedder model loads in a background thread so uvicorn binds the port immediately (Render port-scan compatible). FAISS index and catalog metadata load alongside.
- **Keepalive:** GitHub Actions cron workflow pings `/health` every 10 minutes, preventing Render's free-tier 15-minute spin-down. All runs confirmed successful.
- **Environment:** `GROQ_API_KEY` set as Render environment variable. No secrets in repository.

---

## 9. Design Decisions & Tradeoffs

| Decision | Alternative | Reason chosen |
|----------|-------------|---------------|
| ONNX embedder (no PyTorch) | sentence-transformers | PyTorch = 2GB RAM, OOM on 512MB free tier |
| Groq `llama-3.3-70b` | GPT-4, Gemini | Free tier, low latency, JSON mode |
| FAISS `IndexFlatIP` | Approximate indexes (HNSW) | 377 items — exact search fast enough, no recall loss |
| Stateless API | Server-side sessions | Matches assignment spec; simpler scaling |
| Multi-query retrieval | Single query | Increases Recall@10 by covering role + skills separately |
| Candidate pool of 20 → LLM picks ≤10 | Return top 10 directly | LLM re-ranks by relevance to full conversation context |
| Aggressive recommend posture | Conservative clarify | Recall@10 metric rewards surfacing good assessments early |
