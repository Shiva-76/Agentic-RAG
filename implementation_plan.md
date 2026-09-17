# Upgraded Agentic RAG Architecture — Implementation Plan

## What Changes and Why

The original plan had a linear graph: `check_kb → check_web → finalize`.

The diagram introduces **4 new LLM-powered nodes** and **1 feedback loop (guardrail)**,
turning the pipeline into a self-correcting, disciplined agent:

```
Query
  → [1] QueryRewriter        LLM rewrites/clarifies the raw query
  → [2] NeedsRetrieval?      LLM decides: can I answer directly, or must I search?
        ↓ No (direct)        → Finalize (no tool calls)
        ↓ Yes
  → [3] SourceRouter         LLM picks: KB / Web / Tools&APIs (or combination)
  → [4] Retrieval            Executes the chosen source(s)
  → [5] Synthesize           LLM drafts a cited response from retrieved context
  → [6] RelevanceGuard       LLM evaluates: is this response actually relevant?
        ↓ Yes                → Final response
        ↓ No (retry ≤ 3x)   → back to QueryRewriter with failure context
```

---

## User Review Required

> [!IMPORTANT]
> **Retry loop limit:** The diagram loops "No" from relevance check back to query rewriting
> indefinitely. The plan caps this at **3 retries** to prevent infinite loops on free-tier.
> After 3 failed relevance checks the agent surfaces "I could not produce a relevant answer"
> rather than crashing. Confirm or adjust this cap.

> [!IMPORTANT]
> **"Needs retrieval?" node:** When the LLM says "No" (step 4 in diagram), it answers from
> memory/context alone — no tool calls. Per the strict system prompt, this should only fire
> for greetings and pure chit-chat. The relevance guardrail still applies even to direct
> answers. Confirm this matches your intent.

> [!WARNING]
> **Phase 3 tools must be implemented first** before this graph can be wired end-to-end.
> The plan below implements Phase 3 (tools) and the new graph simultaneously since they
> are tightly coupled.

---

## Proposed Graph — Full Node Map

```
START
  │
  ▼
┌─────────────────────┐
│  node_rewrite_query  │  LLM: "Rewrite this query to be precise and unambiguous"
│  (always runs first) │  → stores rewritten_query in state
└──────────┬──────────┘
           │
           ▼
┌──────────────────────────┐
│  node_needs_retrieval?    │  LLM: "Do I need external sources to answer this?"
│  (intent classifier)      │  → routes to "direct" or "route_source"
└──────┬──────────┬─────────┘
       │ No       │ Yes
       │          ▼
       │  ┌───────────────────┐
       │  │ node_route_source  │  LLM: "Which source(s) will help?"
       │  │                   │  → "kb" | "web" | "both" | "calculator"
       │  └───────┬───────────┘
       │          │
       │          ▼
       │  ┌───────────────────┐
       │  │ node_retrieve      │  executes chosen tools (KB, web, calc)
       │  └───────┬───────────┘
       │          │
       ▼          ▼
┌──────────────────────────┐
│  node_synthesize          │  LLM: drafts cited answer from context
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  node_relevance_guard     │  LLM evaluates: is the draft relevant & grounded?
│  (the guardrail)          │  → "pass" | "retry"
└──────┬──────────┬─────────┘
       │ pass     │ retry (max 3x → then fail gracefully)
       ▼          └──────────────► back to node_rewrite_query
┌──────────────┐
│  END          │  final_answer sent to UI
└──────────────┘
```

---

## Proposed Changes

### Phase 3 — Tool Implementation

---

#### [MODIFY] [knowledge_base.py](file:///c:/Projects/Agentic-Rag/src/tools/knowledge_base.py)
- Full `search_knowledge_base()` implementation
- Embed with `RETRIEVAL_QUERY`, query Qdrant, enforce `MIN_SIMILARITY=0.7` floor
- Return `"NO_RELEVANT_RESULTS"` if nothing clears the floor

#### [MODIFY] [web_search.py](file:///c:/Projects/Agentic-Rag/src/tools/web_search.py)
- Full `search_web()` implementation using `ddgs`
- Retry with backoff on failure

#### [MODIFY] [calculator.py](file:///c:/Projects/Agentic-Rag/src/tools/calculator.py)
- Already implemented ✅

---

### Phase 3+4 — Agent Nodes & Graph

---

#### [MODIFY] [graph.py](file:///c:/Projects/Agentic-Rag/src/agent/graph.py)

Updated `AgentState` adds:
- `rewritten_query: str` — output of QueryRewriter
- `needs_retrieval: bool` — output of NeedsRetrieval classifier  
- `chosen_sources: list[str]` — output of SourceRouter (`["kb"]`, `["web"]`, `["both"]`, etc.)
- `retrieval_context: str` — aggregated tool results
- `draft_answer: str` — output of Synthesize node
- `relevance_pass: bool` — output of RelevanceGuard
- `retry_count: int` — guardrail loop counter (cap at 3)
- `failure_context: str` — what went wrong on retry (fed back to QueryRewriter)

New nodes:
1. **`node_rewrite_query`** — LLM call, structured output: `{rewritten_query: str}`
2. **`node_needs_retrieval`** — LLM call, structured output: `{needs_retrieval: bool, reason: str}`
3. **`node_route_source`** — LLM call, structured output: `{sources: list[str]}`
4. **`node_retrieve`** — executes tools based on `chosen_sources`, no LLM call
5. **`node_synthesize`** — LLM call, produces cited `draft_answer`
6. **`node_relevance_guard`** — LLM call, structured output: `{pass: bool, reason: str}`

---

#### [NEW] [src/agent/nodes/](file:///c:/Projects/Agentic-Rag/src/agent/nodes/)
Split each node into its own file for modularity:
- `rewrite_query.py`
- `needs_retrieval.py`
- `route_source.py`
- `retrieve.py`
- `synthesize.py`
- `relevance_guard.py`

---

#### [MODIFY] [system_prompt.py](file:///c:/Projects/Agentic-Rag/src/agent/system_prompt.py)
Add node-specific prompts as separate constants:
- `REWRITE_PROMPT` — instruct the LLM to improve query precision
- `NEEDS_RETRIEVAL_PROMPT` — intent classification prompt
- `ROUTE_SOURCE_PROMPT` — source selection prompt
- `SYNTHESIZE_PROMPT` — the existing strict system prompt (citations, conflicts)
- `RELEVANCE_GUARD_PROMPT` — relevance evaluation rubric

---

#### [MODIFY] [gemini_client.py](file:///c:/Projects/Agentic-Rag/src/agent/gemini_client.py)
Add `call_llm_json()` helper — calls Gemini with `response_mime_type="application/json"`
for nodes that need structured output (rewriter, classifier, router, guard).

---

### Phase 5 — Streamlit UI

#### [MODIFY] [app.py](file:///c:/Projects/Agentic-Rag/src/ui/app.py)
Add visibility into each node in the `st.status()` panel:
- "🔄 Rewriting query..." → shows original vs rewritten
- "🤔 Checking if retrieval needed..." → shows yes/no + reason
- "🗂️ Routing to source..." → shows chosen source
- "🔍 Retrieving..." → shows tool results
- "✍️ Synthesizing answer..." → shows draft
- "🛡️ Checking relevance..." → shows pass/fail; if retry, shows why

---

## Verification Plan

### Automated
```
python tests/test_tools.py          # Phase 3: all tools
python tests/test_graph.py          # Phase 4: each node in isolation with mocks
```

### Manual — 6 test cases

| Test | Input | Expected path | Pass if |
|------|-------|---------------|---------|
| A | "Hi" | rewrite → no-retrieval → synthesize → guard=pass | 0 tool calls |
| B | "What does my doc say about CNN accuracy?" | rewrite → yes → kb → synthesize → guard=pass | KB result, cited |
| C | "What is today's Sensex?" | rewrite → yes → web → synthesize → guard=pass | Web result, cited |
| D | "What is 3.14 * 100?" | rewrite → yes → calculator → synthesize → guard=pass | Numeric answer |
| E | "Topic not in KB or web" | rewrite → yes → kb=empty → web=empty → synthesize → guard=fail → retry ×3 → graceful fail | "I couldn't find..." |
| F | "What color is the sky per my notes vs. the web?" | → kb(sky=green) + web → synthesize → conflict disclosed | Both citations shown |
