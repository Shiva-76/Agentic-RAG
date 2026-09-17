"""
app.py
------
Streamlit chatbot UI for the Agentic RAG system.

Features:
  - Persistent conversation history in st.session_state
  - st.status() collapsible panel showing each node's activity live
  - Full node trace: rewriter → classifier → router → retrieval → synthesis → guard
  - Final answer displayed in chat bubble
  - Citations render as Markdown inline
"""

import streamlit as st
import logging
import sys
import os

# Make src importable when launched as `streamlit run src/ui/app.py`
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.agent.graph import run_turn

logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Agentic RAG — Gemini",
    page_icon="🔍",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Custom CSS — minimal dark polish
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .stChatMessage { border-radius: 12px; }
    .node-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 0.78em;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-rewriter   { background:#1e3a5f; color:#90caf9; }
    .badge-classifier { background:#1b3a1f; color:#a5d6a7; }
    .badge-router     { background:#3e2723; color:#ffcc80; }
    .badge-retrieve   { background:#1a237e; color:#c5cae9; }
    .badge-synthesize { background:#4a148c; color:#ce93d8; }
    .badge-guard      { background:#b71c1c; color:#ef9a9a; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []            # display messages (role, content)
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []  # Gemini Content format

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🔍 Agentic RAG")
st.caption("Gemini 3.6 Flash · Qdrant · LangGraph · 6-node pipeline with guardrails")

with st.expander("ℹ️ Architecture", expanded=False):
    st.markdown("""
    ```
    Query → QueryRewriter → NeedsRetrieval?
                                 │ No  → Synthesize
                                 │ Yes → SourceRouter → Retrieve → Synthesize
                            RelevanceGuard (guardrail, max 3 retries)
                                 │ Pass → Final answer
                                 │ Fail → back to QueryRewriter
    ```
    **Sources:** Private KB (Qdrant) · Web (DuckDuckGo) · Calculator (numexpr)
    """)

st.divider()

# ---------------------------------------------------------------------------
# Render existing messages
# ---------------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------
if prompt := st.chat_input("Ask anything…"):

    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # ── Run agent with live status panel ────────────────────────────────────
    with st.chat_message("assistant"):
        with st.status("🤔 Thinking…", expanded=True) as status:

            # Node 1 — QueryRewriter
            status.write("🔄 **Rewriting query…**")

            try:
                result = run_turn(
                    user_query=prompt,
                    conversation_history=list(st.session_state.conversation_history),
                )
            except Exception as exc:
                st.error(f"Agent error: {exc}")
                st.stop()

            # Render node traces from node_logs
            for log in result.get("node_logs", []):
                node = log.get("node", "unknown")

                if node == "query_rewriter":
                    st.write(
                        f"🔄 **Query Rewriter** — `{log.get('input', '')}` "
                        f"→ `{log.get('output', '')}`"
                        + (f"  _(retry {log['retry']})_" if log.get("retry") else "")
                    )

                elif node == "needs_retrieval":
                    icon = "✅" if log.get("needs_retrieval") else "💬"
                    st.write(
                        f"{icon} **Intent Classifier** — "
                        f"{'Retrieval needed' if log.get('needs_retrieval') else 'Direct answer'}: "
                        f"_{log.get('reason', '')}_"
                    )

                elif node == "source_router":
                    sources_str = " + ".join(log.get("chosen_sources", []))
                    st.write(
                        f"🗂️ **Source Router** → **{sources_str}**  "
                        f"_{log.get('reasoning', '')}_"
                    )

                elif node == "retrieve":
                    for tool in log.get("tools", []):
                        src = tool["source"].upper()
                        preview = tool.get("result_preview", "")[:150]
                        st.write(f"🔍 **{src}** retrieved {tool['result_length']} chars — `{preview}…`")

                elif node == "synthesize":
                    st.write(
                        f"✍️ **Synthesize** — drafted {log.get('draft_length', 0)} chars "
                        f"({log.get('thought_count', 0)} thought traces)"
                    )

                elif node == "relevance_guard":
                    icon = "🛡️✅" if log.get("pass") else "🛡️❌"
                    st.write(
                        f"{icon} **Relevance Guard** — "
                        f"{'PASS' if log.get('pass') else f'FAIL (retry {log.get(\"retry_count\",0)+1})'}: "
                        f"_{log.get('reason', '')}_"
                    )
                    if not log.get("pass") and log.get("hint"):
                        st.write(f"   💡 Hint: _{log.get('hint')}_")

            # Show thought traces if any
            thoughts = result.get("thought_parts", [])
            if thoughts:
                with st.expander("💭 Reasoning traces", expanded=False):
                    for i, t in enumerate(thoughts, 1):
                        st.markdown(f"**Trace {i}:** {t}")

            status.update(label="✅ Done", state="complete", expanded=False)

        # ── Final answer ────────────────────────────────────────────────────
        final = result.get("final_answer", "_No answer generated._")
        st.markdown(final)

    # ── Persist to session state ─────────────────────────────────────────────
    st.session_state.messages.append({"role": "assistant", "content": final})

    # Update conversation history with what the graph appended
    st.session_state.conversation_history = list(
        result.get("conversation_history", [])
    )
