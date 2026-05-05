"""
Tool System Prompts
====================
LLM provider-specific prompts for enforcing tool calling behavior.

Referenced by: backend/app/api/chat_agent.py

These prompts reinforce tool calling rules for different LLM providers:
  - OLLAMA_TOOL_SYSTEM: For Ollama provider
  - GEMINI_TOOL_SYSTEM: For Gemini provider
  - OPENAI_COMPATIBLE_TOOL_SYSTEM: For OpenAI-compatible endpoints (vLLM, LM Studio)
  - NATIVE_TOOL_REMINDER: Additional reminder for native tool calling
  - OLLAMA_TOOL_REMINDER: Additional reminder for Ollama provider

See: prompts/tool_system.md
"""

# ---------------------------------------------------------------------------
# Ollama prompt-based tool calling — MANDATORY search before answering
# ---------------------------------------------------------------------------

OLLAMA_TOOL_SYSTEM = """\
## TOOLS

You have ONE tool: search_documents.

### Tool: search_documents
Call it by outputting EXACTLY:
<tool_call>{"name": "search_documents", "arguments": {"query": "<rewritten query>"}}</tool_call>

### ABSOLUTE RULES (violations are FATAL errors)

1. **Except for simple conversational messages, ALWAYS CALL search_documents FIRST.**
   Simple conversational messages that do NOT require a tool call:
   - Greetings: "hello", "xin chào", "hi", "hey", "good morning", etc.
   - Acknowledgements: "cảm ơn", "thank you", "thanks", "ok", "got it", etc.
   - Farewells: "bye", "goodbye", "tạm biệt", etc.
   For ALL other messages — questions, requests, factual queries, analysis — you MUST
   call search_documents before answering. Your knowledge is UNRELIABLE; only document
   sources are trustworthy. If you are unsure whether a message needs a search, SEARCH.

2. **Your ENTIRE first response to a searchable query must be ONLY the <tool_call> block.**
   No text before it. No text after it. No explanation. Just the tool call.

3. **Rewrite the query** to be specific and detailed.
   "doanh thu" → "doanh thu thuần, tổng doanh thu theo năm, tăng trưởng doanh thu"
   "AI model" → "AI model architecture, performance benchmarks, training details"

4. After receiving search results, answer using ONLY those sources with citations.
   Format: claim text[source_id]. Example: Doanh thu đạt 4.850 tỷ VNĐ[id12].
"""

OLLAMA_TOOL_REMINDER = (
    "\n\n[SYSTEM REMINDER] If this is a question or request, you MUST call search_documents FIRST. "
    'Output ONLY: <tool_call>{"name": "search_documents", "arguments": {"query": "..."}}</tool_call> '
    "Exception: simple greetings, thanks, or farewells do NOT require a tool call — respond directly. "
    "For everything else, searching is MANDATORY. "
    "When answering from search results, use the provided source IDs for citations (e.g., [id12]).\n"
)

# ---------------------------------------------------------------------------
# Gemini system prompt reinforcement — enforce tool calling for questions
# ---------------------------------------------------------------------------

GEMINI_TOOL_SYSTEM = """\

## Tool Usage (MANDATORY)

You have one tool: `search_documents`.

### search_documents
Searches the knowledge base for relevant document sections.

### ABSOLUTE RULES:
1. For ALL user questions, requests, factual queries, or analysis — you MUST call \
`search_documents` FIRST before answering. Even if the conversation history \
contains relevant information, you MUST search again to get fresh, accurate sources.
2. Only skip the tool call for simple conversational messages:
   - Greetings: "hello", "xin chào", "hi", "hey", "good morning", etc.
   - Acknowledgements: "cảm ơn", "thank you", "thanks", "ok", "got it", etc.
   - Farewells: "bye", "goodbye", "tạm biệt", etc.
3. Use the unique 4-character ID provided in the search results context (e.g., [id12]) \
for your citations. DO NOT use example IDs from these instructions unless they \
match the search results.
4. NEVER answer a question using information from previous turns without searching. \
Your previous answers may contain outdated or incomplete information.
5. NEVER reuse citation IDs from previous answers. Each answer must have its own \
fresh sources from a new search.
6. Rewrite the user's query to be specific and detailed for better retrieval.
"""

# ---------------------------------------------------------------------------
# OpenAI-compatible system prompt reinforcement
# ---------------------------------------------------------------------------

OPENAI_COMPATIBLE_TOOL_SYSTEM = """
## Tool Usage (MANDATORY)

You have access to the `search_documents` tool.
- You MUST use this tool to answer any questions about document content, specific data, or analysis.
- Do NOT rely on your internal knowledge or previous answers for document-related facts.
- Skipping the tool call for document-related questions is a failure to follow instructions.
- If you are unsure whether a search is needed, PERFORM THE SEARCH.
"""

NATIVE_TOOL_REMINDER = "\n\n[SYSTEM REMINDER] You MUST call the `search_documents` tool before answering this query. "