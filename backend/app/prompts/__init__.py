"""
Prompts Index
==============
Central export for all prompts in the codebase.
Import from this module instead of scattered prompt definitions.

Structure:
  __init__.py          - This file - exports all prompts
  legal_kg.md          - LegalKG prompts (entity extraction, person, preamble, resolution)
  chat.md              - Chat system prompts (default + hard system)
  document_classifier.md  - Document type classifier prompt
  agent_intent.md      - Intent classification prompt
  write_agent.md       - Write agent prompts (summarize, edit, grammar, format)
  conversation_summary.md  - Exchange summarization prompt
  tool_system.md       - LLM provider tool calling prompts
  agents/
    supervisor.md     - Supervisor agent prompt (intent classification + routing)
    rag_agent.md       - RAG agent tool descriptions
    write_agent.md     - Write agent prompts

  legal_kg.py          - LegalKG prompt implementations
  chat.py              - Chat prompt implementations
  document_classifier.py  - Document classifier prompt implementation
  agent_intent.py      - Intent classifier prompt implementation
  write_agent.py       - Write agent prompt implementations
  conversation_summary.py  - Conversation summary prompt implementation
  tool_system.py       - Tool system prompt implementations
  agents/
    __init__.py        - Agent prompts exports
    supervisor_prompt.py - Supervisor agent prompt
    write_agent_prompt.py - Write agent prompts
"""

from app.prompts.legal_kg import (
    LEGAL_KG_SYSTEM_PROMPT,
    LEGAL_KG_USER_PROMPT,
    PERSON_EXTRACT_SYSTEM_PROMPT,
    PERSON_EXTRACT_USER_PROMPT,
    PREAMBLE_SYSTEM_PROMPT,
    PREAMBLE_USER_PROMPT,
    ENTITY_RESOLVE_SYSTEM_PROMPT,
    ENTITY_RESOLVE_USER_PROMPT,
    PERSON_DOCUMENT_TRIGGERS,
)

from app.prompts.chat import (
    DEFAULT_SYSTEM_PROMPT,
    HARD_SYSTEM_PROMPT,
)

from app.prompts.document_classifier import (
    _build_llm_system_prompt,
)

from app.prompts.agent_intent import (
    _CLASSIFIER_SYSTEM,
    _VALID_INTENTS,
)

from app.prompts.write_agent import (
    WRITE_PROMPTS,
    FORMAT_CHECK_PROMPT,
)

from app.prompts.conversation_summary import (
    SUMMARIZER_PROMPT,
)

from app.prompts.tool_system import (
    OLLAMA_TOOL_SYSTEM,
    OLLAMA_TOOL_REMINDER,
    GEMINI_TOOL_SYSTEM,
    OPENAI_COMPATIBLE_TOOL_SYSTEM,
    NATIVE_TOOL_REMINDER,
)

from app.prompts.agents.supervisor_prompt import _SUPERVISOR_PROMPT

__all__ = [
    # legal_kg
    "LEGAL_KG_SYSTEM_PROMPT",
    "LEGAL_KG_USER_PROMPT",
    "PERSON_EXTRACT_SYSTEM_PROMPT",
    "PERSON_EXTRACT_USER_PROMPT",
    "PREAMBLE_SYSTEM_PROMPT",
    "PREAMBLE_USER_PROMPT",
    "ENTITY_RESOLVE_SYSTEM_PROMPT",
    "ENTITY_RESOLVE_USER_PROMPT",
    "PERSON_DOCUMENT_TRIGGERS",
    # chat
    "DEFAULT_SYSTEM_PROMPT",
    "HARD_SYSTEM_PROMPT",
    # document_classifier
    "_build_llm_system_prompt",
    # agent_intent
    "_CLASSIFIER_SYSTEM",
    "_VALID_INTENTS",
    # write_agent
    "WRITE_PROMPTS",
    "FORMAT_CHECK_PROMPT",
    # conversation_summary
    "SUMMARIZER_PROMPT",
    # tool_system
    "OLLAMA_TOOL_SYSTEM",
    "OLLAMA_TOOL_REMINDER",
    "GEMINI_TOOL_SYSTEM",
    "OPENAI_COMPATIBLE_TOOL_SYSTEM",
    "NATIVE_TOOL_REMINDER",
    # agents
    "_SUPERVISOR_PROMPT",
]