"""
Conversation Summary Prompts
=============================
Generate per-exchange summaries for conversation context.

Referenced by: backend/app/services/conversation_summary_service.py

The service generates summaries to reduce context payload in long conversations.
Each summary includes: topic_label, key_entities, summary.

See: prompts/conversation_summary.md
"""

SUMMARIZER_PROMPT = """Bạn là assistant chuyên tóm tắt đoạn hội thoại Q&A.

TÓM TẮT đoạn exchange sau thành JSON:
- **topic_label** (tiếng Việt, 3-8 từ): Chủ đề chính của câu hỏi
- **key_entities** (list 3-5 items): Các entities quan trọng (con số, tên riêng, khái niệm)
- **summary** (2-4 câu tiếng Việt): Tóm tắt nội dung CHÍNH, KHÔNG bịa đặt

EXCHANGE:
User: {user_message}

Assistant: {assistant_message}

Trả lời JSON format, không giải thích thêm:
{{"topic_label": "...", "key_entities": ["...", "..."], "summary": "..."}}"""