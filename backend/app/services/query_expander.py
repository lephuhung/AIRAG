"""
Query Expander for Vietnamese Legal Text
=========================================

Expands common Vietnamese legal terms and phrases to improve retrieval recall.
Patterns are based on common legal phrasing in Vietnamese administrative documents.

Usage:
    expanded_query = expand_legal_terms(query)
"""
from __future__ import annotations

import re

# Legal term expansion patterns: (pattern, replacement)
# Order matters - more specific patterns should come first
_LEGAL_TERM_PATTERNS: list[tuple[str, str]] = [
    # Căn cứ expansions
    (r"\bcăn cứ\b", "căn cứ theo quy định"),
    (r"\bcăn cứ\s+vào\b", "căn cứ theo"),
    (r"\bcăn cứ\s+pháp luật\b", "căn cứ theo quy định của pháp luật"),

    # Quy định expansions
    (r"\btheo quy định\b", "theo quy định của pháp luật"),
    (r"\bquy định\s+tại\b", "quy định của"),
    (r"\bquy định\s+chi tiết\b", "quy định chi tiết và hướng dẫn"),

    # Hiệu lực
    (r"\bcó hiệu lực\b", "có hiệu lực pháp luật"),
    (r"\bcó\s+hiệu\s+lực\s+từ\b", "có hiệu lực thi hành từ"),
    (r"\bhết\s+hiệu\s+lực\b", "hết hiệu lực thi hành"),

    # Trình tự thủ tục
    (r"\bthủ\s+tục\s+hành\s+chính\b", "thủ tục hành chính"),
    (r"\btrình\s+tự\s+thủ\s+tục\b", "trình tự thủ tục"),
    (r"\bthẩm\s+quyền\b", "thẩm quyền giải quyết"),

    # Hình thức xử lý
    (r"\bxử\s+lý\s+vi\s+phạm\b", "xử lý vi phạm hành chính"),
    (r"\bphạt\s+tiền\b", "phạt tiền vi phạm hành chính"),
    (r"\bcảnh\s+cáo\b", "cảnh cáo vi phạm hành chính"),

    # Tổ chức thực hiện
    (r"\bchủ\s+trì\b", "chủ trì thực hiện"),
    (r"\bphối\s+hợp\b", "phối hợp thực hiện"),
    (r"\bchịu\s+trách\s+nhiệm\b", "chịu trách nhiệm thực hiện"),

    # Thời hạn
    (r"\btrong\s+thời\s+hạn\b", "trong thời hạn quy định"),
    (r"\bquá\s+thời\s+hạn\b", "quá thời hạn quy định"),

    # Hồ sơ
    (r"\bhồ\s+sơ\s+thủ\s+tục\b", "hồ sơ thủ tục hành chính"),
    (r"\bnộp\s+hồ\s+sơ\b", "nộp hồ sơ tại"),
]


def expand_legal_terms(query: str) -> str:
    """
    Expand common Vietnamese legal terms in a query string.

    Args:
        query: The original query string.

    Returns:
        The query with legal terms expanded for better retrieval.
    """
    if not query or len(query.strip()) < 2:
        return query

    expanded = query
    for pattern, replacement in _LEGAL_TERM_PATTERNS:
        expanded = re.sub(pattern, replacement, expanded, flags=re.IGNORECASE)

    return expanded
