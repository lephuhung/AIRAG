"""
LegalKG extraction prompt tester
================================
Iterate on the entity/relation extraction prompt WITHOUT touching the repo
prompt file or rebuilding/restarting the kg-worker.

What it does
  1. Loads ONE real document (markdown from MinIO + meta from Postgres).
  2. Splits it into Điều chunks (same `split_articles` the worker uses).
  3. Runs the *editable* prompt below through the real KG LLM provider.
  4. Aggregates entities/relations and grades each entity GOOD / GENERIC / JUNK
     so you can see, run over run, whether a prompt tweak removes the noise we
     found in the graph (e.g. "Task" as Organization, "Các Cơ Quan… có liên
     quan", form placeholders "(tên đơn vị đề nghị)", date-suffix Document
     duplicates).

How to iterate
  - Edit SYSTEM_PROMPT / USER_TEMPLATE below (they start as a copy of the repo
    prompt). Re-run. Compare the GENERIC/JUNK counts. Nothing is written back to
    Neo4j and the worker is untouched.
  - When happy, port the winning text into app/prompts/legal_kg.py.

Run (inside the backend container — it has config + network to MinIO/DB/LLM):
  docker exec hrag-backend python -m scripts.test_extract_prompt \
      --workspace 3e05142e-f59d-4444-8c35-ea71798601eb [--doc <doc_id>] \
      [--articles 0]        # 0 = all; N = only first N (cost control)
      [--only 5]            # extract a single Điều by index, repeatable
      [--json out.json]     # also dump raw aggregated result
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import uuid
from collections import defaultdict

from sqlalchemy import select

from app.core.config import settings
from app.core.database import async_session_maker
from app.models.document import Document
from app.services.kg.legal_kg_service import _parse_llm_json, split_articles
from app.services.llm.openai_compatible import OpenAICompatibleLLMProvider
from app.services.llm.types import LLMMessage
from app.services.storage_service import StorageService

# Models under test — resolved from runtime env so they track the deployment.
#   memory  = small KG model the worker currently uses (qwen-memory @ :8088)
#   qwen36  = large model (Qwen3.6-35B @ 10.10.0.240:8000)
MODELS = {
    "memory": dict(
        base_url=settings.MEMORY_AGENT_BASE_URL,
        model=settings.MEMORY_AGENT_MODEL,
        api_key=settings.MEMORY_AGENT_API_KEY,
    ),
    "qwen36": dict(
        base_url=settings.OPENAI_COMPATIBLE_BASE_URL,
        model=settings.OPENAI_COMPATIBLE_MODEL,
        api_key=settings.OPENAI_COMPATIBLE_API_KEY,
    ),
}


async def call_model(provider, system_prompt: str, user_prompt: str) -> str:
    """One extraction call with simple rate-limit backoff (mirrors _call_llm)."""
    messages = [
        LLMMessage(role="system", content=system_prompt),
        LLMMessage(role="user", content=user_prompt),
    ]
    for attempt in range(4):
        try:
            return await provider.acomplete(messages, temperature=0.0, max_tokens=4096)
        except Exception as e:
            err = str(e).lower()
            is_rate = any(k in err for k in ("429", "rate", "quota", "resource_exhausted"))
            if is_rate and attempt < 3:
                await asyncio.sleep(2 ** attempt)
            else:
                raise
    return ""

# ===========================================================================
#  EDIT HERE — prompt under test (seeded from app/prompts/legal_kg.py)
#  Change freely, re-run, compare the quality summary. Repo prompt untouched.
# ===========================================================================
SYSTEM_PROMPT = """Bạn là chuyên gia phân tích văn bản hành chính/pháp luật Việt Nam.
Nhiệm vụ của bạn là trích xuất các thực thể (entities) và mối quan hệ (relations) từ một Điều/Khoản của văn bản được cung cấp.

## Các loại thực thể được phép (entity types):
- Article: Điều, Khoản, Điểm của văn bản hiện tại. Ví dụ: "Điều 5", "Khoản 2 Điều 3"
- Document: Văn bản pháp luật được viện dẫn. Ví dụ: "Nghị định 123/2024/NĐ-CP".
  • Dùng tên ĐỊNH DANH NGẮN GỌN NHẤT. Nếu có số hiệu → ưu tiên kèm số hiệu.
  • TUYỆT ĐỐI KHÔNG thêm hậu tố ngày/tháng/năm ban hành vào tên.
    ĐÚNG: "Luật An ninh mạng"  —  SAI: "Luật An ninh mạng ngày 12 tháng 6 năm 2018".
- Organization: Cơ quan, tổ chức CỤ THỂ, có tên riêng định danh được. PHẢI bổ sung tên đầy đủ dựa vào (issuing_agency). Ví dụ: "UBND Tỉnh Nghệ An" (không dùng "UBND tỉnh").
  • KHÔNG trích xuất tổ chức CHUNG CHUNG / không định danh được. BỎ QUA hoàn toàn các trường hợp:
    - tên trống nghĩa: "Bộ", "Bộ trưởng", "Cơ quan", "Đơn vị", "Doanh nghiệp", "Tổ chức".
    - loại cơ quan: "Cơ quan ngang Bộ", "Cơ quan thuộc Chính phủ", "Cơ quan nhà nước", "Tổ chức chính trị".
    - nhóm/liệt kê: "các cơ quan, tổ chức, cá nhân có liên quan", "Bộ, ngành có liên quan", "các cơ quan Đảng, Nhà nước ở trung ương", "tổ chức, cá nhân có liên quan".
    - placeholder biểu mẫu: bất cứ tên nào chứa "(tên đơn vị...)", "(tên cơ quan...)", "(chủ quản...)".
    - tên biểu mẫu/văn bản con: "Mẫu số 02", "Tờ trình", "Đơn đề nghị".
- Person: Cá nhân. PHẢI dùng Composite Key theo quy tắc ưu tiên (xem bên dưới). Các cá nhân chung chung như "Người có trách nhiệm", "Người liên quan" không được coi là Person
- Task: Nhiệm vụ/công việc cụ thể được giao. HÃY trích xuất MỖI nhiệm vụ chính trong điều khoản, viết dưới dạng CỤM ĐỘNG TỪ NGẮN GỌN (3–6 từ), chuẩn hóa. ĐỪNG bỏ sót nhiệm vụ.
  • Nếu văn bản diễn đạt nhiệm vụ bằng câu dài, PHẢI RÚT GỌN về cụm động từ cốt lõi — KHÔNG sao chép nguyên câu/mệnh đề làm tên Task.
  • Mỗi Task nên gắn với chủ thể thực hiện qua quan hệ CHU_TRI / CHIU_TRACH_NHIEM / PHOI_HOP (Task → Organization/Person).
    ĐÚNG: "Giám sát an ninh mạng", "Thẩm định an ninh mạng", "Kiểm tra, đánh giá an ninh mạng", "Quản lý rủi ro".
    SAI (nguyên câu, quá dài): "Có biện pháp, giải pháp để tìm và phát hiện kịp thời các điểm yếu, lỗ hổng về mặt kỹ thuật".
- Location: Địa điểm, địa danh cụ thể liên quan đến nội dung văn bản hoặc nơi ban hành văn bản.

## Quy tắc Composite Key cho Person (theo thứ tự ưu tiên):
1. "[Họ Tên] (DD/MM/YYYY)" — nếu có ngày sinh
2. "[Họ Tên] (Số CCCD)" — nếu có số CCCD/định danh cá nhân
3. "[Họ Tên] ([Đơn vị công tác rõ nhất])" — ví dụ: "Nguyễn Văn A (Sở Tài chính Nghệ An)"
4. "[Họ Tên] (không xác định)" — nếu không có thông tin định danh nào

## Quy tắc Canonicalization cho Organization:
- Các entity name được format không có các ký tự đặc biệt như: #, ?, *, ...
- Luôn dùng tên đầy đủ. Ví dụ: "UBND Tỉnh Nghệ An", không dùng "UBND tỉnh" hay "UBND"
- Sử dụng document_meta (thông tin văn bản) để suy diễn tên đầy đủ khi văn bản dùng tên tắt

## Các loại quan hệ được phép (PHẢI dùng chính xác tên sau):
- CAN_CU: Văn bản hiện tại căn cứ vào/dựa trên văn bản pháp lý khác. Source: Document → Target: Document
- VIEN_DAN: Điều khoản viện dẫn/tham chiếu một quy định khác. Source: Article → Target: Document/Article
- SUA_DOI: Văn bản sửa đổi, bổ sung văn bản khác. Source: Document → Target: Document
- CHU_TRI: Đơn vị, cơ quan, cá nhân chủ trì thực hiện. Source: Task/Article → Target: Organization/Person
- PHOI_HOP: Đơn vị, cơ quan, cá nhân phối hợp thực hiện. Source: Task/Article → Target: Organization/Person
- CHIU_TRACH_NHIEM: Đơn vị chịu trách nhiệm thi hành hoặc giám sát. Source: Task/Article → Target: Organization/Person
- PART_OF: Điều/Khoản thuộc cấu trúc của văn bản. Source: Article → Target: Document
- REFERENCES: Điều/Khoản tham chiếu chung đến văn bản/điều khác. Source: Article → Target: Document/Article
- KY: Người ký ban hành văn bản. Source: Document → Target: Person

## QUY TẮC NGHIÊM NGẶT:
1. CHỈ trả về JSON hợp lệ, không có markdown, không có giải thích thêm.
2. KHÔNG được tạo ra bất kỳ loại quan hệ nào ngoài danh sách trên.
3. KHÔNG được tạo entity type ngoài danh sách trên.
4. XỬ LÝ TỰ THAM CHIẾU: TUYỆT ĐỐI KHÔNG trích xuất các cụm từ "quy định này", "quyết định này", "văn bản này" làm thực thể độc lập. Khi gặp câu "Điều X của quy định/quyết định này", BỎ QUA cụm từ chỉ văn bản, CHỈ lấy "Điều X" (loại Article). Nếu văn bản nói "Sở này", "cơ quan này", tìm ngữ cảnh trước đó để ghi TÊN ĐẦY ĐỦ.
5. LOẠI BỎ THỰC THỂ CHUNG CHUNG/RÁC: KHÔNG tạo entity cho tổ chức không định danh ("Bộ", "Cơ quan ngang Bộ", "Cơ quan nhà nước", "các … có liên quan"), placeholder biểu mẫu ("(tên đơn vị đề nghị)"), hay tên biểu mẫu ("Mẫu số 02", "Tờ trình"). Thà bỏ sót còn hơn tạo node rác. Nếu cần biểu diễn quan hệ tới một chủ thể chung chung, gắn quan hệ tới Article hoặc Organization định danh được trong ngữ cảnh, KHÔNG tạo node chung chung.
6. Document KHÔNG kèm hậu tố ngày ban hành. Task: VẪN trích xuất đầy đủ các nhiệm vụ chính, nhưng tên Task PHẢI rút gọn thành cụm động từ ngắn (3–6 từ), KHÔNG phải nguyên câu.
7. Nếu không trích xuất được gì, trả về: {"entities": [], "relations": []}
"""

USER_TEMPLATE = """## Thông tin văn bản (document_meta)
Tiêu đề văn bản: "{document_title}"
Số hiệu: {document_number}
Cơ quan ban hành: {issuing_agency}
Ngày ban hành: {published_date}

Nội dung cần phân tích:
{article_text}

## Lưu ý quan trọng khi trích xuất:
- **Phân biệt Document vs Organization**: Tiêu đề văn bản (document_title) thường là TÊN ĐẦY ĐỦ của văn bản pháp luật (VD: "Luật Bảo vệ Bí mật nhà nước", "Kế hoạch triển khai"). Nếu entity trùng hoặc gần trùng với tiêu đề → đây là Document (văn bản), KHÔNG phải Organization.
- **Số hiệu**: Nếu entity chứa số hiệu văn bản (VD: "13/2024/QH15") → đây là Document.
- **Organization**: Là cơ quan/tổ chức CỤ THỂ được nhắc đến trong điều khoản, không phải tên văn bản.

Hãy trích xuất entities và relations theo đúng schema đã quy định.
Trả về JSON có dạng:
{{
  "entities": [
    {{"name": "...", "type": "Article|Document|Organization|Person|Task|Location", "description": "..."}}
  ],
  "relations": [
    {{"source": "...", "relation": "CAN_CU|VIEN_DAN|SUA_DOI|CHU_TRI|PHOI_HOP|CHIU_TRACH_NHIEM|PART_OF|REFERENCES|KY", "target": "...", "description": "..."}}
  ]
}}"""
# ===========================================================================
#  END editable prompt
# ===========================================================================


# --- Quality grader: flags the noise we saw in the live graph -------------
# This mirrors the proposed deterministic code-guard (A). If a prompt tweak is
# working, the GENERIC/JUNK buckets should shrink without the grader.
_JUNK_EXACT = {"task", "tờ trình", "mẫu số", "phương án"}
_GENERIC_EXACT = {
    "bộ", "bộ trưởng", "các bộ trưởng", "cơ quan", "cơ quan ngang bộ",
    "cơ quan thuộc chính phủ", "đơn vị", "doanh nghiệp", "tổ chức",
    "chủ quản hệ thống thông tin", "cá nhân", "tổ chức, cá nhân",
}
_GENERIC_PAT = re.compile(
    r"(có liên quan|liên quan$|^các\b|^những\b|, *cá nhân|, *tổ chức|"
    r"^cơ quan,|^tổ chức,|nhà nước$|^bộ, *ngành)",
    re.IGNORECASE,
)
_PLACEHOLDER_PAT = re.compile(r"\((?:tên|chủ quản|đơn vị|cơ quan)\b[^)]*\)", re.IGNORECASE)
_FORM_PAT = re.compile(r"^(mẫu số|tờ trình|phương án|biểu mẫu|đơn đề nghị)\b", re.IGNORECASE)


def grade(name: str, etype: str) -> str:
    n = (name or "").strip()
    low = n.lower()
    if not n:
        return "JUNK"
    if low in _JUNK_EXACT or _FORM_PAT.search(n) or _PLACEHOLDER_PAT.search(n):
        return "JUNK"
    # comma-separated enumerations ("A, B, C có liên quan") = not one entity
    if etype == "Organization":
        if low in _GENERIC_EXACT or _GENERIC_PAT.search(n):
            return "GENERIC"
        if n.count(",") >= 2:
            return "GENERIC"
    return "GOOD"


async def pick_document(ws: uuid.UUID, doc_arg: str | None) -> Document:
    async with async_session_maker() as db:
        if doc_arg:
            d = await db.scalar(select(Document).where(Document.id == uuid.UUID(doc_arg)))
            if not d:
                sys.exit(f"document {doc_arg} not found")
            return d
        rows = (
            await db.scalars(
                select(Document)
                .where(Document.workspace_id == ws)
                .where(Document.markdown_s3_key.isnot(None))
                .order_by(Document.created_at)
            )
        ).all()
        if not rows:
            sys.exit(f"no documents with markdown in workspace {ws}")
        if len(rows) > 1:
            print("Multiple docs — pass --doc <id> to choose. Available:")
            for r in rows:
                print(f"  {r.id}  {r.document_number or '?'}  {r.original_filename}")
            print(f"\nUsing first: {rows[0].id}\n")
        return rows[0]


async def run_model(label: str, cfg: dict, articles: list[dict], meta: dict) -> dict:
    """Extract every article with one model; print per-article + return aggregate."""
    provider = OpenAICompatibleLLMProvider(
        base_url=cfg["base_url"], model=cfg["model"], api_key=cfg["api_key"]
    )
    print("\n" + "#" * 80)
    print(f"# MODEL '{label}'  →  {cfg['model']}  @  {cfg['base_url']}")
    print("#" * 80)

    all_entities: list[dict] = []
    all_relations: list[dict] = []
    for art in articles:
        user_prompt = USER_TEMPLATE.format(
            document_title=meta["title"] or "Không có tiêu đề",
            document_number=meta["number"] or "Không có số hiệu",
            issuing_agency=meta["agency"] or "Không xác định",
            published_date=meta["published"] or "Không xác định",
            article_text=art["text"][:3000],
        )
        try:
            data = _parse_llm_json(await call_model(provider, SYSTEM_PROMPT, user_prompt))
        except Exception as e:
            print(f"  Điều {art['index']}: LLM FAIL — {e}")
            continue
        ents, rels = data.get("entities", []), data.get("relations", [])
        print(f"── Điều {art['index']}: {len(ents)} entity, {len(rels)} relation")
        for e in ents:
            g = grade(e.get("name", ""), e.get("type", ""))
            print(f"     [{e.get('type','?'):12}] {e.get('name','')}"
                  + ("" if g == "GOOD" else f"  ⚠ {g}"))
            e["_article"] = art["index"]
            all_entities.append(e)
        all_relations.extend(rels)

    # ---- aggregate ----
    seen: dict[tuple, dict] = {}
    for e in all_entities:
        seen.setdefault(
            (str(e.get("name", "")).strip().lower(), str(e.get("type", "")).strip()), e
        )
    by_grade: dict[str, list[tuple[str, str]]] = defaultdict(list)
    by_type: dict[str, int] = defaultdict(int)
    for (name_l, etype), e in seen.items():
        by_grade[grade(e.get("name", ""), etype)].append((e.get("name", ""), etype))
        by_type[etype] += 1

    print(f"\n──── '{label}' AGGREGATE ────")
    for g in ("JUNK", "GENERIC"):
        rows = sorted(by_grade.get(g, []))
        print(f"\n{g}  ({len(rows)}) — should NOT become nodes:")
        for name, etype in rows:
            print(f"   [{etype:12}] {name}")
    print(f"\nDocument names ({by_type.get('Document', 0)}):")
    for (name_l, etype), e in sorted(seen.items()):
        if etype == "Document":
            mark = "📅" if re.search(r"ngày\s+\d", name_l) else "  "
            print(f"   {mark} {e.get('name','')}")

    total = sum(by_type.values())
    junk = len(by_grade.get("JUNK", []))
    gen = len(by_grade.get("GENERIC", []))
    print("\nSUMMARY")
    for t, c in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f"   {t:14} {c}")
    print(f"   {'TOTAL':14} {total}   |   noise: {junk} junk + {gen} generic "
          f"= {junk+gen} ({100*(junk+gen)//max(total,1)}%)")

    return {
        "label": label, "model": cfg["model"], "entities": all_entities,
        "relations": all_relations, "by_type": dict(by_type),
        "unique": len(seen), "total": total, "junk": junk, "generic": gen,
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--doc", default=None)
    ap.add_argument("--articles", type=int, default=0, help="0=all, N=first N")
    ap.add_argument("--only", type=int, action="append", help="extract only this Điều index (repeatable)")
    ap.add_argument("--model", choices=["memory", "qwen36", "both"], default="both",
                    help="which model(s) to run (default: both)")
    ap.add_argument("--json", default=None, help="dump per-model raw to <name>.<label>.json")
    args = ap.parse_args()

    ws = uuid.UUID(args.workspace)
    doc = await pick_document(ws, args.doc)

    title = doc.document_title or ""
    number = doc.document_number or ""
    agency = doc.issuing_agency or ""
    published = doc.published_date or ""

    print("=" * 80)
    print(f"DOC  {doc.id}")
    print(f"file {doc.original_filename}")
    print(f"số hiệu   : {number or '—'}")
    print(f"tiêu đề   : {title or '—'}")
    print(f"cơ quan   : {agency or '—'}")
    print(f"ngày BH   : {published or '—'}")
    print("=" * 80)

    storage = StorageService()
    markdown = await storage.download_markdown(doc.markdown_s3_key)
    articles = split_articles(markdown)
    print(f"split → {len(articles)} điều")

    if args.only:
        articles = [a for a in articles if a["index"] in set(args.only)]
    elif args.articles > 0:
        articles = articles[: args.articles]
    print(f"testing {len(articles)} điều\n")

    meta = {"title": title, "number": number, "agency": agency, "published": published}
    labels = ["memory", "qwen36"] if args.model == "both" else [args.model]

    results = []
    for label in labels:
        res = await run_model(label, MODELS[label], articles, meta)
        results.append(res)
        if args.json:
            path = args.json.replace(".json", f".{label}.json")
            with open(path, "w") as f:
                json.dump({"entities": res["entities"], "relations": res["relations"]},
                          f, ensure_ascii=False, indent=2)
            print(f"raw → {path}")

    # ---- side-by-side comparison ----
    if len(results) > 1:
        print("\n" + "=" * 80)
        print(f"COMPARISON  ({len(articles)} điều)")
        print("=" * 80)
        print(f"{'metric':16}" + "".join(f"{r['label']:>14}" for r in results))
        rows = [
            ("entities(raw)", lambda r: len(r["entities"])),
            ("unique", lambda r: r["unique"]),
            ("relations", lambda r: len(r["relations"])),
            ("junk", lambda r: r["junk"]),
            ("generic", lambda r: r["generic"]),
            ("noise%", lambda r: f"{100*(r['junk']+r['generic'])//max(r['total'],1)}%"),
        ]
        for name, fn in rows:
            print(f"{name:16}" + "".join(f"{str(fn(r)):>14}" for r in results))
        all_types = sorted({t for r in results for t in r["by_type"]})
        print("  — by type —")
        for t in all_types:
            print(f"{t:16}" + "".join(f"{str(r['by_type'].get(t, 0)):>14}" for r in results))
        print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
