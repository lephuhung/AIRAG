"""
Unit tests cho ``validity_extractor`` — thuần regex, không cần stack.

Các fixture lấy NGUYÊN VĂN (kể cả khoản đánh số, chấm phẩy liệt kê) từ các văn
bản thật trong kho ngày 2026-07-03: 85/2016/NĐ-CP Điều 25, 361/2025/NĐ-CP
Điều 13 (hết hiệu lực MỘT PHẦN), 116/2025/QH15 Điều 43-44 (hai luật cùng hết
hiệu lực trong một câu chấm-phẩy + bãi bỏ từng khoản + văn bản công cụ "theo
Luật số 35/2018/QH14" không được tính là đối tượng).
"""
from __future__ import annotations

from app.services.legal.validity_extractor import extract_validity

# Đệm cho điều thi hành nằm ở nửa sau văn bản (extractor chỉ quét sau 30%).
_FILLER = "\n".join(
    f"## Điều {n}. Quy định chung số {n}\n\nNội dung điều {n} nói về phạm vi "
    "trách nhiệm của các cơ quan, tổ chức có liên quan trong việc triển khai."
    for n in range(1, 6)
)


def test_effective_date_full_form_no_events():
    md = f"""{_FILLER}

## Chương V

# ĐIỀU KHOẢN THI HÀNH

## Điều 25. Hiệu lực thi hành

Nghị định này có hiệu lực thi hành từ ngày 01 tháng 7 năm 2016.

## Điều 26. Tổ chức thực hiện

$1. $ Bộ Thông tin và Truyền thông chịu trách nhiệm hướng dẫn, kiểm tra việc thực hiện Nghị định này.
"""
    info = extract_validity(md)
    assert info.effective_date == "01/07/2016"
    assert info.events == []


def test_sign_date_and_partial_supersede():
    md = f"""{_FILLER}

## Chương IV ĐIỀU KHOẢN THI HÀNH

## Điều 13. Hiệu lực thi hành

1. Nghị định này có hiệu lực thi hành kể từ ngày ký ban hành.

2. Kể từ ngày Nghị định này có hiệu lực thi hành, các quy định liên quan đến vị trí việc làm công chức tại Nghị định số 62/2020/NĐ-CP ngày 01 tháng 6 năm 2020 của Chính phủ về vị trí việc làm và biên chế công chức hết hiệu lực thi hành.

3. Trường hợp cấp có thẩm quyền ban hành văn bản quy định về công tác cán bộ có nội dung khác với quy định tại Nghị định này thì thực hiện theo quy định mới của cấp có thẩm quyền.

## Điều 14. Điều khoản chuyển tiếp và áp dụng
"""
    info = extract_validity(md)
    assert info.effective_date == "sign_date"
    assert len(info.events) == 1
    ev = info.events[0]
    assert ev.kind == "het_hieu_luc"
    assert ev.target_number == "62/2020/NĐ-CP"
    assert ev.scope == "partial"


def test_two_laws_expire_one_sentence_instrument_excluded():
    md = f"""{_FILLER}

## Chương VIII ĐIỀU KHOẢN THI HÀNH

## Điều 43. Sửa đổi, bổ sung một số điều của các luật có liên quan

17. Bãi bỏ khoản 3 Điều 49 của Luật Thư viện số 46/2019/QH14.

## Điều 44. Hiệu lực thi hành

1. Luật này có hiệu lực thi hành từ ngày 01 tháng 7 năm 2026.

2. Luật An toàn thông tin mạng số 86/2015/QH13 đã được sửa đổi, bổ sung một số điều theo Luật số 35/2018/QH14; Luật An ninh mạng số 24/2018/QH14 hết hiệu lực kể từ ngày Luật này có hiệu lực thi hành.

## Điều 45. Điều khoản chuyển tiếp
"""
    info = extract_validity(md)
    assert info.effective_date == "01/07/2026"
    by_target = {e.target_number: e for e in info.events}
    # Hai luật cùng hết hiệu lực toàn phần trong MỘT câu chấm-phẩy
    assert by_target["86/2015/QH13"].kind == "het_hieu_luc"
    assert by_target["86/2015/QH13"].scope == "full"
    assert by_target["24/2018/QH14"].kind == "het_hieu_luc"
    assert by_target["24/2018/QH14"].scope == "full"
    # Văn bản công cụ ("theo Luật số 35/2018/QH14") KHÔNG phải đối tượng
    assert "35/2018/QH14" not in by_target
    # Bãi bỏ từng khoản của Luật Thư viện = một phần
    assert by_target["46/2019/QH14"].kind == "bai_bo"
    assert by_target["46/2019/QH14"].scope == "partial"


def test_thay_the_full():
    md = f"""{_FILLER}

## Điều 30. Hiệu lực thi hành

1. Nghị định này có hiệu lực từ ngày 15/03/2025.

2. Nghị định này thay thế Nghị định số 90/2010/NĐ-CP ngày 08 tháng 8 năm 2010 của Chính phủ.
"""
    info = extract_validity(md)
    assert info.effective_date == "15/03/2025"
    assert len(info.events) == 1
    assert info.events[0].kind == "thay_the"
    assert info.events[0].target_number == "90/2010/NĐ-CP"
    assert info.events[0].scope == "full"


def test_body_mention_outside_enforcement_articles_ignored():
    # "thay thế"/số hiệu trong THÂN văn bản (điều nghiệp vụ) không được tính.
    md = f"""## Điều 1. Phạm vi điều chỉnh

Việc thay thế trang thiết bị theo Nghị định số 11/2011/NĐ-CP thực hiện định kỳ.

{_FILLER}

## Điều 30. Hiệu lực thi hành

Nghị định này có hiệu lực thi hành từ ngày 01 tháng 01 năm 2030.
"""
    info = extract_validity(md)
    assert info.effective_date == "01/01/2030"
    assert info.events == []


def test_admin_layout_html_markdown():
    # Đường admin-layout bọc nội dung trong <p class="ocr-*"> (vd 117/2025):
    # extractor phải tự bóc thẻ. Câu thật có typo OCR "kê từ ngày".
    md = f"""{_FILLER}

<p class="ocr-title ocr-center" data-bbox="383,750,670,788" data-page="20">Chương V<br/>ĐIỀU KHOẢN THI HÀNH</p>
<p class="ocr-title ocr-center" data-bbox="200,820,473,841" data-page="20">Điều 27. Hiệu lực thi hành</p>
<p class="ocr-center" data-bbox="200,847,794,865" data-page="20">1. Luật này có hiệu lực thi hành từ ngày 01 tháng 3 năm 2026.</p>
<p class="ocr-body" data-bbox="137,871,908,928" data-page="20">2. Luật Bảo vệ bí mật nhà nước số 29/2018/QH14 đã được sửa đổi, bổ sung một số điều theo Luật số 81/2025/QH15 hết hiệu lực kê từ ngày Luật này có hiệu lực thi hành.</p>
"""
    info = extract_validity(md)
    assert info.effective_date == "01/03/2026"
    by_target = {e.target_number: e for e in info.events}
    assert by_target["29/2018/QH14"].kind == "het_hieu_luc"
    assert by_target["29/2018/QH14"].scope == "full"
    assert "81/2025/QH15" not in by_target  # văn bản công cụ


def test_instrument_enumeration_excluded_and_phrase_swap_is_partial():
    # Câu "thay thế một số cụm từ" = sửa câu chữ (partial), và cả chuỗi luật
    # công cụ sau "sửa đổi, bổ sung ... theo" không được thành đối tượng.
    md = f"""{_FILLER}

## Điều 43. Sửa đổi, bổ sung một số điều của các luật có liên quan

1. Thay thế một số cụm từ của Luật Phí và lệ phí số 97/2015/QH13 đã được sửa đổi, bổ sung một số điều theo Luật số 23/2018/QH14, Luật số 72/2020/QH14, Luật số 16/2023/QH15 như sau: a) Thay thế cụm từ "an toàn thông tin" bằng cụm từ "an ninh mạng".

2. Thay thế một số cụm từ, bãi bỏ một số khoản của Luật Lưu trữ số 33/2024/QH15 như sau: a) Bãi bỏ khoản 2 Điều 41.

## Điều 44. Hiệu lực thi hành

Luật này có hiệu lực thi hành từ ngày 01 tháng 7 năm 2026.
"""
    info = extract_validity(md)
    by_target = {e.target_number: e for e in info.events}
    assert by_target["97/2015/QH13"].scope == "partial"
    assert by_target["33/2024/QH15"].scope == "partial"
    for instrument in ("23/2018/QH14", "72/2020/QH14", "16/2023/QH15"):
        assert instrument not in by_target


def test_enforcement_article_early_in_appendix_heavy_doc():
    # Nghị định kèm phụ lục khổng lồ: điều thi hành nằm ở ~5% đầu file —
    # không được gate theo vị trí.
    appendix = "\n".join(
        f"| STT {i} | Vị trí việc làm nghiệp vụ chuyên ngành số {i} | Chuyên viên |"
        for i in range(400)
    )
    md = f"""## Điều 13. Hiệu lực thi hành

1. Nghị định này có hiệu lực thi hành kể từ ngày ký ban hành.

2. Kể từ ngày Nghị định này có hiệu lực thi hành, các quy định liên quan đến vị trí việc làm công chức tại Nghị định số 62/2020/NĐ-CP hết hiệu lực thi hành.

## PHỤ LỤC I

{appendix}
"""
    info = extract_validity(md)
    assert info.effective_date == "sign_date"
    assert [e.target_number for e in info.events] == ["62/2020/NĐ-CP"]


def test_cong_van_without_structure_returns_empty():
    md = "V/v triển khai tiếp nhận và xử lý vướng mắc về nghĩa vụ thuế.\n\nKính gửi các đơn vị..."
    info = extract_validity(md)
    assert info.effective_date is None
    assert info.events == []


# ── extract_article_nos (heading_path.py) — nguồn metadata article_nos ──────

def test_extract_article_nos_variants():
    from app.services.parsing.heading_path import extract_article_nos

    assert extract_article_nos(["Chương V", "Điều 25. Hiệu lực thi hành"]) == ["25"]
    # chuỗi đã join + nhiều điều + không lặp
    assert extract_article_nos(
        "Chương I > Điều 3. Giải thích từ ngữ > Điều 4. Nguyên tắc > Điều 4. Nguyên tắc"
    ) == ["3", "4"]
    # điều có hậu tố chữ; "ĐIỀU KHOẢN THI HÀNH" (không có số) không được tính
    assert extract_article_nos(["Điều 5a: Bổ sung", "Chương IV ĐIỀU KHOẢN THI HÀNH"]) == ["5a"]
    assert extract_article_nos(None) == []
    assert extract_article_nos("") == []
