"""
MongoDB Schema Registry — People Search Map
==========================================

Định nghĩa ánh xạ từ lookup type (CCCD, BHXH, phone, name)
sang collection + field names trong MongoDB.

Cấu trúc mỗi entry:
    <lookup_type>: {
        "description": str,
        "collections": {
            <schema_name>: {
                "fields": [<field_canonical_name>, ...],   # field dùng để query
                "display_fields": [<field_display_name>, ...], # field hiển thị trong kết quả
            }
        }
    }

Field gốc (canonical) = field name trong MongoDB collection thực tế.
Field hiển thị = field name muốn show ra trong kết quả tra cứu.
"""

from __future__ import annotations

# ============================================================================
# SEARCHABLE COLLECTION MAP
# ============================================================================
# TODO: Xác nhận các trường từ người dùng (đã comment bên dưới)
# ============================================================================

SEARCHABLE_COLLECTION_MAP: dict[str, dict] = {

    # --------------------------------------------------------------------------
    # CCCD / CMND lookup — tìm theo số CCCD hoặc CMND
    # --------------------------------------------------------------------------
    "cccd": {
        "description": "Tìm theo số CCCD/CMND",
        "collections": {
            "bhxh": {
                # soCmnd: Số CMND trong hồ sơ BHXH
                "fields": ["soCmnd"],
                # Các trường hiển thị: hoTen, maSoBhxh, soTheBhyt, ngaySinhHienThi, trangThaiThe, coSoKCB
                "display_fields": ["hoTen", "maSoBhxh", "soTheBhyt", "ngaySinhHienThi", "trangThaiThe", "coSoKCB"],
            },
            "evn": {
                # cmnd: Số CMND trong hồ sơ điện lực
                "fields": ["cmnd"],
                # TODO: Xác nhận các display_fields cho evn
                # Ví dụ: hoTen, dienThoai, diaChi, ...?
                "display_fields": [],
            },
            "lg": {
                # SoDinhDanh: Số CCCD/CMND của hội viên LG
                "fields": ["SoDinhDanh"],
                # display_fields: TenHoiVien, SoDienThoai, NgaySinh, DiaChi, TenHangHoiVien, SoTheHoiVien
                "display_fields": ["TenHoiVien", "SoDienThoai", "NgaySinh", "DiaChi", "TenHangHoiVien", "SoTheHoiVien"],
            },
            "vacxin": {
                # MA_DOI_TUONG: Mã định danh (CCCD/CMND) của đối tượng tiêm chủng
                "fields": ["MA_DOI_TUONG"],
                # display_fields: HO_TEN, NGAY_SINH, TEN_ME, DIEN_THOAI_ME, GIOI_TINH, PID
                "display_fields": ["HO_TEN", "NGAY_SINH", "TEN_ME", "DIEN_THOAI_ME", "GIOI_TINH", "PID"],
            },
            # TODO: cv19 có trường CCCD/CMND không? tên field là gì?
            # "cv19": {
            #     "fields": ["<ten_field_cccd>"],
            #     "display_fields": [],
            # },
            # TODO: uids có trường CCCD/CMND không? tên field là gì?
            # "uids": {
            #     "fields": ["<ten_field_cccd>"],
            #     "display_fields": [],
            # },
            # TODO: vnvc có trường CCCD/CMND không? tên field là gì?
            # "vnvc": {
            #     "fields": ["<ten_field_cccd>"],
            #     "display_fields": [],
            # },
        },
    },

    # --------------------------------------------------------------------------
    # BHXH number lookup — tìm theo số BHXH
    # --------------------------------------------------------------------------
    "bhxh": {
        "description": "Tìm theo số BHXH",
        "collections": {
            "bhxh": {
                # maSoBhxh: Mã số BHXH
                "fields": ["maSoBhxh"],
                "display_fields": ["hoTen", "soTheBhyt", "ngaySinhHienThi", "trangThaiThe", "tyLeBhyt", "tuNgay", "denNgay", "coSoKCB"],
            },
            # TODO: Các schema khác (evn, lg, vacxin, cv19, uids, vnvc) có lưu số BHXH không?
            # Nếu có, thêm vào đây, ví dụ:
            # "evn": {
            #     "fields": ["<ten_field_bhxh>"],
            #     "display_fields": [],
            # },
        },
    },

    # --------------------------------------------------------------------------
    # Phone number lookup — tìm theo số điện thoại
    # --------------------------------------------------------------------------
    "phone": {
        "description": "Tìm theo số điện thoại",
        "collections": {
            "bhxh": {
                "fields": ["soDienThoai"],
                "display_fields": ["hoTen", "maSoBhxh", "soTheBhyt", "ngaySinhHienThi", "trangThaiThe", "coSoKCB"],
            },
            "evn": {
                "fields": ["dienThoai"],
                "display_fields": ["tenKhachHang", "cmnd", "diaChiCapDien", "ngayDangKy"],
            },
            "lg": {
                "fields": ["SoDienThoai"],
                "display_fields": ["TenHoiVien", "SoDinhDanh"],
            },
            "vacxin": {
                # DIEN_THOAI_ME: Điện thoại liên hệ (thường là mẹ)
                "fields": ["DIEN_THOAI_ME"],
                "display_fields": ["HO_TEN", "NGAY_SINH", "TEN_ME", "DIEN_THOAI_ME", "GIOI_TINH", "PID"],
            },
            "cv19": {
                # so_dien_thoai: Số điện thoại trong hồ sơ covid
                "fields": ["so_dien_thoai"],
                # TODO: Xác nhận display_fields cho cv19
                "display_fields": ["ho_ten", "so_dien_thoai", "namsinh", "gioi_tinh", "dia_chi"]
            },
            "uids": {
                "fields": ["phone"],
                "display_fields": ["uids"],
            },
            "vnvc": {
                # mobile: Số điện thoại trong hồ sơ tiêm VNVC
                "fields": ["mobile"],
                # Actual field names: fullName (name), fullNam (birth date), mobile, diaChi (address)
                "display_fields": ["fullName", "fullNam", "mobile", "diaChi", "TEN_ME", "gioi_tinh"],
            },
        },
    },

    # --------------------------------------------------------------------------
    # Name lookup — tìm theo tên người
    # --------------------------------------------------------------------------
    "name": {
        "description": "Tìm theo tên",
        "collections": {
            "bhxh": {
                "fields": ["hoTen"],
                "display_fields": ["hoTen", "maSoBhxh", "soTheBhyt", "soCmnd", "soDienThoai", "ngaySinhHienThi"],
            },
            "lg": {
                "fields": ["TenHoiVien"],
                "display_fields": ["TenHoiVien", "SoDinhDanh", "SoDienThoai", "NgaySinh", "DiaChi"],
            },
            "vacxin": {
                "fields": ["HO_TEN"],
                "display_fields": ["HO_TEN", "MA_DOI_TUONG", "NGAY_SINH", "DIEN_THOAI_ME", "TEN_ME"],
            },
            # TODO: evn có trường tên không? tên field là gì?
            "evn": {
                "fields": ["tenKhachHang"],
                "display_fields": ["tenKhachHang", "cmnd", "phone", "diaChiCapDien", "ngayDangKy"],
            },
            # TODO: cv19 có trường tên không?
            "cv19": {
                "fields": ["ho_ten"],
                "display_fields": ["ho_ten", "so_dien_thoai", "namsinh", "gioi_tinh", "dia_chi"],
            },
            # TODO: uids có trường tên không?
            # "uids": {
            #     "fields": ["<ten_field_ten>"],
            #     "display_fields": [],
            # },
            "vnvc": {
                "fields": ["fullName"],
                "display_fields": ["fullName", "mobile", "fullNam", "diaChi", "TEN_ME"],
            },
        },
    },
}


# ============================================================================
# Lookup type → intent mapping (từ agent intent)
# ============================================================================

INTENT_TO_LOOKUP_TYPE: dict[str, str] = {
    "mongo_search_cccd":  "cccd",
    "mongo_search_bhxh":  "bhxh",
    "mongo_search_phone": "phone",
    "mongo_search_name":  "name",
}


# ============================================================================
# Schema descriptions — giải thích ý nghĩa mỗi schema cho LLM
# ============================================================================

SCHEMA_DESCRIPTIONS: dict[str, str] = {
    "bhxh": "Hồ sơ Bảo hiểm xã hội — thông tin BHXH, BHYT, thẻ y tế, cơ sở khám chữa bệnh",
    "evn":  "Hồ sơ điện lực — thông tin khách hàng điện lực",
    "lg":   "Thông tin thuê bao Vinaphone",
    "vacxin": "Hồ sơ tiêm chủng — thông tin tiêm chủng vaccine, đối tượng tiêm, thông tin phụ huynh",
    "cv19": "Hồ sơ COVID-19 — thông tin xét nghiệm, tiêm vaccine COVID-19",
    "uids": "Hồ sơ UID Facebook",
    "vnvc": "Hồ sơ tiêm chủng VNVC — thông tin đăng ký tiêm vaccine tại VNVC",
}


def get_schema_display_name(schema: str) -> str:
    """Trả về tên viết tắt có mô tả cho một schema."""
    desc = SCHEMA_DESCRIPTIONS.get(schema, schema)
    return desc


def enrich_display_with_schema(display: str, schemas_in_result: list[str]) -> str:
    """
    Thêm mô tả schema vào đầu kết quả để LLM hiểu nguồn dữ liệu.
    """
    schema_lines = []
    for s in schemas_in_result:
        schema_lines.append(f"  • {get_schema_display_name(s)}")

    header = (
        "**Nguồn dữ liệu truy vấn:**\n"
        + "\n".join(schema_lines)
        + "\n\n"
    )
    return header + display
