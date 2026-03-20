# Kế hoạch Tích hợp Người dùng và Phân quyền Workspace

Tài liệu này đề xuất kiến trúc để tích hợp người dùng vào hệ thống NexusRAG, hỗ trợ 3 cấp độ Workspace: **Public** (Công cộng), **Tenant/Company** (Công ty), và **Personal** (Cá nhân).

## 1. Mở rộng Database Schema (PostgreSQL)

Để quản lý người dùng và tổ chức (Tenant), chúng ta cần thêm các bảng sau:

### 1.1 `users` Table
Lưu trữ thông tin người dùng.
- `id`: Định danh duy nhất (UUID/Integer).
- `email`: Địa chỉ email (Unique).
- `password_hash`: Hash mật khẩu (nếu dùng mật khẩu truyền thống).
- `two_factor_secret`: Cột lưu secret key để sau này có thể tích hợp Google Authenticator (OTP/2FA).
- `full_name`: Tên hiển thị của người dùng.
- `is_active`: Trạng thái hoạt động.
- `created_at`, `updated_at`.

### 1.2 `tenants` (hoặc `organizations`) Table
Đại diện cho một Công ty hoặc một Tổ chức.
- `id`: Định danh duy nhất.
- `name`: Tên công ty.
- `domain`: Tên miền (tuỳ chọn - dùng để auto-map user theo email công ty).

### 1.3 `tenant_users` Table (Bảng trung gian)
Liên kết User và Tenant, đồng thời xác định vai trò của User trong Tenant đó.
- `tenant_id`: Foreign Key tham chiếu `tenants.id`.
- `user_id`: Foreign Key tham chiếu `users.id`.
- `role`: Vai trò (VD: `admin`, `member`, `guest`).

---

## 2. Cập nhật Model `KnowledgeBase` (Workspace)

Hiện tại, `KnowledgeBase` là đơn vị phân tách dữ liệu (mỗi workspace 1 ChromaDB collection & 1 LightRAG KG instance). Chúng ta sẽ tận dụng điều này bằng cách thêm các trường phân quyền vào `KnowledgeBase`:

```python
class VisibilityEnum(str, enum.Enum):
    PUBLIC = "PUBLIC"       # Ai cũng truy cập được
    TENANT = "TENANT"       # Chỉ người trong Công ty truy cập được
    PERSONAL = "PERSONAL"   # Chỉ người tạo ra truy cập được

class KnowledgeBase(Base):
    # Các trường hiện tại vẫn giữ nguyên (id, name, description, v.v.)
    
    # [NEW] Phân loại hiển thị
    visibility: Mapped[VisibilityEnum] = mapped_column(String, default=VisibilityEnum.PERSONAL)
    
    # [NEW] Người sở hữu (Dành cho PERSONAL Workspace)
    owner_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("users.id"), nullable=True)
    
    # [NEW] Công ty sở hữu (Dành cho TENANT Workspace)
    tenant_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("tenants.id"), nullable=True)
```

---

## 3. Quản lý Truy cập (Access Control ở Backend)

Dựa vào mô hình trên, FastAPI sẽ có authorizer module để kiểm tra và lấy danh sách Workspace tương ứng với người dùng đang truy cập.

### 3.1 Fetch Danh sách Workspace
Khi người dùng (đã xác thực qua JWT token) gọi API GET Workspaces, truy vấn SQL (hoặc SQLAlchemy) sẽ kết hợp các điều kiện:
1. **Public:** Lấy các workspace có `visibility = 'PUBLIC'`.
2. **Tenant:** Lấy các workspace có `visibility = 'TENANT' AND tenant_id IN (danh sách_tenant_id_của_user)`.
3. **Personal:** Lấy các workspace có `visibility = 'PERSONAL' AND owner_id = current_user.id`. 
   *(Lưu ý quan trọng: System và Tenant/Company tuyệt đối không thể xem hay truy cập vào Personal Workspace hoặc Personal File của người dùng. Dữ liệu cá nhân được cách ly 100%).*

### 3.2 Tương tác với Workspace (Chat & Upload)
- **Chat:** User có thể đặt câu hỏi trong bất kỳ workspace nào họ thấy ở bước 3.1.
- **Upload Document:**
  - `PUBLIC`: Có thể chỉ cho phép Admin hệ thống upload để tránh rác dữ liệu.
  - `TENANT`: Cho phép user có role hợp lệ (`admin` hoặc `member` trong `tenant_users`) tiếp tục upload.
  - `PERSONAL`: Chủ sở hữu (`owner_id`) mới được upload.

---

## 4. Tương tác với Message Queue (RabbitMQ Pipeline) & Theo dõi tiến trình

Hiện tại pipeline đang hoạt động dựa trên background workers (Parse -> Embed -> Caption -> KG).
- Thiết kế mới này **không can thiệp vào logic xử lý NLP/AI của worker**, nhưng cần quản lý trạng thái task (Job Status).
- API endpoint sẽ kiểm tra quyền Access (ở bước 3) của người dùng **trước khi** đẩy message (gồm `document_id`, `knowledge_base_id`, cộng thêm `owner_id`/`tenant_id`) vào RabbitMQ.
- **Tiến trình xử lý (Queue Status):** Người dùng và Tenant chỉ có thể theo dõi trạng thái các task xử lý tài liệu (Queue) thuộc quyền sở hữu của mình:
  - User cá nhân **không thể** xem các task đang chạy của Tenant.
  - Tenant (hoặc Admin của Tenant) **không thể** xem các task đang chạy thuộc Personal Workspace của User.
  - API kiểm tra trạng thái Queue phải đảm bảo đi qua cùng bộ phân quyền (Authorization) như Fetch Workspace để cô lập tuyệt đối.

---

## 5. Thay đổi trên Frontend (React + Zustand)

- **Giao diện:** Chia Knowledge Bases list thành các section rõ rệt: `Public Workspaces`, `Tenant Workspaces (Tên công ty)`, và `My Personal Workspaces`.
- **Luồng Auth:** Thêm trang Login/Register, lưu trữ JWT token. Cập nhật `lib/api.ts` để luôn đính kèm `Authorization: Bearer <token>` vào headers của mọi request lên backend.

---

## 6. Mở rộng tính năng Lịch sử Chat (Chat Sessions)

Hiện tại, model `ChatMessage` chỉ lưu các tin nhắn nhồi chung vào `Workspace` mà chưa có sự tách biệt theo từng "Phiên trò chuyện" (Thread/Session). Đồng thời, ta cần đảm bảo hệ thống lưu trữ toàn vẹn cả lịch sử tin nhắn của Người dùng (`user`) và Phản hồi của Bot (`assistant`).

Để hỗ trợ người dùng tạo nhiều đoạn chat khác nhau trong cùng 1 Workspace, chúng ta cần:

### 6.1 Thêm Model `ChatSession`
Đây là bảng đóng vai trò nhóm các tin nhắn lại thành một cuộc hội thoại riêng biệt.
- `id`: Định danh duy nhất (UUID).
- `title`: Tiêu đề tự động sinh ra cho đoạn chat (ví dụ: "Phân tích báo cáo tài chính...").
- `workspace_id`: Foreign Key tham chiếu tới `KnowledgeBase`.
- `user_id`: Foreign Key tham chiếu tới `User` (Đảm bảo chỉ user đó mới thấy lịch sử chat của mình, dù ở trong Public/Tenant workspace).
- `created_at`, `updated_at`.

### 6.2 Cập nhật Model `ChatMessage`
- Bổ sung trường `session_id`: Foreign Key tham chiếu tới `ChatSession` (thay vì chỉ gắn trực tiếp vào `workspace_id`).
- Hệ thống backend sẽ đảm bảo lưu toàn bộ chuỗi Message: Khi user gửi 1 câu hỏi (lưu với `role = 'user'`), sau khi RAG và LLM trả lời xong, pipeline sẽ ghi phản hồi vào DB (lưu với `role = 'assistant'`) vào cùng 1 `session_id`.

### 6.3 Trên Frontend
- Nền tảng UI `WorkspacePage` sẽ có 1 thanh Sidebar hiển thị **"Recent Chats / Lịch sử chat"**.
- Nút **"New Chat"** để tạo một `ChatSession` mới.
- Khi người dùng click vào lịch sử cũ, hệ thống fetch mảng `ChatMessage` dựa trên `session_id` và khôi phục lại ngữ cảnh.

---

*Lưu ý: Nếu bạn đánh giá OK, chúng ta có thể tiến hành định hình cụ thể các file cần sửa để tạo ra các bảng này qua công cụ Alembic migrations.*
