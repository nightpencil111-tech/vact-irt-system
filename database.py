from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# Mật khẩu đã được đổi @ thành %40 để hệ thống đọc hiểu chính xác
SQLALCHEMY_DATABASE_URL = "postgresql://postgres.hhauuiumctlflauqrkfa:Vact_IRT%402026@aws-1-ap-northeast-1.pooler.supabase.com:5432/postgres"

# Khởi tạo engine kết nối
engine = create_engine(SQLALCHEMY_DATABASE_URL)

# Tạo một "nhà máy" sản xuất các phiên làm việc (Session) với Database
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base dùng để khai báo các model (bảng) sau này
Base = declarative_base()

# Hàm để các API lấy kết nối DB khi cần
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()