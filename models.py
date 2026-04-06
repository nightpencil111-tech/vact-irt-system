from sqlalchemy import Column, Integer, String, Float, DateTime
from database import Base
from datetime import datetime

class Question(Base):
    __tablename__ = "questions"

    id = Column(Integer, primary_key=True, index=True)
    content = Column(String, nullable=False)
    option_a = Column(String, nullable=False)
    option_b = Column(String, nullable=False)
    option_c = Column(String, nullable=False)
    option_d = Column(String, nullable=False)
    correct_answer = Column(String, nullable=False) # A, B, C hoặc D
    
    # Các tham số chuẩn IRT
    param_a = Column(Float, nullable=False) # Độ phân biệt
    param_b = Column(Float, nullable=False) # Độ khó
    param_c = Column(Float, nullable=False) # Độ phỏng đoán
    
# THÊM BẢNG MỚI VÀO DƯỚI CÙNG:
class Submission(Base):
    __tablename__ = "submissions"

    id = Column(Integer, primary_key=True, index=True)
    student_name = Column(String, index=True)
    exam_code = Column(String, default="VACT-2026") # Mã đề thi
    raw_score = Column(String) # Điểm thô (vd: 8/10)
    theta_score = Column(Float) # Điểm năng lực IRT
    submitted_at = Column(DateTime, default=datetime.utcnow) # Thời gian nộp