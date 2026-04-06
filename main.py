from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware

# 1. BẮT BUỘC PHẢI CÓ DÒNG NÀY TRƯỚC:
app = FastAPI(title="V-ACT IRT Scoring API")

# 2. RỒI MỚI ĐẾN DÒNG NÀY:
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

from sqlalchemy.orm import Session
from database import get_db
from pydantic import BaseModel
import numpy as np
from scipy.optimize import minimize
from typing import List
import google.generativeai as genai  # Bổ sung dòng này
import json                          # Bổ sung dòng này
from fastapi.responses import HTMLResponse, FileResponse
import os # Thêm dòng này lên đầu file nếu chưa có
from pydantic import BaseModel
from typing import List, Dict
import numpy as np
from scipy.optimize import minimize
from fastapi.staticfiles import StaticFiles

# CẤU HÌNH API KEY (Thay đoạn text bên dưới bằng Key thật của bạn)
genai.configure(api_key="AIzaSyBISxbNPJWaeXQwFhJqhZAwfbfn-eiGLiE")

app = FastAPI(title="V-ACT IRT Scoring API")

from database import engine
import models

# Lệnh này sẽ kiểm tra và tự động tạo bảng trên Supabase nếu chưa có
models.Base.metadata.create_all(bind=engine)

# --- ĐỊNH NGHĨA CẤU TRÚC DỮ LIỆU NHẬN VÀO TỪ FRONTEND ---
class ExamSubmission(BaseModel):
    a_params: List[float]      # Mảng độ phân biệt của 120 câu
    b_params: List[float]      # Mảng độ khó của 120 câu
    c_params: List[float]      # Mảng độ phỏng đoán của 120 câu
    responses: List[int]       # Mảng kết quả: 1 (Đúng), 0 (Sai)

class QuestionCreate(BaseModel):
    content: str
    option_a: str
    option_b: str
    option_c: str
    option_d: str
    correct_answer: str
    param_a: float
    param_b: float
    param_c: float

class TopicRequest(BaseModel):
    topic: str               # Chủ đề mong muốn (VD: Giải quyết vấn đề Hóa học)
    difficulty_level: str    # Mức độ (VD: Dễ, Trung bình, Khó)

# --- THUẬT TOÁN IRT ---
def irt_3pl_prob(theta, a, b, c):
    return c + (1 - c) / (1 + np.exp(-a * (theta - b)))

def negative_log_likelihood(theta, a, b, c, responses):
    p = irt_3pl_prob(theta, a, b, c)
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return -np.sum(responses * np.log(p) + (1 - responses) * np.log(1 - p))

# --- API ENDPOINT CHẤM ĐIỂM ---
@app.post("/calculate-score/")
async def calculate_vact_score(submission: ExamSubmission):
    # Chuyển dữ liệu sang numpy array
    a = np.array(submission.a_params)
    b = np.array(submission.b_params)
    c = np.array(submission.c_params)
    responses = np.array(submission.responses)
    
    # Tìm mức năng lực theta
    initial_theta = 0.0
    result = minimize(negative_log_likelihood, initial_theta, args=(a, b, c, responses), method='BFGS')
    theta_score = result.x[0]
    
    # Giả lập quy đổi điểm V-ACT (Ví dụ: Theta từ -3 đến 3 quy đổi ra 0 đến 1200)
    # Công thức này sẽ cần tinh chỉnh sau dựa trên dữ liệu thật
    vact_score = int(((theta_score + 3) / 6) * 1200)
    vact_score = max(0, min(1200, vact_score)) # Đảm bảo điểm nằm trong khoảng 0-1200
    
    return {
        "status": "success",
        "theta": round(theta_score, 3),
        "vact_score": vact_score
    }

@app.get("/")
async def root():
    return {"message": "Chào mừng đến với API chấm điểm V-ACT bằng IRT!"}

@app.post("/questions/")
async def create_question(question: QuestionCreate, db: Session = Depends(get_db)):
    # Biến dữ liệu nhận được thành một đối tượng Question trong Database
    db_question = models.Question(
        content=question.content,
        option_a=question.option_a,
        option_b=question.option_b,
        option_c=question.option_c,
        option_d=question.option_d,
        correct_answer=question.correct_answer,
        param_a=question.param_a,
        param_b=question.param_b,
        param_c=question.param_c
    )
    # Thêm vào database và lưu lại
    db.add(db_question)
    db.commit()
    db.refresh(db_question) # Cập nhật lại để lấy ID mới tạo
    
    return {"message": "Đã thêm câu hỏi thành công!", "question_id": db_question.id}
@app.post("/generate-question/")
async def generate_and_save_question(req: TopicRequest, db: Session = Depends(get_db)):
    # 1. Khởi động AI (Thật hoặc Ảo)
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"Tạo 1 câu hỏi trắc nghiệm V-ACT chủ đề {req.topic}, độ khó {req.difficulty_level} dưới dạng JSON chuẩn."
        response = model.generate_content(prompt)
        
        # Xử lý text từ AI
        clean_text = response.text.replace("```json", "").replace("```", "").strip()
        ai_data = json.loads(clean_text)
        source = "AI Thật (Gemini)"
    except Exception as e:
        # Nếu AI lỗi hoặc mạng chặn, dùng dữ liệu mẫu
        print(f"Đang dùng AI ảo do lỗi: {e}")
        ai_data = {
            "content": f"[Mẫu AI] Câu hỏi về {req.topic} đang xử lý...",
            "option_a": "Đáp án A", "option_b": "Đáp án B", 
            "option_c": "Đáp án C", "option_d": "Đáp án D",
            "correct_answer": "A",
            "param_a": 1.0, "param_b": 0.5, "param_c": 0.2
        }
        source = "AI ẢO (Dữ liệu mẫu)"

    # 2. Lưu vào Database (Bắt buộc phải nằm TRƯỚC return)
    db_question = models.Question(
        content=ai_data["content"],
        option_a=ai_data["option_a"],
        option_b=ai_data["option_b"],
        option_c=ai_data["option_c"],
        option_d=ai_data["option_d"],
        correct_answer=ai_data["correct_answer"],
        param_a=ai_data["param_a"],
        param_b=ai_data["param_b"],
        param_c=ai_data["param_c"]
    )
    db.add(db_question)
    db.commit()
    db.refresh(db_question)

    # 3. Trả về kết quả (Dòng cuối cùng của hàm)
    return {
        "status": "Thành công",
        "nguon": source,
        "question_id": db_question.id,
        "data": ai_data
    }
    
    # Gửi lệnh cho AI (Dòng 132 bạn đang bị lỗi ở đây)
    response = model.generate_content(prompt)
    
    try:
        clean_text = response.text.replace("```json", "").replace("```", "").strip()
        ai_data = json.loads(clean_text)
    except Exception as e:
        return {"error": "AI trả về sai định dạng", "details": str(e)}
        
    db_question = models.Question(
        content=ai_data["content"],
        option_a=ai_data["option_a"],
        option_b=ai_data["option_b"],
        option_c=ai_data["option_c"],
        option_d=ai_data["option_d"],
        correct_answer=ai_data["correct_answer"],
        param_a=ai_data["param_a"],
        param_b=ai_data["param_b"],
        param_c=ai_data["param_c"]
    )
    db.add(db_question)
    db.commit()
    db.refresh(db_question)
    
    return {"message": "Thành công!", "question_id": db_question.id}
    
    # Gửi lệnh cho AI
    response = model.generate_content(prompt)
    
    # Chuyển đổi phản hồi của AI từ dạng chữ (String) sang dạng Dữ liệu (Dictionary)
    try:
        # Xóa các ký tự thừa nếu AI lỡ tay bọc markdown
        clean_text = response.text.replace("```json", "").replace("```", "").strip()
        ai_data = json.loads(clean_text)
    except Exception as e:
        return {"error": "AI trả về sai định dạng, hãy thử lại!", "details": str(e), "raw": response.text}
        
    # Lưu câu hỏi AI vừa tạo vào Database của bạn
    db_question = models.Question(
        content=ai_data["content"],
        option_a=ai_data["option_a"],
        option_b=ai_data["option_b"],
        option_c=ai_data["option_c"],
        option_d=ai_data["option_d"],
        correct_answer=ai_data["correct_answer"],
        param_a=ai_data["param_a"],
        param_b=ai_data["param_b"],
        param_c=ai_data["param_c"]
    )
    db.add(db_question)
    db.commit()
    db.refresh(db_question)
    
    return {
        "message": "AI đã tạo và nạp thành công câu hỏi vào Ngân hàng đề thi!", 
        "question_id": db_question.id, 
        "ai_generated_data": ai_data
    }
    
@app.post("/add-and-classify/")
async def add_and_classify_question(question: dict, db: Session = Depends(get_db)):
    # Lấy gợi ý từ giao diện
    hint = question.get("difficulty_hint", "Trung bình")
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""
        Thẩm định câu hỏi V-ACT sau. Người ra đề đánh giá mức độ này là: {hint}.
        Câu hỏi: {question['content']}
        
        Dựa vào nội dung và gợi ý '{hint}', hãy gán 3 tham số IRT (a, b, c) chính xác dưới dạng JSON.
        Quy tắc: b phải khớp với mức {hint} (Dễ: b < -1, TB: b quanh 0, Khó: b > 1).
        """
        response = model.generate_content(prompt)
        clean_text = response.text.replace("```json", "").replace("```", "").strip()
        params = json.loads(clean_text)
    except:
        # Nếu AI lỗi mạng, gán mức mặc định (Độ khó trung bình)
        params = {"a": 1.0, "b": 0.0, "c": 0.2}

    db_question = models.Question(
        content=question.content,
        option_a=question.option_a,
        option_b=question.option_b,
        option_c=question.option_c,
        option_d=question.option_d,
        correct_answer=question.correct_answer,
        param_a=params["a"],
        param_b=params["b"],
        param_c=params["c"]
    )
    db.add(db_question)
    db.commit()
    db.refresh(db_question)
    
    return {"status": "Đã thêm và tự gán độ khó", "id": db_question.id, "params": params}
@app.get("/questions/")
async def get_all_questions(db: Session = Depends(get_db)):
    return db.query(models.Question).all()



@app.get("/admin-zone", response_class=HTMLResponse)
async def admin_page(key: str = None):
    if key != "vact_admin_2026":
        return "<h1>⛔ Truy cập bị từ chối!</h1>"
    
    # Giao diện Admin "nhúng" trực tiếp
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>V-ACT Admin Pro</title>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-slate-50 p-8 font-sans">
        <div class="max-w-6xl mx-auto">
            <h1 class="text-3xl font-bold text-blue-900 mb-8">🏦 Hệ thống Quản trị Ngân hàng đề V-ACT</h1>
            
            <div class="grid grid-cols-1 lg:grid-cols-3 gap-8">
                <div class="lg:col-span-1 space-y-6">
                    <div class="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
                        <h2 class="font-bold text-lg mb-4 text-slate-700 flex items-center">
                            <span class="mr-2">📝</span> Thêm câu hỏi mới
                        </h2>
                        <div class="space-y-4">
                            <textarea id="content" placeholder="Nội dung câu hỏi..." class="w-full p-3 border border-slate-200 rounded-xl h-32 focus:ring-2 focus:ring-blue-500 outline-none"></textarea>
                            <input id="optA" placeholder="Đáp án A" class="w-full p-2 border rounded-lg">
                            <input id="optB" placeholder="Đáp án B" class="w-full p-2 border rounded-lg">
                            <input id="optC" placeholder="Đáp án C" class="w-full p-2 border rounded-lg">
                            <input id="optD" placeholder="Đáp án D" class="w-full p-2 border rounded-lg">
                            
                            <div class="flex gap-2">
                                <select id="correct" class="w-1/2 p-2 border rounded-lg bg-slate-50">
                                    <option value="">Đáp án đúng</option>
                                    <option value="A">A</option><option value="B">B</option>
                                    <option value="C">C</option><option value="D">D</option>
                                </select>
                                <select id="level_hint" class="w-1/2 p-2 border rounded-lg bg-blue-50 text-blue-700 font-bold">
                                    <option value="Trung bình">Mức độ: TB</option>
                                    <option value="Dễ">Mức độ: Dễ</option>
                                    <option value="Khó">Mức độ: Khó</option>
                                    <option value="Cực khó">Mức độ: Cực khó</option>
                                </select>
                            </div>

                            <button onclick="save()" id="btnSave" class="w-full bg-blue-600 text-white py-3 rounded-xl font-bold hover:bg-blue-700 transition-all shadow-lg shadow-blue-200">
                                AI Phân loại & Lưu trữ
                            </button>
                        </div>
                    </div>
                </div>

                <div class="lg:col-span-2">
                    <div class="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden">
                        <table class="w-full text-left border-collapse">
                            <thead>
                                <tr class="bg-slate-50 text-slate-500 text-sm uppercase">
                                    <th class="p-4 border-b">ID</th>
                                    <th class="p-4 border-b">Nội dung</th>
                                    <th class="p-4 border-b text-center">Phân biệt (a)</th>
                                    <th class="p-4 border-b text-center">Độ khó (b)</th>
                                    <th class="p-4 border-b text-center">Đoán (c)</th>
                                </tr>
                            </thead>
                            <tbody id="list" class="text-slate-600"></tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>

        <script>
            async function load() {
                const res = await fetch('/questions/');
                const data = await res.json();
                document.getElementById('list').innerHTML = data.map(q => `
                    <tr class="hover:bg-slate-50 transition-colors border-b border-slate-100">
                        <td class="p-4 font-bold text-blue-600">#${q.id}</td>
                        <td class="p-4 text-sm">${q.content.substring(0, 80)}...</td>
                        <td class="p-4 text-center font-mono text-green-600">${q.param_a.toFixed(2)}</td>
                        <td class="p-4 text-center font-mono font-bold ${q.param_b > 1 ? 'text-red-500' : q.param_b < -1 ? 'text-blue-500' : 'text-slate-700'}">
                            ${q.param_b.toFixed(2)}
                        </td>
                        <td class="p-4 text-center font-mono text-orange-500">${q.param_c.toFixed(2)}</td>
                    </tr>
                `).join('');
            }

            async function save() {
                const btn = document.getElementById('btnSave');
                btn.innerText = "Đang phân tích...";
                btn.disabled = true;

                const payload = {
                    content: document.getElementById('content').value,
                    option_a: document.getElementById('optA').value,
                    option_b: document.getElementById('optB').value,
                    option_c: document.getElementById('optC').value,
                    option_d: document.getElementById('optD').value,
                    correct_answer: document.getElementById('correct').value,
                    // Chúng ta gửi kèm gợi ý mức độ cho AI
                    difficulty_hint: document.getElementById('level_hint').value 
                };

                const res = await fetch('/add-and-classify/', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(payload)
                });

                if (res.ok) {
                    alert("Đã lưu và AI đã gán chỉ số thành công!");
                    location.reload();
                } else {
                    alert("Có lỗi xảy ra!");
                    btn.innerText = "Thử lại";
                    btn.disabled = false;
                }
            }
            load();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/get-exam")
async def get_exam(db: Session = Depends(get_db)):
    # Lấy ngẫu nhiên hoặc lấy tất cả câu hỏi, nhưng ẩn đáp án đúng và tham số IRT
    questions = db.query(models.Question).all()
    # Chỉ trả về nội dung và các lựa chọn
    return [{
        "id": q.id,
        "content": q.content,
        "options": [q.option_a, q.option_b, q.option_c, q.option_d]
    } for q in questions]
    
# --- GIAO DIỆN PHÒNG THI DÀNH CHO HỌC VIÊN ---

@app.get("/api/exam")
async def get_exam_questions(db: Session = Depends(get_db)):
    # Lấy ngẫu nhiên 10 câu hỏi từ ngân hàng đề (hoặc lấy tất cả)
    questions = db.query(models.Question).limit(10).all()
    
    # RẤT QUAN TRỌNG: Chỉ gửi id, nội dung và 4 đáp án. TUYỆT ĐỐI ẨN correct_answer và a, b, c
    safe_questions = []
    for q in questions:
        safe_questions.append({
            "id": q.id,
            "content": q.content,
            "options": {
                "A": q.option_a,
                "B": q.option_b,
                "C": q.option_c,
                "D": q.option_d
            }
        })
    return safe_questions

# Định nghĩa cấu trúc dữ liệu khi học sinh nộp bài
class ExamSubmission(BaseModel):
    student_name: str
    answers: Dict[int, str]

# --- ĐỂ 2 HÀM TOÁN HỌC Ở NGOÀI, BÊN TRÊN API ---
def probability_3pl(theta, a, b, c):
    """Tính xác suất trả lời đúng theo mô hình 3PL"""
    z = np.clip(-a * (theta - b), -700, 700)
    return c + (1 - c) / (1 + np.exp(z))

def estimate_theta(responses, item_params):
    def negative_log_likelihood(theta):
        nll = 0
        for i, (a, b, c) in enumerate(item_params):
            p = probability_3pl(theta[0], a, b, c)
            p = np.clip(p, 1e-10, 1 - 1e-10)
            y = responses[i]
            nll -= y * np.log(p) + (1 - y) * np.log(1 - p)
        return nll

    initial_guess = [0.0]
    bounds = [(-4.0, 4.0)]
    result = minimize(negative_log_likelihood, initial_guess, bounds=bounds, method='L-BFGS-B')
    return result.x[0]

# --- API NỘP BÀI NẰM Ở DƯỚI CÙNG ---
@app.post("/api/submit")
async def submit_exam(submission: ExamSubmission, db: Session = Depends(get_db)):
    correct_count = 0
    total_questions = len(submission.answers)
    
    responses = []
    item_params = []
    
    # 1. Chấm điểm & Thu thập tham số
    for q_id_str, ans in submission.answers.items():
        q_id = int(q_id_str)
        q_db = db.query(models.Question).filter(models.Question.id == q_id).first()
        
        if q_db:
            is_correct = 1 if q_db.correct_answer == ans else 0
            correct_count += is_correct
            responses.append(is_correct)
            item_params.append((q_db.param_a, q_db.param_b, q_db.param_c))
            
    # 2. Chạy thuật toán IRT để tìm Theta
    estimated_theta = 0.0
    if total_questions > 0:
        if correct_count == total_questions:
            estimated_theta = 4.0 
        elif correct_count == 0:
            estimated_theta = -4.0
        else:
            estimated_theta = estimate_theta(responses, item_params)
            
    # ------ ĐOẠN CODE THÊM MỚI: LƯU VÀO DATABASE ------
    final_theta = round(float(estimated_theta), 2)
    
    new_submission = models.Submission(
        student_name=submission.student_name,
        raw_score=f"{correct_count}/{total_questions}",
        theta_score=final_theta
    )
    db.add(new_submission)
    db.commit()
    db.refresh(new_submission)
    # ----------------------------------------------------

    return {
        "message": "Nộp bài thành công!",
        "student": submission.student_name,
        "raw_score": f"{correct_count}/{total_questions}",
        "estimated_theta": final_theta,
        "submission_id": new_submission.id
    }
    
@app.get("/api/dashboard-stats")
async def get_dashboard_stats(db: Session = Depends(get_db)):
    # Lấy toàn bộ bài nộp, sắp xếp mới nhất lên đầu
    submissions = db.query(models.Submission).order_by(models.Submission.submitted_at.desc()).all()
    total = len(submissions)
    
    if total == 0:
        return {"total": 0, "avg_theta": 0.0, "distribution": [0]*7, "recent": []}

    # 1. Tính Năng lực trung bình của toàn bộ học viên
    avg_theta = sum(s.theta_score for s in submissions) / total
    
    # 2. Phân bố điểm cho biểu đồ hình chuông IRT (Area Chart)
    # Các mức: -3, -2, -1, 0, +1, +2, +3
    dist = [0, 0, 0, 0, 0, 0, 0]
    for s in submissions:
        t = s.theta_score
        if t <= -2.5: dist[0] += 1
        elif t <= -1.5: dist[1] += 1
        elif t <= -0.5: dist[2] += 1
        elif t <= 0.5: dist[3] += 1
        elif t <= 1.5: dist[4] += 1
        elif t <= 2.5: dist[5] += 1
        else: dist[6] += 1

    # 3. Lấy 5 bài nộp gần nhất cho Bảng dữ liệu
    recent = []
    for s in submissions[:5]:
        # Xếp loại thông minh dựa trên Theta
        if s.theta_score >= 1.5: rank = ("Xuất sắc", "bg-emerald-100 text-emerald-700")
        elif s.theta_score >= 0.5: rank = ("Khá giỏi", "bg-blue-100 text-blue-700")
        elif s.theta_score >= -0.5: rank = ("Trung bình", "bg-slate-100 text-slate-700")
        else: rank = ("Cần cố gắng", "bg-orange-100 text-orange-700")
        
        recent.append({
            "name": s.student_name,
            "code": s.exam_code,
            "raw": s.raw_score,
            "theta": s.theta_score,
            "rank_name": rank[0],
            "rank_css": rank[1]
        })

    return {
        "total": total,
        "avg_theta": round(avg_theta, 2),
        "distribution": dist,
        "recent": recent
    }
    
# --- ROUTE CHO GIAO DIỆN WEB ---

@app.get("/exam", response_class=HTMLResponse)
async def serve_exam_page():
    # Lấy đường dẫn chuẩn của file exam.html nằm cùng thư mục
    base_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_path, "exam.html")
    return FileResponse(file_path)

@app.get("/dashboard", response_class=HTMLResponse)
async def serve_dashboard_page():
    base_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_path, "user_stats.html")
    return FileResponse(file_path)

