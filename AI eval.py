import streamlit as st
import pandas as pd
import tempfile
import os
import time
import google.generativeai as genai

# ==========================================
# 1. 페이지 설정 및 에듀테크 플랫폼 스타일 CSS
# ==========================================
st.set_page_config(page_title="AI 채점 대시보드", page_icon="📝", layout="wide")

# 첨부된 이미지(왓퀴즈) 스타일을 모방한 커스텀 CSS
st.markdown("""
    <style>
    /* 전체 배경색 (연한 회색) */
    .stApp { background-color: #f5f7f9; }
    
    /* 폰트 설정 */
    * { font-family: 'Pretendard', 'Noto Sans KR', sans-serif; }
    
    /* 메인 타이틀 */
    .main-title { font-size: 28px; font-weight: 800; color: #111827; margin-bottom: 5px; }
    .sub-title { font-size: 14px; color: #6b7280; margin-bottom: 30px; }
    
    /* 하얀색 카드 UI (본문 영역) */
    .card { background-color: #ffffff; border-radius: 12px; padding: 25px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); margin-bottom: 20px; border: 1px solid #e5e7eb; }
    
    /* 버튼 스타일 (블루 포인트) */
    .stButton>button { background-color: #3b82f6; color: white; border-radius: 6px; font-weight: 600; padding: 0.5rem 1rem; border: none; transition: all 0.2s; }
    .stButton>button:hover { background-color: #2563eb; box-shadow: 0 4px 6px rgba(59, 130, 246, 0.2); }
    
    /* 결과 출력 박스 */
    .result-box { background-color: #f8fafc; padding: 20px; border-radius: 8px; border-left: 4px solid #3b82f6; font-size: 15px; line-height: 1.6; color: #334155; margin-top: 15px; }
    
    /* 탭(Tabs) 스타일 튜닝 */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { background-color: transparent; border-radius: 6px; padding: 8px 16px; font-weight: 600; color: #6b7280; border: 1px solid #e5e7eb; }
    .stTabs [aria-selected="true"] { background-color: #3b82f6; color: white; border-color: #3b82f6; }
    
    /* 사이드바 메뉴 텍스트 스타일 */
    .sidebar-menu { font-size: 15px; font-weight: 500; color: #4b5563; padding: 10px 0; border-bottom: 1px solid #e5e7eb; cursor: pointer; }
    .sidebar-menu.active { color: #3b82f6; font-weight: 700; border-right: 3px solid #3b82f6; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 교육학적 세부 루브릭 데이터베이스 (확장)
# ==========================================
RUBRIC_DB = {
    "고등_화학_실험보고서": """
    [영역 1: 이론적 배경 및 가설] 우수(25-30): 핵심 화학 원리 정확히 설명, 논리적 가설 설정 / 보통(15-24): 원리 설명 일부 누락, 가설 연결성 부족 / 미흡(0-14): 원리 부정확, 가설 없음
    [영역 2: 결과 분석 및 해석] 우수(35-40): 데이터 과학적 변환, 독립/종속 변인 심층 분석 / 보통(20-34): 표면적 데이터 제시, 원리 해석 부족 / 미흡(0-19): 사실 나열, 해석 오류
    [영역 3: 오차 논의] 우수(25-30): 오차 원인을 화학적/환경적 요인으로 다각도 분석, 구체적 개선안 / 보통(15-24): 단순 계산 실수로 언급 / 미흡(0-14): 오차 분석 없음
    """,
    "고등_과학_논술평가": """
    [영역 1: 과학적 사실의 정확성] 우수(35-40): 교과 개념을 정확히 적용하여 논제 해결 / 보통(20-34): 일부 개념 오개념 존재 / 미흡(0-19): 과학적 근거 부족
    [영역 2: 논리적 전개] 우수(25-30): 서론-본론-결론이 명확하고 문장 간 연결이 매끄러움 / 보통(15-24): 흐름이 다소 끊어짐 / 미흡(0-14): 주장만 있고 근거가 불명확함
    [영역 3: 문제 해결 및 창의성] 우수(25-30): 사회적/환경적 문제와 연계하여 독창적 대안 제시 / 보통(15-24): 일반적이고 뻔한 대안 제시 / 미흡(0-14): 대안 제시 없음
    """,
    "고등_AI융합_산출물": """
    [영역 1: AI 도구 활용의 적절성] 우수(35-40): 문제 해결을 위해 적절한 AI 도구를 선택하고 한계를 보완하여 사용 / 보통(20-34): AI를 사용했으나 산출물과의 연관성 다소 부족 / 미흡(0-19): 부적절한 활용
    [영역 2: 프롬프트 엔지니어링] 우수(25-30): 구체적이고 체계적인 프롬프트를 설계하여 원하는 결과를 도출 / 보통(15-24): 단답형 지시어 위주 사용 / 미흡(0-14): 프롬프트 설계 기록 없음
    [영역 3: 융합적 사고] 우수(25-30): 타 교과(화학 등)의 지식과 AI 기술을 유기적으로 융합하여 결론 도출 / 보통(15-24): 융합적 시도는 있으나 깊이가 얕음 / 미흡(0-14): 단일 교과 수준에 머무름
    """
}

# ==========================================
# 3. AI 엔진 기능 정의
# ==========================================
def evaluate_with_gemini(api_key, text_content=None, uploaded_file_path=None, rubric=""):
    """학생 과제를 루브릭에 따라 채점하는 함수"""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-3-flash-preview")
        
        prompt = f"""
        너는 고등학생을 지도하는 전문적이고 따뜻한 교사야. 
        아래 [교육학적 세부 루브릭]을 엄격하게 적용하여 학생의 과제를 채점하고 피드백을 줘.
        
        1. 각 영역별 점수와 성취 수준(우수/보통/미흡)을 명시하고, 루브릭의 기준을 바탕으로 '왜 이 점수를 주었는지' 이유를 적어줘.
        2. 마지막에는 학생의 성장을 돕는 따뜻한 종합 피드백(보완할 점 포함)을 3문장 정도로 작성해.
        3. 마크다운 포맷(JSON 등) 없이 깔끔한 텍스트로만 출력해.

        [교육학적 세부 루브릭]
        {rubric}
        """

        contents = [prompt]
        if text_content: contents.append(f"\n[학생 과제 내용]\n{text_content}")
        if uploaded_file_path:
            uploaded_file = genai.upload_file(path=uploaded_file_path)
            contents.append(uploaded_file)
            
        response = model.generate_content(contents, generation_config={"temperature": 0.2})
        if uploaded_file_path: genai.delete_file(uploaded_file.name)
        return response.text 
    except Exception as e:
        return f"⚠️ 오류 발생 (API 키를 확인하세요): {str(e)}"

def generate_rubric_with_gemini(api_key, topic):
    """교사가 입력한 주제로 새로운 루브릭을 생성해주는 함수"""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-3-flash-preview") # 루브릭 생성은 조금 더 똑똑한 Pro 모델 사용
        
        prompt = f"""
        당신은 교육평가 전문가입니다. 교사가 제시한 다음 과제 주제를 바탕으로, 학생을 평가하기 위한 '분석적 루브릭(Analytic Rubric)'을 개발해주세요.
        
        [과제 주제/성취기준]: {topic}
        
        [출력 조건]
        1. 평가 영역을 3~4가지로 나누세요. (예: 이론적 배경, 탐구 과정, 결론 도출 등)
        2. 각 영역별로 100점 만점 기준의 배점을 할당하세요.
        3. 각 영역은 '우수', '보통', '미흡'의 3단계 성취 수준으로 나누고, 어떤 상태일 때 해당 수준인지 구체적인 학생의 행동 지표(서술형)로 작성해주세요.
        4. 그대로 복사해서 시스템에 넣을 수 있도록 깔끔하게 정리해주세요.
        """
        response = model.generate_content(prompt, generation_config={"temperature": 0.4})
        return response.text
    except Exception as e:
        return f"⚠️ 루브릭 생성 오류: {str(e)}"


# ==========================================
# 4. 화면 구성 (대시보드 UI)
# ==========================================
# --- 사이드바 (메뉴 디자인) ---
with st.sidebar:
    st.markdown("<h2 style='color:#3b82f6; font-weight:800;'>🐘 왓퀴즈(Edu)</h2>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("<div class='sidebar-menu'>💬 AI 채팅</div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-menu'>📝 AI 생기부 생성</div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-menu active'>✅ AI 서술형 채점</div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-menu'>📊 AI 평가계획서</div>", unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("🔒 **시스템 설정**")
    api_key_input = st.text_input("Google API Key 입력", type="password", placeholder="AIzaSy...")

# --- 메인 화면 ---
st.markdown("<div class='main-title'>AI 서술형 채점 및 루브릭 관리</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>밤새던 채점 업무, 이제 3분이면 끝 · 교육학적 루브릭 기반 평가</div>", unsafe_allow_html=True)

# 탭을 마치 우측 상단의 버튼처럼 활용
tab1, tab2, tab3 = st.tabs(["📝 단일 채점 (파일/텍스트)", "📁 학급 전체 채점 (엑셀)", "✨ AI 루브릭 설계소"])

# ------------------------------------------
# [탭 1] 단일 채점 
# ------------------------------------------
with tab1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("#### ⚙️ 평가 기준 선택")
    selected_category = st.selectbox("적용할 루브릭(평가 기준)을 선택하세요.", list(RUBRIC_DB.keys()))
    
    with st.expander("📌 선택된 루브릭 상세 보기"):
        st.write(RUBRIC_DB[selected_category])
    
    st.markdown("---")
    st.markdown("#### 📄 학생 과제 입력")
    upload_type = st.radio("입력 방식을 선택하세요.", ["직접 텍스트 입력", "파일 업로드 (PDF/사진)"], horizontal=True)
    
    student_text = None
    uploaded_file_path = None
    tmp_path = None
    
    if upload_type == "직접 텍스트 입력":
        student_text = st.text_area("학생이 작성한 글을 붙여넣으세요.", height=150)
    else:
        uploaded_file = st.file_uploader("학생의 과제 파일(PDF, JPG, PNG)을 업로드하세요.", type=['pdf', 'png', 'jpg', 'jpeg'])
        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
                uploaded_file_path = tmp_path

    if st.button("🚀 AI 채점 시작하기", key="btn_single"):
        if not api_key_input: st.error("왼쪽 사이드바에서 API Key를 입력해주세요.")
        elif not student_text and not uploaded_file_path: st.warning("과제 내용을 입력하거나 파일을 업로드해주세요.")
        else:
            with st.spinner("AI가 교사의 시선으로 채점을 진행하고 있습니다..."):
                result = evaluate_with_gemini(api_key_input, text_content=student_text, uploaded_file_path=uploaded_file_path, rubric=RUBRIC_DB[selected_category])
                st.markdown(f'<div class="result-box"><b>[AI 채점 결과]</b><br><br>{result}</div>', unsafe_allow_html=True)
            if tmp_path: os.remove(tmp_path) # 임시 파일 삭제
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------
# [탭 2] 학급 전체 채점 (엑셀)
# ------------------------------------------
with tab2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("#### 📊 학급 일괄 채점 (CSV 업로드)")
    st.markdown("필수 열 이름: **이름**, **과제내용**")
    
    batch_category = st.selectbox("일괄 채점에 적용할 루브릭 선택", list(RUBRIC_DB.keys()), key="batch_cat")
    uploaded_csv = st.file_uploader("CSV 파일 선택", type=['csv'])
    
    if uploaded_csv and st.button("🚀 학급 전체 일괄 채점 시작", key="btn_batch"):
        if not api_key_input: 
            st.error("왼쪽 사이드바에서 API Key를 입력해주세요.")
        else:
            try:
                df = pd.read_csv(uploaded_csv, encoding='utf-8')
            except UnicodeDecodeError:
                uploaded_csv.seek(0)
                df = pd.read_csv(uploaded_csv, encoding='cp949')

            if '과제내용' not in df.columns: 
                st.error("'과제내용' 열을 찾을 수 없습니다. 파일 양식을 확인해주세요.")
            else:
                progress_text = "채점 진행 중... 잠시만 기다려주세요."
                my_bar = st.progress(0, text=progress_text)
                results, total = [], len(df)
                
                for i, row in df.iterrows():
                    res = evaluate_with_gemini(api_key_input, text_content=str(row['과제내용']), rubric=RUBRIC_DB[batch_category])
                    results.append(res)
                    my_bar.progress(int(((i + 1) / total) * 100), text=f"채점 중... ({i+1}/{total}명 완료)")
                    time.sleep(1)
                    
                df['AI_피드백'] = results
                st.success("✅ 모든 학생의 채점이 완료되었습니다!")
                st.dataframe(df)
                st.download_button("📥 평가 결과 다운로드 (CSV)", df.to_csv(index=False).encode('utf-8-sig'), "일괄평가결과.csv", "text/csv")
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------
# [탭 3] AI 루브릭 설계소 (신규 기능)
# ------------------------------------------
with tab3:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("#### ✨ AI 루브릭 자동 생성기")
    st.markdown("선생님께서 구상하신 수행평가나 과제의 **주제 및 성취기준**을 입력하시면, AI가 전문적인 3단계(우수/보통/미흡) 루브릭을 설계해 드립니다.")
    
    rubric_topic = st.text_input("수행평가 주제 입력 (예: 중화적정 실험을 통한 아세트산 농도 구하기)")
    
    if st.button("🪄 루브릭 설계 요청하기", key="btn_rubric"):
        if not api_key_input: 
            st.error("왼쪽 사이드바에서 API Key를 입력해주세요.")
        elif not rubric_topic:
            st.warning("과제 주제를 입력해주세요.")
        else:
            with st.spinner("교육평가 전문가 AI가 루브릭을 설계하고 있습니다..."):
                generated_rubric = generate_rubric_with_gemini(api_key_input, rubric_topic)
                st.markdown(f'<div class="result-box"><b>[생성된 루브릭 제안]</b><br><br>{generated_rubric}</div>', unsafe_allow_html=True)
                st.info("💡 위 생성된 루브릭을 복사하여 파이썬 코드의 `RUBRIC_DB` 딕셔너리에 추가해두시면, 언제든 채점 기준으로 사용할 수 있습니다.")
    st.markdown("</div>", unsafe_allow_html=True)