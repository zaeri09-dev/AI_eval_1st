import streamlit as st
import pandas as pd
import tempfile
import os
import time
import google.generativeai as genai

# ==========================================
# 1. 페이지 설정 및 세련된 대시보드 CSS
# ==========================================
st.set_page_config(page_title="교사용 AI 채점 대시보드", page_icon="🏫", layout="wide")

st.markdown("""
    <style>
    /* 전체 배경색 */
    .stApp { background-color: #f8fafc; }
    * { font-family: 'Pretendard', 'Noto Sans KR', sans-serif; }
    
    /* 메인 타이틀 */
    .main-title { font-size: 28px; font-weight: 800; color: #1e293b; margin-bottom: 5px; }
    .sub-title { font-size: 14.5px; color: #64748b; margin-bottom: 30px; }
    
    /* 카드 UI */
    .card { background-color: #ffffff; border-radius: 14px; padding: 25px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); margin-bottom: 20px; border: 1px solid #e2e8f0; }
    
    /* 버튼 스타일 */
    .stButton>button { background-color: #3b82f6; color: white; border-radius: 8px; font-weight: 600; padding: 0.5rem 1rem; border: none; transition: all 0.2s; }
    .stButton>button:hover { background-color: #2563eb; transform: translateY(-1px); }
    
    /* 결과 출력 박스 */
    .result-box { background-color: #f1f5f9; padding: 25px; border-radius: 12px; border-left: 5px solid #3b82f6; font-size: 15px; line-height: 1.7; color: #334155; margin-top: 15px; }
    
    /* ========================================== */
    /* 🎨 사이드바 메뉴 디자인 */
    /* ========================================== */
    [data-testid="stSidebar"] { background-color: #ffffff; }
    [data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] { gap: 0.1rem; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label > div:first-child { display: none; }
    
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        padding: 10px 14px;
        border-radius: 10px;
        cursor: pointer;
        transition: all 0.2s ease;
        margin-bottom: 4px;
    }
    
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background-color: #f1f5f9;
    }
    
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
        background-color: #eff6ff; 
    }
    
    /* 기본 텍스트 스타일 (글자 깨짐 방지 적용) */
    [data-testid="stSidebar"] [data-testid="stRadio"] label p {
        font-size: 14.5px; /* 글자 크기를 살짝 줄여서 한 줄에 맞춤 */
        font-weight: 600;
        color: #475569;
        white-space: nowrap; /* 글자가 밑으로 떨어지지 않게 강제 한 줄 처리 */
    }
    
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p {
        color: #2563eb !important;
        font-weight: 700;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 교육학적 세부 루브릭 데이터베이스 (가독성 대폭 개선)
# ==========================================
# 💡 [수정 필요] 상황에 맞게 루브릭의 내용과 점수를 수정하세요. (AI가 잘 읽도록 별표(*)와 줄바꿈을 유지해 주시면 좋습니다.)
RUBRIC_DB = {
    "고등_화학_실험보고서": """
**[영역 1: 이론적 배경 및 가설]**
* **우수(25-30점)**: 핵심 화학 원리를 정확히 설명하고, 이를 바탕으로 논리적인 가설을 설정함
* **보통(15-24점)**: 원리 설명이 일부 누락되었거나, 가설과의 연결성이 다소 부족함
* **미흡(0-14점)**: 원리 설명이 부정확하거나 가설이 없음

**[영역 2: 결과 분석 및 해석]**
* **우수(35-40점)**: 데이터를 과학적으로 변환하고, 독립/종속 변인의 관계를 심층적으로 분석함
* **보통(20-34점)**: 표면적인 데이터만 제시하였으며, 원리를 통한 해석이 부족함
* **미흡(0-19점)**: 단순 사실 나열에 그치거나 해석에 오류가 있음

**[영역 3: 오차 논의]**
* **우수(25-30점)**: 오차 원인을 화학적/환경적 요인으로 다각도 분석하고 구체적 개선안을 제시함
* **보통(15-24점)**: 오차 원인을 단순 계산이나 측정 실수 정도로만 언급함
* **미흡(0-14점)**: 오차 분석이 전혀 없음
    """,
    
    "고등_과학_논술평가": """
**[영역 1: 과학적 사실의 정확성]**
* **우수(35-40점)**: 교과 개념을 정확히 적용하여 논제를 완벽히 해결함
* **보통(20-34점)**: 개념 적용을 시도했으나 일부 오개념이 존재함
* **미흡(0-19점)**: 과학적 근거가 매우 부족함

**[영역 2: 논리적 전개]**
* **우수(25-30점)**: 서론-본론-결론이 명확하고 문장 간 연결이 매끄러움
* **보통(15-24점)**: 흐름이 다소 끊어지거나 논리적 비약이 있음
* **미흡(0-14점)**: 주장만 있고 이를 뒷받침하는 근거가 불명확함

**[영역 3: 문제 해결 및 창의성]**
* **우수(25-30점)**: 사회적/환경적 문제와 연계하여 독창적인 대안을 제시함
* **보통(15-24점)**: 일반적이고 누구나 생각할 수 있는 뻔한 대안을 제시함
* **미흡(0-14점)**: 문제에 대한 대안 제시가 전혀 없음
    """
}

# ==========================================
# 3. AI 엔진 (gemini-3-flash-preview)
# ==========================================
def evaluate_with_gemini(api_key, text_content=None, uploaded_file_path=None, rubric=""):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-3-flash-preview")
        
        prompt = f"""
        너는 고등학생을 지도하는 전문적이고 따뜻한 교사야. 
        아래 [교육학적 세부 루브릭]을 엄격하게 적용하여 학생의 과제를 채점하고 피드백을 줘.
        
        1. 각 영역별 점수와 성취 수준(우수/보통/미흡)을 명시하고, '왜 이 점수를 주었는지' 이유를 적어줘.
        2. 마지막에는 성장을 돕는 따뜻한 종합 피드백을 3문장 정도로 작성해.
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
        return f"⚠️ 오류 발생 (API 키 확인): {str(e)}"

def generate_rubric_with_gemini(api_key, topic):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-3-flash-preview") 
        
        prompt = f"""
        당신은 교육평가 전문가입니다. 교사가 제시한 과제 주제를 바탕으로 '분석적 루브릭'을 개발해주세요.
        
        [과제 주제/성취기준]: {topic}
        [출력 조건]
        1. 평가 영역을 3~4가지로 나누고 100점 만점으로 배점.
        2. '우수', '보통', '미흡'의 3단계로 나누어 학생의 구체적인 행동 지표 서술.
        3. 그대로 복사할 수 있게 깔끔하게 정리.
        """
        response = model.generate_content(prompt, generation_config={"temperature": 0.4})
        return response.text
    except Exception as e:
        return f"⚠️ 루브릭 생성 오류: {str(e)}"

# ==========================================
# 4. 화면 구성
# ==========================================
with st.sidebar:
    st.markdown("<h2 style='color:#2563eb; font-weight:800; margin-bottom: 20px;'>🏫 AI 평가 보조</h2>", unsafe_allow_html=True)
    
    main_menu = st.radio(
        "메인메뉴", 
        ["📋 AI 서술형 채점", "✨ AI 루브릭 설계"], 
        label_visibility="collapsed"
    )
    
    sub_menu = None
    if main_menu == "📋 AI 서술형 채점":
        st.markdown("<div style='margin-top: -10px;'></div>", unsafe_allow_html=True)
        sub_menu = st.radio(
            "하위메뉴", 
            ["ㅤㅤ📄 단일 채점 (텍스트/파일)", "ㅤㅤ📁 학급 전체 채점 (엑셀)"], 
            label_visibility="collapsed"
        )
        
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.markdown("🔒 **시스템 설정**")
    api_key_input = st.text_input("Google API Key 입력", type="password", placeholder="AIzaSy...")

st.markdown(f"<div class='main-title'>{main_menu.replace('📋 ', '').replace('✨ ', '')}</div>", unsafe_allow_html=True)

# ------------------------------------------
# 메뉴 1: AI 서술형 채점 
# ------------------------------------------
if main_menu == "📋 AI 서술형 채점":
    st.markdown("<div class='sub-title'>교육학적 루브릭 기반 학생 과제 심층 평가</div>", unsafe_allow_html=True)
    
    if sub_menu == "ㅤㅤ📄 단일 채점 (텍스트/파일)":
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("#### ⚙️ 평가 기준 선택")
        selected_category = st.selectbox("적용할 루브릭(평가 기준)을 선택하세요.", list(RUBRIC_DB.keys()))
        
        # 화면에 루브릭이 예쁘게 줄바꿈 되어 보이도록 마크다운 처리 적용
        with st.expander("📌 선택된 루브릭 상세 보기"):
            st.markdown(RUBRIC_DB[selected_category])
        
        st.markdown("---")
        st.markdown("#### 📄 학생 과제 입력")
        upload_type = st.radio("입력 방식을 선택하세요.", ["직접 텍스트 입력", "파일 업로드 (PDF/사진)"], horizontal=True)
        
        student_text, uploaded_file_path, tmp_path = None, None, None
        
        if upload_type == "직접 텍스트 입력":
            student_text = st.text_area("학생이 작성한 글을 붙여넣으세요.", height=150)
        else:
            uploaded_file = st.file_uploader("학생의 과제 파일(PDF, JPG, PNG) 업로드", type=['pdf', 'png', 'jpg', 'jpeg'])
            if uploaded_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                    uploaded_file_path = tmp_path

        if st.button("🚀 단일 채점 시작", key="btn_single"):
            if not api_key_input: st.error("👈 사이드바에서 API Key를 입력해주세요.")
            elif not student_text and not uploaded_file_path: st.warning("과제 내용을 입력하거나 업로드해주세요.")
            else:
                with st.spinner("AI가 채점을 진행 중입니다..."):
                    result = evaluate_with_gemini(api_key_input, text_content=student_text, uploaded_file_path=uploaded_file_path, rubric=RUBRIC_DB[selected_category])
                    st.markdown(f'<div class="result-box"><b>[AI 채점 결과]</b><br><br>{result}</div>', unsafe_allow_html=True)
                if tmp_path: os.remove(tmp_path) 
        st.markdown("</div>", unsafe_allow_html=True)

    elif sub_menu == "ㅤㅤ📁 학급 전체 채점 (엑셀)":
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("#### 📊 학급 일괄 채점 (CSV 업로드)")
        st.markdown("필수 열 이름: **이름**, **과제내용**")
        
        batch_category = st.selectbox("일괄 채점에 적용할 루브릭 선택", list(RUBRIC_DB.keys()), key="batch_cat")
        
        with st.expander("📌 선택된 루브릭 상세 보기"):
            st.markdown(RUBRIC_DB[batch_category])
            
        uploaded_csv = st.file_uploader("CSV 파일 선택", type=['csv'])
        
        if uploaded_csv and st.button("🚀 일괄 채점 시작", key="btn_batch"):
            if not api_key_input: st.error("👈 사이드바에서 API Key를 입력해주세요.")
            else:
                # 💡 [수정 필요] 윈도우 엑셀의 한글 깨짐 오류 방지 로직입니다. (그대로 두시면 됩니다.)
                try: df = pd.read_csv(uploaded_csv, encoding='utf-8')
                except UnicodeDecodeError:
                    uploaded_csv.seek(0)
                    df = pd.read_csv(uploaded_csv, encoding='cp949')

                if '과제내용' not in df.columns: st.error("'과제내용' 열을 찾을 수 없습니다.")
                else:
                    my_bar = st.progress(0, text="채점 진행 중...")
                    results, total = [], len(df)
                    
                    for i, row in df.iterrows():
                        res = evaluate_with_gemini(api_key_input, text_content=str(row['과제내용']), rubric=RUBRIC_DB[batch_category])
                        results.append(res)
                        my_bar.progress(int(((i + 1) / total) * 100), text=f"채점 중... ({i+1}/{total}명 완료)")
                        time.sleep(1)
                        
                    df['AI_피드백'] = results
                    st.success("✅ 채점 완료!")
                    st.dataframe(df)
                    st.download_button("📥 평가 결과 다운로드 (CSV)", df.to_csv(index=False).encode('utf-8-sig'), "일괄평가결과.csv", "text/csv")
        st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------
# 메뉴 2: AI 루브릭 설계 
# ------------------------------------------
elif main_menu == "✨ AI 루브릭 설계":
    st.markdown("<div class='sub-title'>수행평가 주제 기반 3단계 성취수준 자동 생성기</div>", unsafe_allow_html=True)
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    
    rubric_topic = st.text_input("수행평가 주제 입력 (예: 르샤틀리에 원리를 이용한 화학 평형 이동 실험)")
    
    if st.button("🪄 루브릭 설계 요청하기", key="btn_rubric"):
        if not api_key_input: st.error("👈 사이드바에서 API Key를 입력해주세요.")
        elif not rubric_topic: st.warning("과제 주제를 입력해주세요.")
        else:
            with st.spinner("교육평가 전문가 AI가 루브릭을 설계 중입니다..."):
                generated_rubric = generate_rubric_with_gemini(api_key_input, rubric_topic)
                st.markdown(f'<div class="result-box"><b>[생성된 루브릭 제안]</b><br><br>{generated_rubric}</div>', unsafe_allow_html=True)
                st.info("💡 생성된 루브릭을 파이썬 코드의 `RUBRIC_DB`에 복사해 두시면 계속 사용할 수 있습니다.")
    st.markdown("</div>", unsafe_allow_html=True)