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
    .stApp { background-color: #f8fafc; }
    * { font-family: 'Pretendard', 'Noto Sans KR', sans-serif; }
    .main-title { font-size: 28px; font-weight: 800; color: #1e293b; margin-bottom: 5px; }
    .sub-title { font-size: 14.5px; color: #64748b; margin-bottom: 30px; }
    .card { background-color: #ffffff; border-radius: 14px; padding: 25px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); margin-bottom: 20px; border: 1px solid #e2e8f0; }
    .stButton>button { background-color: #3b82f6; color: white; border-radius: 8px; font-weight: 600; padding: 0.5rem 1rem; border: none; transition: all 0.2s; }
    .stButton>button:hover { background-color: #2563eb; transform: translateY(-1px); }
    .result-box { background-color: #f1f5f9; padding: 25px; border-radius: 12px; border-left: 5px solid #3b82f6; font-size: 15px; line-height: 1.7; color: #334155; margin-top: 15px; }
    
    /* 사이드바 메뉴 디자인 */
    [data-testid="stSidebar"] { background-color: #ffffff; }
    [data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] { gap: 0.1rem; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label > div:first-child { display: none; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label { padding: 10px 14px; border-radius: 10px; cursor: pointer; transition: all 0.2s ease; margin-bottom: 4px; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover { background-color: #f1f5f9; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) { background-color: #eff6ff; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label p { font-size: 14.5px; font-weight: 600; color: #475569; white-space: nowrap; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p { color: #2563eb !important; font-weight: 700; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 루브릭 바인더(Session State) 초기화
# ==========================================
# 앱이 시작될 때 기본 루브릭을 바인더에 넣어둡니다.
if 'rubric_binder' not in st.session_state:
    st.session_state.rubric_binder = {
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

# 메뉴 확장 상태 관리
if 'menu_expanded' not in st.session_state:
    st.session_state.menu_expanded = True
if 'current_page' not in st.session_state:
    st.session_state.current_page = "ㅤㅤ📄 단일 채점 (텍스트/파일)"

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
        1. 각 영역별 점수와 성취 수준(우수/보통/미흡)을 명시하고 이유를 적어줘.
        2. 마지막에는 따뜻한 종합 피드백을 3문장 작성해.
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
        return f"⚠️ 오류 발생: {str(e)}"

def generate_rubric_with_gemini(api_key, topic):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-3-flash-preview") 
        prompt = f"교육평가 전문가로서 '{topic}'에 대한 3단계(우수/보통/미흡) 분석적 루브릭을 개발해줘."
        response = model.generate_content(prompt, generation_config={"temperature": 0.4})
        return response.text
    except Exception as e:
        return f"⚠️ 오류: {str(e)}"

# ==========================================
# 4. 화면 구성 및 내비게이션
# ==========================================
with st.sidebar:
    st.markdown("<h2 style='color:#2563eb; font-weight:800; margin-bottom: 20px;'>🏫 AI 평가 보조</h2>", unsafe_allow_html=True)
    
    menu_options = ["📂 AI 서술형 채점 ▾", "ㅤㅤ📄 단일 채점 (텍스트/파일)", "ㅤㅤ📁 학급 전체 채점 (엑셀)", "✨ AI 루브릭 설계"] if st.session_state.menu_expanded else ["📂 AI 서술형 채점 ▸", "✨ AI 루브릭 설계"]
    
    try: current_index = menu_options.index(st.session_state.current_page)
    except: current_index = 0

    selected = st.radio("메뉴", menu_options, index=current_index, label_visibility="collapsed")
    
    if "▾" in selected:
        st.session_state.menu_expanded = False
        st.rerun()
    elif "▸" in selected:
        st.session_state.menu_expanded = True
        st.rerun()
    elif selected != st.session_state.current_page:
        st.session_state.current_page = selected
        st.rerun()

    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.markdown("🔒 **시스템 설정**")
    api_key_input = st.text_input("Google API Key 입력", type="password", placeholder="AIzaSy...")

display_title = "📋 AI 서술형 채점" if "AI 서술형 채점" in st.session_state.current_page else "✨ AI 루브릭 설계"
st.markdown(f"<div class='main-title'>{display_title}</div>", unsafe_allow_html=True)

# ------------------------------------------
# 메뉴 1: 단일 채점 모드
# ------------------------------------------
if st.session_state.current_page == "ㅤㅤ📄 단일 채점 (텍스트/파일)":
    st.markdown("<div class='sub-title'>선생님만의 루브릭을 실시간으로 추가하고 평가할 수 있습니다.</div>", unsafe_allow_html=True)
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    
    st.markdown("#### ⚙️ 평가 기준 선택 및 추가")
    # 선택 목록에 "새 루브릭 추가" 옵션을 병합
    rubric_options = list(st.session_state.rubric_binder.keys()) + ["➕ 새 루브릭 직접 추가"]
    selected_rubric_name = st.selectbox("사용할 루브릭을 선택하거나 새로 만드세요.", rubric_options)
    
    # 💡 [새 기능] 새 루브릭 추가 로직
    if selected_rubric_name == "➕ 새 루브릭 직접 추가":
        st.info("새로운 평가 기준을 바인더에 등록합니다.")
        new_name = st.text_input("루브릭 이름 (예: 화학 반응 속도 실험 평가)")
        new_content = st.text_area("루브릭 상세 내용 (영역별 기준을 작성하세요)", height=200)
        if st.button("💾 바인더에 저장하기"):
            if new_name and new_content:
                st.session_state.rubric_binder[new_name] = new_content
                st.success(f"'{new_name}' 루브릭이 등록되었습니다! 목록에서 선택해 주세요.")
                time.sleep(1)
                st.rerun()
            else:
                st.error("이름과 내용을 모두 입력해 주세요.")
    else:
        # 기존 루브릭 선택 시
        current_rubric_text = st.session_state.rubric_binder[selected_rubric_name]
        with st.expander("📌 선택된 루브릭 상세 보기"):
            st.markdown(current_rubric_text)
        
        st.markdown("---")
        st.markdown("#### 📄 학생 과제 입력")
        upload_type = st.radio("입력 방식", ["직접 텍스트 입력", "파일 업로드 (PDF/사진)"], horizontal=True)
        
        student_text, uploaded_file_path, tmp_path = None, None, None
        if upload_type == "직접 텍스트 입력":
            student_text = st.text_area("학생의 글을 붙여넣으세요.", height=150)
        else:
            uploaded_file = st.file_uploader("파일 업로드", type=['pdf', 'png', 'jpg', 'jpeg'])
            if uploaded_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                    uploaded_file_path = tmp_path

        if st.button("🚀 채점 시작"):
            if not api_key_input: st.error("👈 API Key를 입력해주세요.")
            elif not student_text and not uploaded_file_path: st.warning("내용을 입력해주세요.")
            else:
                with st.spinner("AI가 채점 중입니다..."):
                    result = evaluate_with_gemini(api_key_input, text_content=student_text, uploaded_file_path=uploaded_file_path, rubric=current_rubric_text)
                    st.markdown(f'<div class="result-box"><b>[AI 채점 결과]</b><br><br>{result}</div>', unsafe_allow_html=True)
                if tmp_path: os.remove(tmp_path) 
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------
# 메뉴 2: 학급 전체 채점 (동일하게 바인더 적용)
# ------------------------------------------
elif st.session_state.current_page == "ㅤㅤ📁 학급 전체 채점 (엑셀)":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("#### 📊 학급 일괄 채점")
    batch_rubric_name = st.selectbox("일괄 채점에 적용할 루브릭", list(st.session_state.rubric_binder.keys()))
    st.markdown(f"선택된 루브릭: **{batch_rubric_name}**")
    
    uploaded_csv = st.file_uploader("CSV 파일 선택", type=['csv'])
    if uploaded_csv and st.button("🚀 일괄 채점 시작"):
        if not api_key_input: st.error("👈 API Key 입력 필요")
        else:
            try: df = pd.read_csv(uploaded_csv, encoding='utf-8')
            except: df = pd.read_csv(uploaded_csv, encoding='cp949')
            
            if '과제내용' not in df.columns: st.error("'과제내용' 열 없음")
            else:
                my_bar = st.progress(0, text="채점 중...")
                results, total = [], len(df)
                for i, row in df.iterrows():
                    res = evaluate_with_gemini(api_key_input, text_content=str(row['과제내용']), rubric=st.session_state.rubric_binder[batch_rubric_name])
                    results.append(res)
                    my_bar.progress(int(((i + 1) / total) * 100))
                    time.sleep(1)
                df['AI_피드백'] = results
                st.success("✅ 완료!")
                st.dataframe(df)
                st.download_button("📥 다운로드", df.to_csv(index=False).encode('utf-8-sig'), "결과.csv", "text/csv")
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------
# 메뉴 3: AI 루브릭 설계
# ------------------------------------------
elif st.session_state.current_page == "✨ AI 루브릭 설계":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("#### 🪄 AI 루브릭 설계소")
    rubric_topic = st.text_input("수행평가 주제 입력")
    if st.button("🪄 설계 요청"):
        if not api_key_input: st.error("API Key 필요")
        else:
            with st.spinner("설계 중..."):
                gen_rubric = generate_rubric_with_gemini(api_key_input, rubric_topic)
                st.markdown(f'<div class="result-box"><b>[추천 루브릭]</b><br><br>{gen_rubric}</div>', unsafe_allow_html=True)
                st.info("💡 팁: 위 내용을 복사해서 '단일 채점' 메뉴의 '새 루브릭 추가'에 붙여넣어 사용하세요!")
    st.markdown("</div>", unsafe_allow_html=True)