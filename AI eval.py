import streamlit as st
import pandas as pd
import tempfile
import os
import time
import google.generativeai as genai

# ==========================================
# 1. 페이지 설정 및 디자인 (CSS)
# ==========================================
st.set_page_config(page_title="AI 학생 과제 평가 시스템", page_icon="📝", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stTextArea textarea { border-radius: 10px; border: 1px solid #dee2e6; padding: 15px; }
    .stButton>button { border-radius: 8px; font-weight: 600; padding: 0.5rem 1rem; width: 100%; background-color: #4a90e2; color: white; border: none; }
    .stButton>button:hover { background-color: #357abd; }
    .result-box { background-color: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); margin-top: 20px; border-left: 6px solid #4a90e2; font-size: 16px; line-height: 1.6; }
    h1, h2, h3 { color: #2c3e50; font-family: 'Pretendard', sans-serif; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 교육학적 세부 루브릭 데이터베이스
# ==========================================
# 💡 [수정 필요] 상황에 맞게 루브릭의 내용과 점수를 수정하여 사용하세요.
RUBRIC_DB = {
    "고등_화학_실험보고서": """
    [평가 영역 1: 이론적 배경 및 가설 설정 (30점)]
    - 우수 (25-30점): 핵심 화학 원리(예: 반응 속도론, 화학 평형 등)를 정확히 설명하고, 이를 바탕으로 변인 간의 관계를 포함한 논리적이고 검증 가능한 가설을 설정함.
    - 보통 (15-24점): 화학 원리를 설명하였으나 일부 오류나 누락이 있고, 가설과 원리 간의 연결성이 다소 부족함.
    - 미흡 (0-14점): 원리 설명이 부정확하거나 누락되었으며, 가설이 비논리적이거나 제시되지 않음.

    [평가 영역 2: 실험 결과 분석 및 해석 (40점)]
    - 우수 (35-40점): 수집된 데이터(표, 그래프 등)를 과학적으로 변환하고, 독립/종속 변인 간의 관계를 화학적 원리를 통해 심층적으로 분석함.
    - 보통 (20-34점): 데이터를 제시하였으나 분석이 피상적이고, 원리와의 연결을 통한 해석이 부족함.
    - 미흡 (0-19점): 데이터 처리에 오류가 있거나 단순 사실 나열에 그쳐 분석 및 해석이 이루어지지 않음.

    [평가 영역 3: 결론 도출 및 오차 논의 (30점)]
    - 우수 (25-30점): 실험 결과를 바탕으로 타당한 결론을 도출하고, 오차 발생 원인을 실험 장치, 환경, 화학적 요인 등으로 나누어 다각도로 분석한 뒤 구체적인 개선 방안을 제시함.
    - 보통 (15-24점): 결론은 도출하였으나 오차 원인 분석이 '단순 계산 실수' 등 일차원적이거나 개선 방안이 구체적이지 않음.
    - 미흡 (0-14점): 결론이 실험 결과와 무관하거나, 오차 분석 및 개선 방안이 전혀 제시되지 않음.
    """
}

FEW_SHOT_EXAMPLES = """
[모범 채점 예시]
- 항목별 평가 및 점수 산출: 
  1. 이론적 배경 및 가설 설정 (28/30점 - 우수): 르샤틀리에 원리를 바탕으로 압력 변화에 따른 평형 이동 가설을 매우 논리적으로 설정했습니다. 다만, 온도의 영향에 대한 배경 설명이 한 줄 누락되어 2점 감점하였습니다.
  2. 실험 결과 분석 및 해석 (25/40점 - 보통): 표와 그래프로 데이터는 잘 정리했으나, 결과가 '왜' 그렇게 나왔는지 화학 평형 상수(K)의 개념을 연결하여 해석하지 않고 현상만 단순 서술하여 '보통' 수준에 해당합니다.
  3. 결론 도출 및 오차 논의 (10/30점 - 미흡): 오차 원인을 '측정할 때 눈금을 잘못 본 것 같다'는 단순 실수로만 언급했습니다. 플라스크 내부의 실제 기체 압력 손실 등 화학적 요인에 대한 분석이 없어 아쉽습니다.
- 총점: 63점
- 종합 피드백: 데이터 시각화 능력과 가설 설정 능력이 탁월합니다! 다음 보고서에서는 결과를 분석할 때 '화학 1 교과서 몇 페이지에 나온 개념이 적용되었을까?'를 고민해 보고, 오차 원인도 단순 실수를 넘어 '실험 환경이나 기기의 근본적 한계' 측면에서 파고든다면 훨씬 훌륭한 과학자로 성장할 수 있을 거예요.
"""

# ==========================================
# 3. AI 평가 엔진 
# ==========================================
def evaluate_with_gemini(api_key, text_content=None, uploaded_file_path=None, grade="", rubric=""):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        prompt = f"""
        너는 {grade} 학생들을 지도하는 꼼꼼하고 통찰력 있는 교사야. 
        아래 제공된 [교육학적 세부 루브릭]에 따라 제출된 학생의 과제를 공정하게 평가해줘.
        
        [지시사항]
        1. 각 평가 영역별로 학생의 글이 '우수', '보통', '미흡' 중 어디에 해당하는지 명확히 판단해.
        2. 점수를 부여할 때, 반드시 루브릭에 명시된 서술 내용(기준)을 근거로 '왜 이 점수를 부여했는지' 이유를 상세히 적어.
        3. 손글씨 문서의 경우 글씨를 주의 깊게 판독해.
        4. 마크다운 포맷(JSON 형식 등의 포장지)을 절대 쓰지 말고, 읽기 편한 순수 텍스트로만 출력해.

        [교육학적 세부 루브릭]
        {rubric}

        [선생님의 채점 예시]
        {FEW_SHOT_EXAMPLES}
        """

        contents_to_send = [prompt]
        if text_content: contents_to_send.append(f"\n[학생 과제 내용]\n{text_content}")
        if uploaded_file_path:
            uploaded_gemini_file = genai.upload_file(path=uploaded_file_path)
            contents_to_send.append(uploaded_gemini_file)
            
        response = model.generate_content(contents_to_send, generation_config={"temperature": 0.2})
        
        if uploaded_file_path: genai.delete_file(uploaded_gemini_file.name)
        return response.text 

    except Exception as e:
        return f"⚠️ 오류가 발생했습니다. API 키나 파일 상태를 확인해 주세요.\n(상세 내용: {str(e)})"

# ==========================================
# 4. 화면 구성 (UI)
# ==========================================
st.title("AI 학생 과제 평가 시스템 🔬")
st.markdown("세분화된 평가 루브릭을 바탕으로 AI가 학생들의 보고서를 심층 분석합니다.")

with st.sidebar:
    st.header("⚙️ 평가 설정")
    api_key_input = st.text_input("Google API Key 입력", type="password")
    st.markdown("---")
    target_grade = st.selectbox("학생 학교급", ["고등학교", "중학교"])
    selected_category = st.selectbox("과제 유형 (평가 루브릭)", list(RUBRIC_DB.keys()))

current_rubric = RUBRIC_DB[selected_category]

with st.expander("📌 현재 적용된 세부 평가 기준(루브릭) 확인하기"):
    st.write(current_rubric)

tab1, tab2, tab3 = st.tabs(["📝 텍스트 직접 입력", "📄 파일 업로드 (PDF/사진)", "📊 엑셀 일괄 평가"])

# 탭 1
with tab1:
    student_input = st.text_area("학생이 작성한 글을 붙여넣으세요.", height=150)
    if st.button("평가 시작", key="text_btn"):
        if not api_key_input: st.error("👈 API Key를 입력해주세요.")
        elif not student_input: st.warning("내용을 입력해주세요.")
        else:
            with st.spinner("루브릭을 바탕으로 심층 분석 중입니다..."):
                result = evaluate_with_gemini(api_key_input, text_content=student_input, grade=target_grade, rubric=current_rubric)
                st.markdown(f'<div class="result-box">{result}</div>', unsafe_allow_html=True)

# 탭 2
with tab2:
    st.markdown("학생이 제출한 **PDF**나 **손글씨 사진**을 올려주세요.")
    uploaded_file = st.file_uploader("파일 선택", type=['pdf', 'png', 'jpg', 'jpeg'])
    if st.button("문서 평가 시작", key="file_btn"):
        if not api_key_input: st.error("👈 API Key를 입력해주세요.")
        elif not uploaded_file: st.warning("파일을 업로드해주세요.")
        else:
            with st.spinner("문서를 판독하고 루브릭에 맞춰 채점 중입니다..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                result = evaluate_with_gemini(api_key_input, uploaded_file_path=tmp_file_path, grade=target_grade, rubric=current_rubric)
                os.remove(tmp_file_path)
                st.markdown(f'<div class="result-box">{result}</div>', unsafe_allow_html=True)

# 탭 3 (오류가 수정된 부분)
with tab3:
    st.markdown("필수 열 이름: **이름**, **과제내용**")
    uploaded_csv = st.file_uploader("CSV 파일 선택", type=['csv'])
    if uploaded_csv and st.button("일괄 채점", key="batch_btn"):
        if not api_key_input: 
            st.error("👈 API Key를 입력해주세요.")
        else:
            # 💡 [수정 필요] 윈도우 엑셀의 한글 깨짐 오류(UnicodeDecodeError)를 해결하는 마법의 코드입니다.
            try:
                # 1. 먼저 세계 표준(utf-8)으로 읽기를 시도합니다.
                df = pd.read_csv(uploaded_csv, encoding='utf-8')
            except UnicodeDecodeError:
                # 2. 에러가 나면, 파일을 다시 처음부터 한국어 엑셀 전용 방식(cp949)으로 읽어 들입니다.
                uploaded_csv.seek(0)
                df = pd.read_csv(uploaded_csv, encoding='cp949')

            if '과제내용' not in df.columns: 
                st.error("'과제내용' 열이 없습니다. 엑셀의 첫 번째 줄(제목)이 '과제내용'인지 확인해주세요.")
            else:
                progress_bar = st.progress(0)
                results, total = [], len(df)
                for i, row in df.iterrows():
                    res = evaluate_with_gemini(api_key_input, text_content=str(row['과제내용']), grade=target_grade, rubric=current_rubric)
                    results.append(res)
                    progress_bar.progress(int(((i + 1) / total) * 100))
                    time.sleep(1)
                df['AI_피드백'] = results
                st.success("채점 완료!")
                st.download_button("결과 다운로드", df.to_csv(index=False).encode('utf-8-sig'), "평가결과.csv", "text/csv")