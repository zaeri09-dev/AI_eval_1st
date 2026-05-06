import streamlit as st
import pandas as pd
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# ==========================================
# 1. 페이지 기본 설정 및 디자인 (CSS)
# ==========================================
st.set_page_config(page_title="AI 학생 과제 평가 시스템", page_icon="📝", layout="wide")

# 세련되고 깔끔한 UI를 위한 커스텀 CSS 적용 (이모티콘 남발 방지, 여백과 그림자 활용)
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stTextArea textarea { border-radius: 10px; border: 1px solid #dee2e6; padding: 15px; }
    .stButton>button { border-radius: 8px; font-weight: 600; padding: 0.5rem 1rem; width: 100%; transition: all 0.3s; }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .result-box { background-color: white; padding: 25px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); margin-top: 20px; border-left: 5px solid #4a90e2; }
    h1, h2, h3 { color: #2c3e50; font-family: 'Pretendard', sans-serif; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 데이터베이스 및 템플릿 설정
# ==========================================
# 💡 [수정 필요] 선생님의 과목과 상황에 맞게 평가 기준을 자유롭게 수정하거나 추가하세요.
RUBRIC_DB = {
    "고등_화학_실험보고서": "1. 실험 목적과 원리를 정확히 서술했는가? (30점)\n2. 실험 과정과 결과를 논리적으로 분석했는가? (40점)\n3. 오차의 원인을 과학적으로 추론했는가? (30점)",
    "고등_독서감상문": "1. 책의 핵심 내용을 정확히 파악했는가? (30점)\n2. 자신의 삶과 연결하여 성찰했는가? (40점)\n3. 문장이 학술적이고 매끄러운가? (30점)",
    "초중등_일반에세이": "1. 주제가 명확하게 드러나는가? (30점)\n2. 논리적 흐름이 자연스러운가? (40점)\n3. 표현이 풍부하고 창의적인가? (30점)"
}

# 💡 [수정 필요] 선생님만의 평가 스타일이 담긴 예시를 넣어주세요.
FEW_SHOT_EXAMPLES = """
[모범 채점 예시 - 화학 실험 보고서]
학생 글: 이번 실험에서 감압 플라스크를 썼다. 결과가 이론값보다 작게 나왔다. 이유는 잘 모르겠지만 실수한 것 같다.
채점 결과:
- 항목별 평가: 실험 목적/원리(10/30점 - 서술 부족), 과정/결과 분석(15/40점 - 단순 사실만 나열), 오차 추론(5/30점 - 과학적 근거 부족)
- 총점: 30점
- 종합 피드백: 플라스크를 사용한 점을 잘 기록해 주었어요. 하지만 오차가 발생한 '과학적인 이유(예: 압력 변화, 불순물 등)'를 교과서 개념과 연결해서 한 줄만 더 고민해 본다면 훨씬 훌륭한 보고서가 될 것입니다.
"""

# AI에게 내릴 지시문 (프롬프트 템플릿)
TEMPLATE_TEXT = """
너는 {grade} 학생들을 지도하는 통찰력 있고 따뜻한 교사야.
아래 제공된 [평가 기준]과 [채점 예시]를 참고해서, 학생의 글을 공정하게 평가하고 성장을 돕는 피드백을 작성해줘.

[평가 기준]
{criteria}

{examples}

[학생 글]
{student_text}

[출력 형식]
- 항목별 평가: (각 기준별 충족 여부와 코멘트)
- 총점: 
- 종합 피드백: (성장을 독려하는 따뜻한 조언)
"""

prompt_template = PromptTemplate(
    input_variables=["grade", "criteria", "examples", "student_text"],
    template=TEMPLATE_TEXT
)

# ==========================================
# 3. AI 평가 함수 정의 (에러 처리 포함)
# ==========================================
def evaluate_student_text(api_key, text, grade, rubric):
    try:
        # AI 모델 설정
        llm = ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview", # 최신 모델 적용
            google_api_key=api_key,
            temperature=0.2 # 일관성 있는 평가를 위해 창의성 낮춤
        )
        
        # 파이프라인(Chain) 연결 및 실행
        chain = prompt_template | llm
        result = chain.invoke({
            "grade": grade,
            "criteria": rubric,
            "examples": FEW_SHOT_EXAMPLES,
            "student_text": text
        })
        return result.content
    except Exception as e:
        # 에러 발생 시 프로그램이 뻗지 않고 안내 메시지 반환
        return f"⚠️ 오류가 발생했습니다. API 키가 정확한지, 인터넷 연결이 되어 있는지 확인해 주세요.\n(상세 내용: {str(e)})"


# ==========================================
# 4. 화면 구성 (UI)
# ==========================================
st.title("AI 학생 과제 평가 시스템")
st.markdown("학생들의 텍스트 과제를 빠르고 일관되게 분석하여 맞춤형 피드백을 제공합니다.")

# 사이드바 (설정 영역) - 보안을 위해 API 키를 여기서 입력받습니다.
with st.sidebar:
    st.header("⚙️ 평가 설정")
    
    # 비밀번호 형태로 입력받아 화면에 노출되지 않음
    api_key_input = st.text_input("Google API Key 입력", type="password", help="발급받으신 Gemini API 키를 입력하세요.")
    
    st.markdown("---")
    target_grade = st.selectbox("학생 학교급", ["고등학교", "중학교", "초등학교"])
    selected_category = st.selectbox("과제 유형 (평가 루브릭)", list(RUBRIC_DB.keys()))
    
    st.markdown("---")
    st.info("💡 API 키는 저장되지 않으며, 현재 세션에서만 안전하게 사용됩니다.")

# 현재 선택된 루브릭 화면에 살짝 보여주기
current_rubric = RUBRIC_DB[selected_category]
with st.expander("현재 적용된 평가 기준 확인하기"):
    st.write(current_rubric)

# 기능 탭 분리 (개별 입력 vs 엑셀 일괄 처리)
tab1, tab2 = st.tabs(["단일 학생 평가 (직접 입력)", "다수 학생 일괄 평가 (엑셀 업로드)"])

# ------------------------------------------
# 탭 1: 단일 학생 평가
# ------------------------------------------
with tab1:
    st.subheader("과제 텍스트 입력")
    student_input = st.text_area("학생이 작성한 글을 아래에 붙여넣으세요.", height=200, placeholder="여기에 과제 내용을 입력하세요...")
    
    if st.button("평가 시작하기", key="single_eval"):
        if not api_key_input:
            st.warning("👈 왼쪽 사이드바에서 Google API Key를 먼저 입력해 주세요.")
        elif not student_input.strip():
            st.warning("학생의 글을 입력해 주세요.")
        else:
            with st.spinner("AI가 꼼꼼하게 채점하고 피드백을 작성 중입니다..."):
                evaluation_result = evaluate_student_text(api_key_input, student_input, target_grade, current_rubric)
                
                # 결과 출력 (커스텀 CSS로 디자인된 박스 안에 출력)
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.markdown("### 📊 평가 결과")
                st.write(evaluation_result)
                st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------
# 탭 2: 다수 학생 일괄 평가 (엑셀/CSV)
# ------------------------------------------
with tab2:
    st.subheader("파일 업로드 (CSV)")
    st.markdown("이름(또는 학번)과 과제 내용이 포함된 CSV 파일을 업로드하세요. <br>필수 열 이름: **'이름'**, **'과제내용'**", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("CSV 파일 선택", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head(3)) # 파일이 잘 올라왔는지 미리보기 제공
        
        if '과제내용' not in df.columns:
            st.error("업로드한 파일에 '과제내용'이라는 열(Column)이 없습니다. 파일 양식을 확인해 주세요.")
        else:
            if st.button("일괄 평가 시작", key="batch_eval"):
                if not api_key_input:
                    st.warning("👈 왼쪽 사이드바에서 Google API Key를 먼저 입력해 주세요.")
                else:
                    progress_text = "일괄 평가 진행 중..."
                    my_bar = st.progress(0, text=progress_text)
                    
                    results = []
                    total_students = len(df)
                    
                    for i, row in df.iterrows():
                        text_to_eval = str(row['과제내용'])
                        # AI 호출 (과부하 방지를 위해 살짝 대기)
                        res = evaluate_student_text(api_key_input, text_to_eval, target_grade, current_rubric)
                        results.append(res)
                        
                        # 진행률 바 업데이트
                        progress_percent = int(((i + 1) / total_students) * 100)
                        my_bar.progress(progress_percent, text=f"{progress_text} ({i+1}/{total_students}명 완료)")
                        time.sleep(1) # API 호출 제한(Rate Limit) 방지용 대기 시간
                    
                    # 결과를 데이터프레임에 추가
                    df['AI_피드백'] = results
                    st.success("🎉 모든 학생의 평가가 완료되었습니다!")
                    st.dataframe(df)
                    
                    # 결과를 다시 CSV로 다운로드할 수 있게 제공
                    csv = df.to_csv(index=False).encode('utf-8-sig') # 한글 깨짐 방지 utf-8-sig
                    st.download_button(
                        label="📥 평가 결과 다운로드 (CSV)",
                        data=csv,
                        file_name="AI_평가결과.csv",
                        mime="text/csv",
                    )