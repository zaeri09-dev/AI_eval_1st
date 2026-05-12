
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
from google import genai
from google.genai import types

# =========================================================
# 기본 설정
# =========================================================
APP_TITLE = "교사용 AI 채점 대시보드"
DEFAULT_MODEL_OPTIONS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-3-flash-preview",
]

REQUIRED_COLUMNS = ["학번", "이름", "과제내용"]

EVALUATION_SCHEMA: Dict[str, Any] = {
    "type": "OBJECT",
    "required": [
        "total_score",
        "achievement_level",
        "areas",
        "summary_feedback",
        "strengths",
        "improvements",
        "teacher_note",
    ],
    "properties": {
        "total_score": {"type": "INTEGER"},
        "achievement_level": {"type": "STRING"},
        "areas": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "required": ["area", "score", "level", "reason", "next_step"],
                "properties": {
                    "area": {"type": "STRING"},
                    "score": {"type": "INTEGER"},
                    "level": {"type": "STRING"},
                    "reason": {"type": "STRING"},
                    "next_step": {"type": "STRING"},
                },
            },
        },
        "summary_feedback": {"type": "STRING"},
        "strengths": {"type": "ARRAY", "items": {"type": "STRING"}},
        "improvements": {"type": "ARRAY", "items": {"type": "STRING"}},
        "teacher_note": {"type": "STRING"},
    },
}


# =========================================================
# UI 스타일
# =========================================================
def apply_css() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: #f8fafc; }
        * { font-family: 'Pretendard', 'Noto Sans KR', system-ui, sans-serif; }
        .page-title {
            font-size: 1.7rem;
            font-weight: 800;
            color: #0f172a;
            margin: 0.4rem 0 0.2rem 0;
        }
        .page-subtitle {
            font-size: 0.95rem;
            color: #64748b;
            margin: 0 0 1.4rem 0;
        }
        .notice-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 1rem 1.1rem;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
        }
        .small-muted { color: #64748b; font-size: 0.86rem; }
        .result-card {
            background: #ffffff;
            border: 1px solid #dbeafe;
            border-left: 6px solid #2563eb;
            border-radius: 14px;
            padding: 1.2rem;
            margin-top: 1rem;
        }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            padding: 0.8rem;
            border-radius: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# 루브릭 / 상태 관리
# =========================================================
def default_rubrics() -> Dict[str, str]:
    return {
        "고등_화학_실험보고서": """
**[영역 1: 이론적 배경 및 가설 | 30점]**
- 우수(25-30점): 핵심 화학 원리를 정확히 설명하고, 이를 바탕으로 논리적인 가설을 설정함
- 보통(15-24점): 원리 설명이 일부 누락되었거나, 가설과의 연결성이 다소 부족함
- 미흡(0-14점): 원리 설명이 부정확하거나 가설이 없음

**[영역 2: 결과 분석 및 해석 | 40점]**
- 우수(35-40점): 데이터를 과학적으로 변환하고, 독립/종속 변인의 관계를 심층적으로 분석함
- 보통(20-34점): 표면적인 데이터만 제시하였으며, 원리를 통한 해석이 부족함
- 미흡(0-19점): 단순 사실 나열에 그치거나 해석에 오류가 있음

**[영역 3: 오차 논의 | 30점]**
- 우수(25-30점): 오차 원인을 화학적/환경적 요인으로 다각도 분석하고 구체적 개선안을 제시함
- 보통(15-24점): 오차 원인을 단순 계산이나 측정 실수 정도로만 언급함
- 미흡(0-14점): 오차 분석이 전혀 없음
        """.strip(),
        "고등_과학_논술평가": """
**[영역 1: 과학적 사실의 정확성 | 40점]**
- 우수(35-40점): 교과 개념을 정확히 적용하여 논제를 완벽히 해결함
- 보통(20-34점): 개념 적용을 시도했으나 일부 오개념이 존재함
- 미흡(0-19점): 과학적 근거가 매우 부족함

**[영역 2: 논리적 전개 | 30점]**
- 우수(25-30점): 서론-본론-결론이 명확하고 문장 간 연결이 매끄러움
- 보통(15-24점): 흐름이 다소 끊어지거나 논리적 비약이 있음
- 미흡(0-14점): 주장만 있고 이를 뒷받침하는 근거가 불명확함

**[영역 3: 문제 해결 및 창의성 | 30점]**
- 우수(25-30점): 사회적/환경적 문제와 연계하여 독창적인 대안을 제시함
- 보통(15-24점): 일반적이고 누구나 생각할 수 있는 뻔한 대안을 제시함
- 미흡(0-14점): 문제에 대한 대안 제시가 전혀 없음
        """.strip(),
    }


def init_session_state() -> None:
    if "rubrics" not in st.session_state:
        st.session_state.rubrics = default_rubrics()
    if "last_single_result" not in st.session_state:
        st.session_state.last_single_result = None
    if "batch_result_df" not in st.session_state:
        st.session_state.batch_result_df = None


def get_secret_api_key() -> str:
    try:
        return str(st.secrets.get("GEMINI_API_KEY", "")).strip()
    except Exception:
        return ""


# =========================================================
# Gemini 호출 유틸
# =========================================================
def make_client(api_key: str) -> genai.Client:
    if not api_key:
        raise ValueError("API Key가 비어 있습니다.")
    return genai.Client(api_key=api_key)


def build_evaluation_prompt(
    rubric: str,
    student_id: str = "",
    student_name: str = "",
    send_personal_info: bool = False,
) -> str:
    if send_personal_info:
        student_info = f"- 학번: {student_id or '미제공'}\n- 이름: {student_name or '미제공'}"
        name_rule = "학생 이름이 제공되면 summary_feedback의 첫 문장을 자연스럽게 이름으로 시작해도 됩니다."
    else:
        student_info = "- 개인정보 최소화 모드: 학생 이름과 학번은 모델에 제공하지 않습니다."
        name_rule = "학생의 실명이나 학번을 추측하거나 만들지 마세요."

    return f"""
너는 고등학교 교사를 돕는 평가 보조 AI입니다.
아래 루브릭만 기준으로 학생 과제를 채점하세요. 최종 성적 확정자는 담당 교사입니다.

[평가 원칙]
1. 과제에 없는 내용을 추측하지 마세요.
2. 각 영역의 점수는 루브릭의 점수 범위와 성취수준에 맞추세요.
3. 부족한 점은 비난하지 말고, 다음 제출에서 바로 고칠 수 있는 행동으로 제안하세요.
4. 전체 총점은 영역별 점수 합산을 기준으로 하세요.
5. {name_rule}
6. 반드시 지정된 JSON 형식으로만 답하세요.

[학생 정보]
{student_info}

[세부 루브릭]
{rubric}
    """.strip()


def build_rubric_prompt(topic: str) -> str:
    return f"""
교육평가 전문가로서 고등학교 수행평가 주제 '{topic}'에 대한 분석적 루브릭을 작성하세요.

조건:
- 총점 100점
- 3개 영역
- 각 영역마다 우수/보통/미흡 3단계
- 각 단계에 점수 범위와 관찰 가능한 행동 기준 포함
- 교사가 바로 복사해 사용할 수 있는 Markdown 형식
    """.strip()


def strip_code_fence(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```").strip()
        cleaned = cleaned.removesuffix("```").strip()
    return cleaned


def safe_json_loads(text: str) -> Dict[str, Any]:
    try:
        return json.loads(strip_code_fence(text))
    except Exception:
        return {
            "total_score": None,
            "achievement_level": "파싱 필요",
            "areas": [],
            "summary_feedback": text,
            "strengths": [],
            "improvements": [],
            "teacher_note": "모델 응답이 JSON으로 파싱되지 않았습니다. 원문을 확인하세요.",
            "_parse_error": True,
        }


def evaluate_assignment(
    *,
    api_key: str,
    model_name: str,
    rubric: str,
    text_content: str = "",
    uploaded_file_path: Optional[str] = None,
    student_id: str = "",
    student_name: str = "",
    send_personal_info: bool = False,
    temperature: float = 0.2,
) -> Dict[str, Any]:
    client = make_client(api_key)
    prompt = build_evaluation_prompt(
        rubric=rubric,
        student_id=student_id,
        student_name=student_name,
        send_personal_info=send_personal_info,
    )

    contents: List[Any] = [prompt]
    if text_content.strip():
        contents.append(f"\n[학생 과제 내용]\n{text_content.strip()}")

    remote_file = None
    try:
        if uploaded_file_path:
            remote_file = client.files.upload(file=uploaded_file_path)
            contents.append(remote_file)

        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=temperature,
                response_mime_type="application/json",
                response_schema=EVALUATION_SCHEMA,
            ),
        )
        result = safe_json_loads(response.text or "")
        result["student_id"] = student_id
        result["student_name"] = student_name
        return result
    finally:
        if remote_file is not None:
            try:
                client.files.delete(name=remote_file.name)
            except Exception:
                # 원격 파일 삭제 실패는 채점 결과 표시를 막지 않는다.
                pass


def generate_rubric(
    *,
    api_key: str,
    model_name: str,
    topic: str,
    temperature: float = 0.4,
) -> str:
    client = make_client(api_key)
    response = client.models.generate_content(
        model=model_name,
        contents=build_rubric_prompt(topic),
        config=types.GenerateContentConfig(temperature=temperature),
    )
    return response.text or ""


# =========================================================
# 파일 처리
# =========================================================
def save_uploaded_file_temporarily(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        return tmp.name


def read_roster_file(uploaded_file) -> pd.DataFrame:
    suffix = Path(uploaded_file.name).suffix.lower()

    if suffix == ".csv":
        try:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding="utf-8-sig")
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding="cp949")

    if suffix == ".xlsx":
        uploaded_file.seek(0)
        return pd.read_excel(uploaded_file)

    raise ValueError("CSV 또는 XLSX 파일만 지원합니다.")


def validate_roster_columns(df: pd.DataFrame) -> List[str]:
    return [col for col in REQUIRED_COLUMNS if col not in df.columns]


def make_sample_csv() -> bytes:
    sample = pd.DataFrame(
        {
            "학번": ["10101", "10102"],
            "이름": ["홍길동", "김민지"],
            "과제내용": [
                "실험 목적은 반응 속도에 영향을 주는 요인을 확인하는 것이다...",
                "자료를 보면 온도가 높을수록 반응이 빨라진다...",
            ],
        }
    )
    return sample.to_csv(index=False).encode("utf-8-sig")


# =========================================================
# 결과 표시 / 다운로드
# =========================================================
def render_result(result: Dict[str, Any]) -> None:
    total_score = result.get("total_score")
    achievement = result.get("achievement_level", "-")

    col1, col2, col3 = st.columns(3)
    col1.metric("총점", "-" if total_score is None else f"{total_score}점")
    col2.metric("성취 수준", achievement)
    col3.metric("영역 수", len(result.get("areas", [])))

    areas = result.get("areas", [])
    if areas:
        st.markdown("#### 영역별 평가")
        st.dataframe(pd.DataFrame(areas), use_container_width=True, hide_index=True)

    st.markdown("#### 종합 피드백")
    st.markdown(result.get("summary_feedback", ""))

    strengths = result.get("strengths", [])
    improvements = result.get("improvements", [])

    left, right = st.columns(2)
    with left:
        st.markdown("#### 잘한 점")
        if strengths:
            for item in strengths:
                st.markdown(f"- {item}")
        else:
            st.caption("모델이 별도 강점을 반환하지 않았습니다.")

    with right:
        st.markdown("#### 보완할 점")
        if improvements:
            for item in improvements:
                st.markdown(f"- {item}")
        else:
            st.caption("모델이 별도 개선점을 반환하지 않았습니다.")

    teacher_note = result.get("teacher_note")
    if teacher_note:
        with st.expander("교사용 메모"):
            st.markdown(teacher_note)

    if result.get("_parse_error"):
        st.warning("응답이 구조화 JSON으로 완전히 파싱되지 않았습니다. 위 원문 피드백을 확인하세요.")


def flatten_result(result: Dict[str, Any]) -> Dict[str, Any]:
    areas = result.get("areas", [])
    area_score_text = " / ".join(
        [
            f"{a.get('area', '-')}: {a.get('score', '-')}점({a.get('level', '-')})"
            for a in areas
        ]
    )

    return {
        "AI_총점": result.get("total_score"),
        "AI_수준": result.get("achievement_level"),
        "AI_영역별점수": area_score_text,
        "AI_종합피드백": result.get("summary_feedback", ""),
        "AI_강점": " | ".join(result.get("strengths", [])),
        "AI_보완점": " | ".join(result.get("improvements", [])),
        "AI_교사용메모": result.get("teacher_note", ""),
        "AI_원문JSON": json.dumps(result, ensure_ascii=False),
    }


# =========================================================
# 화면: 단일 채점
# =========================================================
def page_single_grading(api_key: str, model_name: str, temperature: float, send_personal_info: bool) -> None:
    st.markdown("<div class='page-title'>📄 단일 채점</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='page-subtitle'>텍스트 또는 PDF/이미지 파일 1개를 루브릭 기준으로 평가합니다.</div>",
        unsafe_allow_html=True,
    )

    rubric_name = st.selectbox("사용할 루브릭", list(st.session_state.rubrics.keys()))
    rubric = st.session_state.rubrics[rubric_name]

    with st.expander("선택된 루브릭 보기"):
        st.markdown(rubric)

    st.markdown("#### 학생 정보")
    col1, col2 = st.columns(2)
    with col1:
        student_id = st.text_input("학번", placeholder="예: 10101")
    with col2:
        student_name = st.text_input("이름", placeholder="예: 홍길동")

    input_type = st.radio("입력 방식", ["직접 텍스트 입력", "파일 업로드"], horizontal=True)

    text_content = ""
    uploaded_file = None

    if input_type == "직접 텍스트 입력":
        text_content = st.text_area("학생 과제 내용", height=220, placeholder="학생의 서술형 답안이나 보고서 내용을 붙여넣으세요.")
    else:
        uploaded_file = st.file_uploader(
            "PDF/이미지 업로드",
            type=["pdf", "png", "jpg", "jpeg"],
            max_upload_size=50,
            help="PDF는 50MB 이하를 권장합니다.",
        )
        if uploaded_file is not None:
            st.caption(f"업로드됨: {uploaded_file.name}")

    submitted = st.button("🚀 채점 시작", type="primary", use_container_width=True)

    if submitted:
        temp_path = None
        if not api_key:
            st.error("API Key를 입력하거나 Streamlit secrets에 GEMINI_API_KEY를 설정하세요.")
        elif input_type == "직접 텍스트 입력" and not text_content.strip():
            st.warning("학생 과제 내용을 입력하세요.")
        elif input_type == "파일 업로드" and uploaded_file is None:
            st.warning("채점할 파일을 업로드하세요.")
        else:
            try:
                if uploaded_file is not None:
                    temp_path = save_uploaded_file_temporarily(uploaded_file)

                with st.spinner("AI가 루브릭 기준으로 채점 중입니다."):
                    result = evaluate_assignment(
                        api_key=api_key,
                        model_name=model_name,
                        rubric=rubric,
                        text_content=text_content,
                        uploaded_file_path=temp_path,
                        student_id=student_id.strip(),
                        student_name=student_name.strip(),
                        send_personal_info=send_personal_info,
                        temperature=temperature,
                    )
                    st.session_state.last_single_result = result
            except Exception as exc:
                st.error(f"채점 중 오류가 발생했습니다: {exc}")
            finally:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)

    if st.session_state.last_single_result:
        with st.container(border=True):
            render_result(st.session_state.last_single_result)

        json_bytes = json.dumps(st.session_state.last_single_result, ensure_ascii=False, indent=2).encode("utf-8")
        st.download_button(
            "📥 단일 채점 결과 JSON 다운로드",
            data=json_bytes,
            file_name="단일채점결과.json",
            mime="application/json",
            on_click="ignore",
        )


# =========================================================
# 화면: 학급 일괄 채점
# =========================================================
def page_batch_grading(api_key: str, model_name: str, temperature: float, send_personal_info: bool, delay_seconds: float) -> None:
    st.markdown("<div class='page-title'>📁 학급 일괄 채점</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='page-subtitle'>CSV 또는 XLSX 파일의 학번·이름·과제내용 열을 읽어 순차 채점합니다.</div>",
        unsafe_allow_html=True,
    )

    st.info("필수 열 이름: 학번, 이름, 과제내용")
    st.download_button(
        "예시 CSV 다운로드",
        data=make_sample_csv(),
        file_name="일괄채점_예시.csv",
        mime="text/csv",
        on_click="ignore",
    )

    rubric_name = st.selectbox("일괄 채점 루브릭", list(st.session_state.rubrics.keys()))
    rubric = st.session_state.rubrics[rubric_name]

    with st.expander("선택된 루브릭 보기"):
        st.markdown(rubric)

    uploaded_roster = st.file_uploader("학급 명단/답안 파일 업로드", type=["csv", "xlsx"])

    if uploaded_roster is not None:
        try:
            df = read_roster_file(uploaded_roster)
            st.markdown("#### 미리보기")
            st.dataframe(df.head(10), use_container_width=True, hide_index=True)

            missing = validate_roster_columns(df)
            if missing:
                st.error(f"필수 열이 없습니다: {', '.join(missing)}")
                return

            if st.button("🚀 일괄 채점 시작", type="primary", use_container_width=True):
                if not api_key:
                    st.error("API Key를 입력하거나 Streamlit secrets에 GEMINI_API_KEY를 설정하세요.")
                    return

                progress = st.progress(0, text="채점 준비 중...")
                rows: List[Dict[str, Any]] = []
                total = len(df)

                for pos, (_, row) in enumerate(df.iterrows(), start=1):
                    student_id = "" if pd.isna(row["학번"]) else str(row["학번"]).strip()
                    student_name = "" if pd.isna(row["이름"]) else str(row["이름"]).strip()
                    assignment = "" if pd.isna(row["과제내용"]) else str(row["과제내용"]).strip()

                    progress.progress(pos / max(total, 1), text=f"{pos}/{total} 채점 중: {student_name or student_id or '이름 없음'}")

                    base_row = row.to_dict()

                    if not assignment:
                        base_row.update(
                            {
                                "AI_총점": None,
                                "AI_수준": "미채점",
                                "AI_영역별점수": "",
                                "AI_종합피드백": "과제내용이 비어 있어 채점하지 않았습니다.",
                                "AI_강점": "",
                                "AI_보완점": "",
                                "AI_교사용메모": "",
                                "AI_원문JSON": "",
                            }
                        )
                        rows.append(base_row)
                        continue

                    try:
                        result = evaluate_assignment(
                            api_key=api_key,
                            model_name=model_name,
                            rubric=rubric,
                            text_content=assignment,
                            student_id=student_id,
                            student_name=student_name,
                            send_personal_info=send_personal_info,
                            temperature=temperature,
                        )
                        base_row.update(flatten_result(result))
                    except Exception as exc:
                        base_row.update(
                            {
                                "AI_총점": None,
                                "AI_수준": "오류",
                                "AI_영역별점수": "",
                                "AI_종합피드백": f"채점 오류: {exc}",
                                "AI_강점": "",
                                "AI_보완점": "",
                                "AI_교사용메모": "",
                                "AI_원문JSON": "",
                            }
                        )

                    rows.append(base_row)

                    if delay_seconds > 0 and pos < total:
                        time.sleep(delay_seconds)

                result_df = pd.DataFrame(rows)
                st.session_state.batch_result_df = result_df
                progress.progress(1.0, text="채점 완료")
                st.success("일괄 채점이 완료되었습니다.")
        except Exception as exc:
            st.error(f"파일을 읽는 중 오류가 발생했습니다: {exc}")

    if st.session_state.batch_result_df is not None:
        st.markdown("#### 채점 결과")
        st.dataframe(st.session_state.batch_result_df, use_container_width=True, hide_index=True)

        csv_data = st.session_state.batch_result_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "📥 결과 CSV 다운로드",
            data=csv_data,
            file_name="일괄채점결과.csv",
            mime="text/csv",
            type="primary",
            on_click="ignore",
        )


# =========================================================
# 화면: AI 루브릭 설계
# =========================================================
def page_rubric_generator(api_key: str, model_name: str, temperature: float) -> None:
    st.markdown("<div class='page-title'>✨ AI 루브릭 설계</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='page-subtitle'>수행평가 주제만 입력하면 100점 만점 3영역 루브릭 초안을 생성합니다.</div>",
        unsafe_allow_html=True,
    )

    topic = st.text_input("수행평가 주제", placeholder="예: 르샤틀리에 원리 실험 보고서")
    if st.button("🪄 루브릭 생성", type="primary", use_container_width=True):
        if not api_key:
            st.error("API Key를 입력하거나 Streamlit secrets에 GEMINI_API_KEY를 설정하세요.")
        elif not topic.strip():
            st.warning("수행평가 주제를 입력하세요.")
        else:
            try:
                with st.spinner("루브릭 초안을 생성 중입니다."):
                    rubric_text = generate_rubric(
                        api_key=api_key,
                        model_name=model_name,
                        topic=topic.strip(),
                        temperature=temperature,
                    )
                    st.session_state.generated_rubric = rubric_text
            except Exception as exc:
                st.error(f"루브릭 생성 중 오류가 발생했습니다: {exc}")

    generated = st.session_state.get("generated_rubric")
    if generated:
        st.markdown("#### 생성된 루브릭")
        st.markdown(generated)

        with st.form("save_generated_rubric_form"):
            name = st.text_input("저장할 루브릭 이름", value=topic.strip() if topic else "")
            submitted = st.form_submit_button("💾 루브릭 바인더에 저장")
            if submitted:
                if not name.strip():
                    st.error("루브릭 이름을 입력하세요.")
                else:
                    st.session_state.rubrics[name.strip()] = generated
                    st.success("루브릭 바인더에 저장했습니다.")


# =========================================================
# 화면: 루브릭 관리
# =========================================================
def page_rubric_manager() -> None:
    st.markdown("<div class='page-title'>🧰 루브릭 관리</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='page-subtitle'>루브릭을 추가·삭제하고 JSON으로 내보내거나 가져올 수 있습니다.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("#### 현재 루브릭")
    selected = st.selectbox("수정/삭제할 루브릭", list(st.session_state.rubrics.keys()))
    edited = st.text_area("루브릭 내용", value=st.session_state.rubrics[selected], height=260)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("수정 내용 저장", use_container_width=True):
            st.session_state.rubrics[selected] = edited
            st.success("수정 내용을 저장했습니다.")
    with col2:
        if st.button("선택 루브릭 삭제", use_container_width=True):
            if len(st.session_state.rubrics) <= 1:
                st.warning("최소 1개의 루브릭은 남겨야 합니다.")
            else:
                del st.session_state.rubrics[selected]
                st.success("삭제했습니다.")
                st.rerun()

    st.markdown("---")
    st.markdown("#### 새 루브릭 직접 추가")
    with st.form("add_rubric_form"):
        new_name = st.text_input("새 루브릭 이름")
        new_content = st.text_area("새 루브릭 내용", height=180)
        submitted = st.form_submit_button("추가")
        if submitted:
            if not new_name.strip() or not new_content.strip():
                st.error("이름과 내용을 모두 입력하세요.")
            else:
                st.session_state.rubrics[new_name.strip()] = new_content.strip()
                st.success("새 루브릭을 추가했습니다.")

    st.markdown("---")
    st.markdown("#### 가져오기 / 내보내기")
    rubrics_json = json.dumps(st.session_state.rubrics, ensure_ascii=False, indent=2).encode("utf-8")
    st.download_button(
        "루브릭 JSON 내보내기",
        data=rubrics_json,
        file_name="rubrics.json",
        mime="application/json",
        on_click="ignore",
    )

    imported = st.file_uploader("루브릭 JSON 가져오기", type=["json"])
    if imported is not None and st.button("가져오기 실행"):
        try:
            data = json.loads(imported.getvalue().decode("utf-8"))
            if not isinstance(data, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in data.items()):
                st.error("올바른 루브릭 JSON 형식이 아닙니다. {이름: 내용} 형태여야 합니다.")
            else:
                st.session_state.rubrics.update(data)
                st.success("루브릭을 가져왔습니다.")
        except Exception as exc:
            st.error(f"가져오기 실패: {exc}")


# =========================================================
# 메인
# =========================================================
def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🏫", layout="wide")
    apply_css()
    init_session_state()

    st.sidebar.markdown("## 🏫 AI 평가 보조")
    page = st.sidebar.radio(
        "메뉴",
        ["📄 단일 채점", "📁 학급 일괄 채점", "✨ AI 루브릭 설계", "🧰 루브릭 관리"],
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔒 시스템 설정")

    secret_key = get_secret_api_key()
    api_key_input = st.sidebar.text_input(
        "Google Gemini API Key",
        type="password",
        placeholder="AIza...",
        help="Streamlit secrets에 GEMINI_API_KEY를 저장하면 입력하지 않아도 됩니다.",
    )
    api_key = api_key_input.strip() or secret_key

    model_name = st.sidebar.selectbox("모델", DEFAULT_MODEL_OPTIONS, index=0)
    temperature = st.sidebar.slider("창의성/변동성", min_value=0.0, max_value=1.0, value=0.2, step=0.1)
    send_personal_info = st.sidebar.toggle(
        "이름/학번을 모델에 전달",
        value=False,
        help="꺼두면 개인정보를 모델에 보내지 않고, 결과 파일에만 원래 학번/이름을 유지합니다.",
    )
    delay_seconds = st.sidebar.slider("일괄 채점 간격(초)", 0.0, 3.0, 0.5, 0.5)

    st.sidebar.markdown(
        "<p class='small-muted'>AI 평가는 보조 자료입니다. 최종 점수는 교사가 검토해 확정하세요.</p>",
        unsafe_allow_html=True,
    )

    if page == "📄 단일 채점":
        page_single_grading(api_key, model_name, temperature, send_personal_info)
    elif page == "📁 학급 일괄 채점":
        page_batch_grading(api_key, model_name, temperature, send_personal_info, delay_seconds)
    elif page == "✨ AI 루브릭 설계":
        page_rubric_generator(api_key, model_name, temperature)
    else:
        page_rubric_manager()


if __name__ == "__main__":
    main()
