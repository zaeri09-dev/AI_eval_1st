import html
import io
import json
import os
import re
import tempfile
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from google import genai

FIXED_MODEL_NAME = "gemini-3-flash-preview"
MAX_TEXT_CHARS = 12000
DEFAULT_SCORE_SCALE = 100

# 💡 [수정 필요] 학교/과목 상황에 맞게 기본 안내 문구를 바꿀 수 있습니다.
TEACHER_REVIEW_NOTICE = "AI 결과는 최종 성적 확정이 아니라 교사의 검토를 돕는 보조 자료입니다."


# ==========================================
# 1. 페이지 설정 및 기존 분위기를 유지한 CSS
# ==========================================
st.set_page_config(page_title="교사용 AI 채점 대시보드", page_icon="🏫", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background-color: #f8fafc; }
    * { font-family: 'Pretendard', 'Noto Sans KR', sans-serif; }

    .main-title {
        font-size: 28px;
        font-weight: 800;
        color: #1e293b;
        margin-bottom: 5px;
        margin-top: 10px;
    }

    .sub-title {
        font-size: 14.5px;
        color: #64748b;
        margin-bottom: 30px;
    }

    .stButton>button {
        background-color: #3b82f6;
        color: white;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1rem;
        border: none;
        transition: all 0.2s;
    }

    .stButton>button:hover {
        background-color: #2563eb;
        transform: translateY(-1px);
        color: white;
    }

    .stDownloadButton>button {
        background-color: #0f172a;
        color: white;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1rem;
        border: none;
    }

    .result-box {
        background-color: #f1f5f9;
        padding: 24px;
        border-radius: 12px;
        border-left: 5px solid #3b82f6;
        font-size: 15px;
        line-height: 1.7;
        color: #334155;
        margin-top: 15px;
    }

    .soft-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 14px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 2px rgba(15,23,42,0.04);
        margin-bottom: 14px;
    }

    .notice-card {
        background-color: #eff6ff;
        padding: 14px 16px;
        border-radius: 12px;
        border: 1px solid #bfdbfe;
        color: #1e3a8a;
        font-size: 14px;
    }

    .danger-card {
        background-color: #fff7ed;
        padding: 14px 16px;
        border-radius: 12px;
        border: 1px solid #fed7aa;
        color: #9a3412;
        font-size: 14px;
        margin-top: 10px;
    }

    .small-muted {
        color: #64748b;
        font-size: 13px;
        line-height: 1.55;
    }

    .score-pill {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 999px;
        background-color: #dbeafe;
        color: #1d4ed8;
        font-weight: 800;
        margin-right: 6px;
    }

    .level-pill {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 999px;
        background-color: #ecfdf5;
        color: #047857;
        font-weight: 800;
    }

    .rubric-preview {
        background-color:#ffffff;
        border:1px solid #e2e8f0;
        border-radius:12px;
        padding:16px;
    }

    /* 사이드바 메뉴 디자인: 기존 UI 분위기 유지 */
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

    [data-testid="stSidebar"] [data-testid="stRadio"] label p {
        font-size: 14.5px;
        font-weight: 600;
        color: #475569;
        white-space: nowrap;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p {
        color: #2563eb !important;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================
# 2. 기본 루브릭 및 세션 상태
# ==========================================
def default_rubric_binder() -> Dict[str, str]:
    return {
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
        """.strip(),
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
        """.strip(),
        "고등_화학_개념설명형": """
**[영역 1: 핵심 개념 정확성]**
* **우수(35-40점)**: 화학 개념, 용어, 조건을 정확히 사용하고 오개념이 없음
* **보통(20-34점)**: 핵심 개념은 대체로 맞지만 일부 용어 사용이나 조건 설명이 부족함
* **미흡(0-19점)**: 핵심 개념 설명이 부정확하거나 중요한 오개념이 있음

**[영역 2: 근거와 연결성]**
* **우수(25-30점)**: 현상, 입자 수준 설명, 식 또는 그래프를 논리적으로 연결함
* **보통(15-24점)**: 일부 근거를 제시하지만 개념 간 연결이 충분하지 않음
* **미흡(0-14점)**: 암기식 문장 또는 단편적 주장에 그침

**[영역 3: 표현의 명료성]**
* **우수(25-30점)**: 고등학생 수준에서 읽기 쉬우며 문장 흐름이 명확함
* **보통(15-24점)**: 의미 전달은 가능하지만 문장 구조가 다소 어색함
* **미흡(0-14점)**: 문장이 불명확하여 평가자가 의도를 파악하기 어려움
        """.strip(),
    }


def init_session_state() -> None:
    if "rubric_binder" not in st.session_state:
        st.session_state.rubric_binder = default_rubric_binder()

    if "menu_expanded" not in st.session_state:
        st.session_state.menu_expanded = True

    if "current_page" not in st.session_state:
        st.session_state.current_page = "ㅤㅤ📄 단일 채점 (텍스트/파일)"

    if "single_history" not in st.session_state:
        st.session_state.single_history = []

    if "generated_rubric_text" not in st.session_state:
        st.session_state.generated_rubric_text = ""

    if "generated_rubric_topic" not in st.session_state:
        st.session_state.generated_rubric_topic = ""


init_session_state()


# ==========================================
# 3. 공용 도구 함수
# ==========================================
def safe_get_secret(key: str) -> str:
    """Streamlit secrets가 없을 때도 앱이 멈추지 않도록 안전하게 읽습니다."""
    try:
        return str(st.secrets.get(key, ""))
    except Exception:
        return ""


def get_api_key_from_inputs(sidebar_value: str) -> str:
    """입력칸 → Streamlit secrets → 환경변수 순서로 API Key를 찾습니다."""
    return (
        sidebar_value.strip()
        or safe_get_secret("GEMINI_API_KEY").strip()
        or os.environ.get("GEMINI_API_KEY", "").strip()
    )


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def escape_text(value: Any) -> str:
    return html.escape(str(value if value is not None else ""))


def strip_code_fence(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    return text


def parse_json_safely(text: str) -> Dict[str, Any]:
    """AI가 JSON 앞뒤에 설명을 붙여도 최대한 읽어냅니다."""
    cleaned = strip_code_fence(text)

    try:
        return json.loads(cleaned)
    except Exception:
        pass

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    return {
        "student_id": "",
        "student_name": "",
        "total_score": None,
        "overall_level": "확인 필요",
        "summary": "AI 응답을 표준 형식으로 읽지 못했습니다. 원문 응답을 확인해 주세요.",
        "criteria": [],
        "strengths": [],
        "improvements": [],
        "next_actions": [],
        "student_feedback": cleaned,
        "teacher_review_note": "응답 형식 확인 필요",
        "possible_concerns": ["JSON 파싱 실패"],
    }


def clamp_score(value: Any, scale: int = DEFAULT_SCORE_SCALE) -> Optional[int]:
    try:
        score = int(round(float(value)))
        return max(0, min(scale, score))
    except Exception:
        return None


def ensure_list(value: Any) -> List[str]:
    if value is None:
        return []

    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]

    if isinstance(value, str):
        lines = [line.strip(" -•\t") for line in value.split("\n")]
        return [line for line in lines if line]

    return [str(value)]


def normalize_evaluation_result(
    data: Dict[str, Any],
    student_id: str = "",
    student_name: str = "",
) -> Dict[str, Any]:
    data = dict(data or {})

    data["student_id"] = str(data.get("student_id") or student_id or "")
    data["student_name"] = str(data.get("student_name") or student_name or "")
    data["total_score"] = clamp_score(data.get("total_score"))
    data["overall_level"] = str(data.get("overall_level") or "확인 필요")
    data["summary"] = str(data.get("summary") or "")
    data["strengths"] = ensure_list(data.get("strengths"))
    data["improvements"] = ensure_list(data.get("improvements"))
    data["next_actions"] = ensure_list(data.get("next_actions"))
    data["possible_concerns"] = ensure_list(data.get("possible_concerns"))
    data["student_feedback"] = str(data.get("student_feedback") or "")
    data["teacher_review_note"] = str(data.get("teacher_review_note") or TEACHER_REVIEW_NOTICE)

    criteria = data.get("criteria") or []
    if not isinstance(criteria, list):
        criteria = []

    cleaned_criteria = []
    for item in criteria:
        if not isinstance(item, dict):
            continue

        cleaned_criteria.append(
            {
                "area": str(item.get("area") or ""),
                "score": clamp_score(item.get("score"), 100),
                "max_score": clamp_score(item.get("max_score"), 100),
                "level": str(item.get("level") or ""),
                "reason": str(item.get("reason") or ""),
                "evidence": str(item.get("evidence") or ""),
            }
        )

    data["criteria"] = cleaned_criteria
    return data


def make_download_link_from_dict(data: Dict[str, str]) -> bytes:
    return json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8-sig")


def make_sample_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "학번": "10101",
                "이름": "홍길동",
                "과제내용": (
                    "이번 실험에서 반응 속도는 농도가 증가할수록 빨라진다고 판단했다. "
                    "입자 충돌 횟수가 증가하기 때문이다. 다만 온도 통제가 완전하지 않아 "
                    "오차가 생길 수 있다."
                ),
            },
            {
                "학번": "10102",
                "이름": "김화학",
                "과제내용": (
                    "르샤틀리에 원리에 따르면 평형 상태에서 농도, 압력, 온도가 변하면 "
                    "그 변화를 줄이는 방향으로 반응이 이동한다."
                ),
            },
        ]
    )


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def dataframe_to_xlsx_bytes(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="AI채점결과")
    return output.getvalue()


def read_uploaded_table(uploaded_file) -> pd.DataFrame:
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".csv"):
        try:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding="utf-8-sig")
        except Exception:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding="cp949")

    uploaded_file.seek(0)
    return pd.read_excel(uploaded_file)


def save_uploaded_file_to_temp(uploaded_file) -> str:
    suffix = "." + uploaded_file.name.split(".")[-1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        return tmp.name


def truncate_text_for_prompt(text: str, max_chars: int = MAX_TEXT_CHARS) -> Tuple[str, bool]:
    text = str(text or "").strip()

    if len(text) <= max_chars:
        return text, False

    return text[:max_chars] + "\n\n[안내: 과제 내용이 길어 앞부분만 평가에 사용되었습니다.]", True


def auto_guess_column(columns: List[str], candidates: List[str]) -> Optional[str]:
    normalized = {
        str(col).strip().replace(" ", "").lower(): col
        for col in columns
    }

    for cand in candidates:
        key = cand.strip().replace(" ", "").lower()
        if key in normalized:
            return normalized[key]

    for col in columns:
        compact = str(col).strip().replace(" ", "").lower()
        if any(cand.strip().replace(" ", "").lower() in compact for cand in candidates):
            return col

    return None


# ==========================================
# 4. Gemini AI 엔진
# ==========================================
EVALUATION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "student_id": {
            "type": "string",
            "description": "학생 학번. 제공되지 않으면 빈 문자열",
        },
        "student_name": {
            "type": "string",
            "description": "학생 이름. 제공되지 않으면 빈 문자열",
        },
        "total_score": {
            "type": "integer",
            "description": "100점 만점 총점. 확인 불가하면 0",
            "minimum": 0,
            "maximum": 100,
        },
        "overall_level": {
            "type": "string",
            "description": "전체 성취 수준",
            "enum": ["우수", "보통", "미흡", "확인 필요"],
        },
        "summary": {
            "type": "string",
            "description": "교사용 한 줄 요약",
        },
        "criteria": {
            "type": "array",
            "description": "루브릭 영역별 세부 평가",
            "items": {
                "type": "object",
                "properties": {
                    "area": {
                        "type": "string",
                        "description": "평가 영역명",
                    },
                    "score": {
                        "type": "integer",
                        "description": "해당 영역 점수",
                        "minimum": 0,
                        "maximum": 100,
                    },
                    "max_score": {
                        "type": "integer",
                        "description": "해당 영역 만점",
                        "minimum": 0,
                        "maximum": 100,
                    },
                    "level": {
                        "type": "string",
                        "description": "영역 성취 수준",
                        "enum": ["우수", "보통", "미흡", "확인 필요"],
                    },
                    "reason": {
                        "type": "string",
                        "description": "채점 이유",
                    },
                    "evidence": {
                        "type": "string",
                        "description": "학생 답안에서 확인한 근거. 직접 인용은 짧게",
                    },
                },
                "required": ["area", "score", "max_score", "level", "reason", "evidence"],
            },
        },
        "strengths": {
            "type": "array",
            "items": {"type": "string"},
            "description": "잘한 점 2~3개",
        },
        "improvements": {
            "type": "array",
            "items": {"type": "string"},
            "description": "보완할 점 2~3개",
        },
        "next_actions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "다음 학습 행동 2~3개",
        },
        "student_feedback": {
            "type": "string",
            "description": "학생에게 전달할 따뜻한 종합 피드백 3~5문장",
        },
        "teacher_review_note": {
            "type": "string",
            "description": "교사가 다시 확인해야 할 부분",
        },
        "possible_concerns": {
            "type": "array",
            "items": {"type": "string"},
            "description": "판독 불가, 근거 부족, 점수 애매함 등 주의사항",
        },
    },
    "required": [
        "student_id",
        "student_name",
        "total_score",
        "overall_level",
        "summary",
        "criteria",
        "strengths",
        "improvements",
        "next_actions",
        "student_feedback",
        "teacher_review_note",
        "possible_concerns",
    ],
}


def build_evaluation_prompt(
    rubric: str,
    student_id: str = "",
    student_name: str = "",
    assignment_title: str = "",
    text_content: str = "",
) -> str:
    assignment_title = assignment_title.strip() or "서술형 과제"
    student_label = f"{student_name} 학생" if student_name else "학생"

    return f"""
너는 고등학교 학생을 지도하는 전문적이고 따뜻한 교사이자 평가 보조자다.
아래 루브릭을 기준으로 {assignment_title}을 평가하라.

[평가 원칙]
1. 반드시 루브릭에 근거해 채점한다.
2. 학생 답안에 없는 내용을 추측해서 칭찬하거나 감점하지 않는다.
3. 과학 개념 오류가 있으면 부드럽지만 분명하게 짚는다.
4. 전체 점수는 100점 만점으로 환산한다. 루브릭에 영역별 배점이 있으면 그 배점을 우선 적용한다.
5. 영역별 점수, 성취 수준, 채점 이유, 답안 근거를 분리해서 작성한다.
6. 학생 피드백은 {student_label}에게 바로 전달할 수 있는 문장으로 쓴다.
7. 학번/이름이 제공되면 JSON의 student_id, student_name에도 그대로 넣는다.
8. 결과는 지정된 JSON 형식으로만 작성한다.
9. 개인정보 보호를 위해 불필요한 개인정보를 새로 만들거나 추정하지 않는다.
10. 판독이 어렵거나 근거가 부족하면 possible_concerns와 teacher_review_note에 명시한다.

[학생 정보]
- 학번: {student_id or "제공되지 않음"}
- 이름: {student_name or "제공되지 않음"}

[세부 루브릭]
{rubric}

[학생 과제 내용]
{text_content if text_content else "파일이 첨부된 경우 첨부 파일의 내용을 읽어 평가하라."}
""".strip()


def get_gemini_client(api_key: str):
    return genai.Client(api_key=api_key)


def evaluate_with_gemini(
    api_key: str,
    rubric: str,
    text_content: Optional[str] = None,
    uploaded_file_path: Optional[str] = None,
    student_id: str = "",
    student_name: str = "",
    assignment_title: str = "",
) -> Tuple[Dict[str, Any], str]:
    """
    Gemini로 평가하고, 표준화된 dict와 원문 응답을 함께 돌려줍니다.
    """
    if not api_key:
        raise ValueError("API Key가 없습니다.")

    if not rubric.strip():
        raise ValueError("루브릭이 비어 있습니다.")

    prepared_text, truncated = truncate_text_for_prompt(text_content or "")

    prompt = build_evaluation_prompt(
        rubric=rubric,
        student_id=student_id,
        student_name=student_name,
        assignment_title=assignment_title,
        text_content=prepared_text,
    )

    if truncated:
        prompt += "\n\n[교사용 주의] 과제 내용이 길어 일부만 제공되었으므로 teacher_review_note에 길이 제한 가능성을 언급하라."

    client = get_gemini_client(api_key)
    uploaded_resource = None

    try:
        if uploaded_file_path:
            uploaded_resource = client.files.upload(file=uploaded_file_path)
            contents = [uploaded_resource, prompt]
        else:
            contents = prompt

        response = client.models.generate_content(
            model=FIXED_MODEL_NAME,
            contents=contents,
            config={
                "response_format": {
                    "text": {
                        "mime_type": "application/json",
                        "schema": EVALUATION_SCHEMA,
                    }
                },
                "temperature": 0.2,
            },
        )

        raw_text = response.text or ""
        parsed = parse_json_safely(raw_text)
        normalized = normalize_evaluation_result(
            parsed,
            student_id=student_id,
            student_name=student_name,
        )
        return normalized, raw_text

    finally:
        if uploaded_resource is not None:
            try:
                client.files.delete(name=uploaded_resource.name)
            except Exception:
                pass


def generate_rubric_with_gemini(
    api_key: str,
    topic: str,
    subject: str,
    score_scale: int = 100,
) -> str:
    if not api_key:
        raise ValueError("API Key가 없습니다.")

    if not topic.strip():
        raise ValueError("수행평가 주제를 입력해 주세요.")

    prompt = f"""
너는 고등학교 {subject or "과학"} 교사와 함께 수행평가 루브릭을 설계하는 교육평가 전문가다.
'{topic}'에 대한 분석적 루브릭을 만들어라.

[요구 사항]
1. 총점은 {score_scale}점 만점으로 설계한다.
2. 3~5개 평가 영역으로 나눈다.
3. 각 영역은 우수/보통/미흡 3단계로 제시한다.
4. 고등학생 수준에서 이해 가능한 표현을 사용한다.
5. 교사가 바로 복사해 사용할 수 있게 Markdown 형식으로 작성한다.
6. 과학 개념 평가에서는 정확성, 근거, 자료 해석, 오차 분석 또는 논리적 설명을 적절히 포함한다.
""".strip()

    client = get_gemini_client(api_key)
    response = client.models.generate_content(
        model=FIXED_MODEL_NAME,
        contents=prompt,
        config={"temperature": 0.4},
    )
    return response.text or ""


# ==========================================
# 5. 화면 출력 도구
# ==========================================
def render_header(title: str, subtitle: str) -> None:
    st.markdown(
        f"<div class='main-title'>{escape_text(title)}</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='sub-title'>{escape_text(subtitle)}</div>",
        unsafe_allow_html=True,
    )


def render_notice(text: str) -> None:
    st.markdown(
        f"<div class='notice-card'>{escape_text(text)}</div>",
        unsafe_allow_html=True,
    )


def render_warning_card(text: str) -> None:
    st.markdown(
        f"<div class='danger-card'>{escape_text(text)}</div>",
        unsafe_allow_html=True,
    )


def render_result_box(result: Dict[str, Any]) -> None:
    score = result.get("total_score")
    score_text = "확인 필요" if score is None else f"{score}점 / 100점"
    level_text = result.get("overall_level", "확인 필요")

    criteria_html = ""
    for item in result.get("criteria", []):
        area = escape_text(item.get("area", ""))
        score_part = item.get("score")
        max_part = item.get("max_score")

        score_line = "확인 필요" if score_part is None else f"{score_part}점"
        if max_part is not None:
            score_line += f" / {max_part}점"

        criteria_html += f"""
        <div style='margin-top:14px; padding:14px; border-radius:10px; background:#ffffff; border:1px solid #e2e8f0;'>
            <b>{area}</b><br>
            <span class='small-muted'>점수: {escape_text(score_line)} · 수준: {escape_text(item.get('level', ''))}</span><br>
            <span>{escape_text(item.get('reason', ''))}</span><br>
            <span class='small-muted'>근거: {escape_text(item.get('evidence', ''))}</span>
        </div>
        """

    def list_to_html(title: str, values: List[str]) -> str:
        if not values:
            return ""

        lis = "".join(f"<li>{escape_text(v)}</li>" for v in values)
        return f"<b>{escape_text(title)}</b><ul>{lis}</ul>"

    html_block = f"""
    <div class='result-box'>
        <div style='margin-bottom:14px;'><b>[AI 채점 결과]</b></div>
        <span class='score-pill'>{escape_text(score_text)}</span>
        <span class='level-pill'>{escape_text(level_text)}</span>

        <p style='margin-top:16px;'><b>교사용 요약</b><br>{escape_text(result.get('summary', ''))}</p>

        {criteria_html}

        <div style='margin-top:16px;'>
            {list_to_html('잘한 점', result.get('strengths', []))}
            {list_to_html('보완할 점', result.get('improvements', []))}
            {list_to_html('다음 학습 행동', result.get('next_actions', []))}
        </div>

        <p><b>학생 피드백</b><br>{escape_text(result.get('student_feedback', ''))}</p>

        <p class='small-muted'><b>교사 확인 메모</b><br>{escape_text(result.get('teacher_review_note', TEACHER_REVIEW_NOTICE))}</p>
    </div>
    """

    st.markdown(html_block, unsafe_allow_html=True)

    concerns = result.get("possible_concerns", [])
    if concerns:
        render_warning_card("교사 확인 필요: " + " / ".join(concerns))


def criteria_to_text(criteria: List[Dict[str, Any]]) -> str:
    lines = []

    for item in criteria or []:
        area = item.get("area", "")
        score = item.get("score", "")
        max_score = item.get("max_score", "")
        level = item.get("level", "")
        reason = item.get("reason", "")
        lines.append(f"[{area}] {score}/{max_score}점, {level}: {reason}")

    return "\n".join(lines)


def result_to_row(
    result: Dict[str, Any],
    raw_text: str,
    rubric_name: str,
    assignment_title: str,
) -> Dict[str, Any]:
    return {
        "채점시각": now_text(),
        "모델": FIXED_MODEL_NAME,
        "루브릭": rubric_name,
        "과제명": assignment_title,
        "학번": result.get("student_id", ""),
        "이름": result.get("student_name", ""),
        "AI_총점": result.get("total_score"),
        "AI_수준": result.get("overall_level", ""),
        "AI_요약": result.get("summary", ""),
        "AI_세부점수": criteria_to_text(result.get("criteria", [])),
        "AI_강점": "\n".join(result.get("strengths", [])),
        "AI_보완점": "\n".join(result.get("improvements", [])),
        "AI_다음학습": "\n".join(result.get("next_actions", [])),
        "AI_학생피드백": result.get("student_feedback", ""),
        "AI_교사확인": result.get("teacher_review_note", ""),
        "AI_주의사항": "\n".join(result.get("possible_concerns", [])),
        "AI_원문JSON": raw_text,
    }


def render_rubric_manager(selected_rubric_name: str) -> None:
    current_text = st.session_state.rubric_binder[selected_rubric_name]

    with st.expander("📌 선택된 루브릭 상세 보기 / 수정 / 삭제", expanded=False):
        st.markdown("<div class='rubric-preview'>", unsafe_allow_html=True)
        st.markdown(current_text)
        st.markdown("</div>", unsafe_allow_html=True)

        edited_text = st.text_area(
            "루브릭 수정",
            value=current_text,
            height=260,
            key=f"edit_{selected_rubric_name}",
        )

        c1, c2, c3 = st.columns([1, 1, 2])

        with c1:
            if st.button("💾 수정 저장", key=f"save_{selected_rubric_name}"):
                if edited_text.strip():
                    st.session_state.rubric_binder[selected_rubric_name] = edited_text.strip()
                    st.success("수정한 루브릭을 저장했습니다.")
                    st.rerun()
                else:
                    st.error("루브릭 내용을 비워둘 수 없습니다.")

        with c2:
            if st.button("🗑️ 삭제", key=f"delete_{selected_rubric_name}"):
                if len(st.session_state.rubric_binder) <= 1:
                    st.error("최소 1개의 루브릭은 남겨두어야 합니다.")
                else:
                    del st.session_state.rubric_binder[selected_rubric_name]
                    st.success("루브릭을 삭제했습니다.")
                    st.rerun()

        with c3:
            st.download_button(
                "📤 전체 루브릭 바인더 백업(JSON)",
                data=make_download_link_from_dict(st.session_state.rubric_binder),
                file_name="rubric_binder_backup.json",
                mime="application/json",
                use_container_width=True,
            )

    with st.expander("➕ 새 루브릭 직접 추가 / 백업 불러오기", expanded=False):
        new_name = st.text_input(
            "새 루브릭 이름",
            placeholder="예: 화학 반응 속도 실험 평가",
        )
        new_content = st.text_area(
            "새 루브릭 상세 내용",
            height=180,
            placeholder="영역별 기준과 배점을 입력하세요.",
        )

        if st.button("💾 새 루브릭 저장"):
            if not new_name.strip() or not new_content.strip():
                st.error("이름과 내용을 모두 입력해 주세요.")
            elif new_name.strip() in st.session_state.rubric_binder:
                st.error("이미 같은 이름의 루브릭이 있습니다.")
            else:
                st.session_state.rubric_binder[new_name.strip()] = new_content.strip()
                st.success("새 루브릭을 저장했습니다.")
                st.rerun()

        st.markdown("---")

        backup_file = st.file_uploader(
            "루브릭 바인더 백업(JSON) 불러오기",
            type=["json"],
            key="rubric_backup_loader",
        )

        if backup_file and st.button("📥 백업 불러와 병합"):
            try:
                loaded = json.loads(backup_file.getvalue().decode("utf-8-sig"))

                if not isinstance(loaded, dict):
                    st.error("JSON 형식이 올바르지 않습니다.")
                else:
                    for k, v in loaded.items():
                        if str(k).strip() and str(v).strip():
                            st.session_state.rubric_binder[str(k).strip()] = str(v).strip()

                    st.success("백업 루브릭을 현재 바인더에 병합했습니다.")
                    st.rerun()

            except Exception as exc:
                st.error(f"백업 파일을 읽지 못했습니다: {exc}")


# ==========================================
# 6. 사이드바
# ==========================================
with st.sidebar:
    st.markdown(
        "<h2 style='color:#2563eb; font-weight:800; margin-bottom: 20px;'>🏫 AI 평가 보조</h2>",
        unsafe_allow_html=True,
    )

    menu_options = (
        [
            "📂 AI 서술형 채점 ▾",
            "ㅤㅤ📄 단일 채점 (텍스트/파일)",
            "ㅤㅤ📁 학급 전체 채점 (엑셀/CSV)",
            "✨ AI 루브릭 설계",
        ]
        if st.session_state.menu_expanded
        else [
            "📂 AI 서술형 채점 ▸",
            "✨ AI 루브릭 설계",
        ]
    )

    try:
        current_index = menu_options.index(st.session_state.current_page)
    except Exception:
        current_index = 0

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

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("🔒 **시스템 설정**")

    api_key_input = st.text_input(
        "Google API Key 입력",
        type="password",
        placeholder="AIzaSy...",
    )
    api_key = get_api_key_from_inputs(api_key_input)

    st.markdown(
        f"""
        <div class='soft-card'>
            <b>고정 AI 모델</b><br>
            <span class='small-muted'>{escape_text(FIXED_MODEL_NAME)}</span><br><br>
            <b>저장된 루브릭</b><br>
            <span class='small-muted'>{len(st.session_state.rubric_binder)}개</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not api_key:
        st.caption("")


# ==========================================
# 7. 메뉴 1: 단일 채점
# ==========================================
if st.session_state.current_page == "ㅤㅤ📄 단일 채점 (텍스트/파일)":
    render_header(
        "📋 AI 서술형 채점",
        "선생님만의 루브릭을 유지하면서 단일 과제를 심층 평가합니다.",
    )
    render_notice(TEACHER_REVIEW_NOTICE)

    st.markdown("#### ⚙️ 평가 기준 선택")

    rubric_options = list(st.session_state.rubric_binder.keys())
    selected_rubric_name = st.selectbox("사용할 루브릭", rubric_options)
    current_rubric_text = st.session_state.rubric_binder[selected_rubric_name]

    render_rubric_manager(selected_rubric_name)

    st.markdown("---")
    st.markdown("#### 📄 학생 정보 및 과제 입력")

    assignment_title = st.text_input(
        "과제명",
        placeholder="예: 화학 반응 속도 실험보고서",
    )

    col1, col2 = st.columns(2)
    with col1:
        single_std_id = st.text_input("학생 학번 (선택)", placeholder="예: 10101")
    with col2:
        single_std_name = st.text_input("학생 이름 (선택)", placeholder="예: 홍길동")

    upload_type = st.radio(
        "입력 방식",
        ["직접 텍스트 입력", "파일 업로드 (PDF/사진)"],
        horizontal=True,
    )

    student_text = ""
    uploaded_file_path = None
    tmp_path = None

    if upload_type == "직접 텍스트 입력":
        student_text = st.text_area(
            "학생의 글을 붙여넣으세요.",
            height=190,
            placeholder="학생 답안을 여기에 붙여넣으세요.",
        )
        st.caption(f"현재 글자 수: {len(student_text):,}자 / 권장 최대 {MAX_TEXT_CHARS:,}자")

    else:
        uploaded_file = st.file_uploader(
            "파일 업로드",
            type=["pdf", "png", "jpg", "jpeg"],
        )

        if uploaded_file:
            tmp_path = save_uploaded_file_to_temp(uploaded_file)
            uploaded_file_path = tmp_path
            st.success(f"파일을 준비했습니다: {uploaded_file.name}")

    if st.button("🚀 채점 시작", use_container_width=True):
        if not api_key:
            st.error("왼쪽 사이드바에 Google API Key를 입력해 주세요.")

        elif upload_type == "직접 텍스트 입력" and not student_text.strip():
            st.warning("학생 답안 내용을 입력해 주세요.")

        elif upload_type == "파일 업로드 (PDF/사진)" and not uploaded_file_path:
            st.warning("평가할 파일을 업로드해 주세요.")

        else:
            try:
                with st.spinner("AI가 루브릭을 기준으로 채점 중입니다..."):
                    result, raw_text = evaluate_with_gemini(
                        api_key=api_key,
                        text_content=student_text,
                        uploaded_file_path=uploaded_file_path,
                        rubric=current_rubric_text,
                        student_id=single_std_id,
                        student_name=single_std_name,
                        assignment_title=assignment_title,
                    )

                render_result_box(result)

                row = result_to_row(
                    result,
                    raw_text,
                    selected_rubric_name,
                    assignment_title,
                )
                st.session_state.single_history.insert(0, row)

                result_df = pd.DataFrame([row])

                c1, c2 = st.columns(2)
                with c1:
                    st.download_button(
                        "📥 이번 결과 CSV 다운로드",
                        data=dataframe_to_csv_bytes(result_df),
                        file_name="단일채점결과.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                with c2:
                    st.download_button(
                        "📥 이번 결과 XLSX 다운로드",
                        data=dataframe_to_xlsx_bytes(result_df),
                        file_name="단일채점결과.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                    )

            except Exception as exc:
                st.error(f"채점 중 오류가 발생했습니다: {exc}")

            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass

    if st.session_state.single_history:
        st.markdown("---")
        st.markdown("#### 🗂️ 이번 실행 중 단일 채점 이력")

        history_df = pd.DataFrame(st.session_state.single_history)
        st.dataframe(history_df, use_container_width=True)

        c1, c2, c3 = st.columns([1, 1, 1])

        with c1:
            st.download_button(
                "📥 이력 CSV 다운로드",
                data=dataframe_to_csv_bytes(history_df),
                file_name="단일채점이력.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with c2:
            st.download_button(
                "📥 이력 XLSX 다운로드",
                data=dataframe_to_xlsx_bytes(history_df),
                file_name="단일채점이력.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

        with c3:
            if st.button("🧹 이력 비우기", use_container_width=True):
                st.session_state.single_history = []
                st.rerun()


# ==========================================
# 8. 메뉴 2: 학급 전체 채점
# ==========================================
elif st.session_state.current_page == "ㅤㅤ📁 학급 전체 채점 (엑셀/CSV)":
    render_header(
        "📋 AI 서술형 채점",
        "학급 전체 과제를 엑셀 또는 CSV로 한 번에 평가합니다.",
    )
    render_notice("필수 정보는 학번, 이름, 과제내용입니다. 열 이름이 달라도 아래에서 직접 연결할 수 있습니다.")

    st.markdown("#### 📊 학급 일괄 채점")

    sample_df = make_sample_dataframe()
    c1, c2 = st.columns(2)

    with c1:
        st.download_button(
            "📄 예시 CSV 다운로드",
            data=dataframe_to_csv_bytes(sample_df),
            file_name="AI채점_입력예시.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with c2:
        st.download_button(
            "📄 예시 XLSX 다운로드",
            data=dataframe_to_xlsx_bytes(sample_df),
            file_name="AI채점_입력예시.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    batch_rubric_name = st.selectbox(
        "일괄 채점에 적용할 루브릭",
        list(st.session_state.rubric_binder.keys()),
    )

    with st.expander("📌 선택된 루브릭 상세 보기"):
        st.markdown(st.session_state.rubric_binder[batch_rubric_name])

    assignment_title_batch = st.text_input(
        "과제명",
        placeholder="예: 르샤틀리에 원리 논술형 평가",
        key="batch_assignment_title",
    )

    uploaded_table = st.file_uploader(
        "엑셀 또는 CSV 파일 선택",
        type=["xlsx", "xls", "csv"],
    )

    if uploaded_table:
        try:
            df_input = read_uploaded_table(uploaded_table)
            df_input.columns = [str(c).strip() for c in df_input.columns]

            st.success(f"파일을 읽었습니다. 총 {len(df_input)}행, {len(df_input.columns)}열")
            st.dataframe(df_input.head(10), use_container_width=True)

            columns = list(df_input.columns)

            guessed_id = auto_guess_column(
                columns,
                ["학번", "번호", "student_id", "id"],
            )
            guessed_name = auto_guess_column(
                columns,
                ["이름", "성명", "student_name", "name"],
            )
            guessed_answer = auto_guess_column(
                columns,
                ["과제내용", "답안", "서술형답안", "내용", "answer", "text"],
            )

            st.markdown("##### 🔗 열 연결 확인")

            col1, col2, col3 = st.columns(3)

            with col1:
                id_col = st.selectbox(
                    "학번 열",
                    columns,
                    index=columns.index(guessed_id) if guessed_id in columns else 0,
                )

            with col2:
                name_col = st.selectbox(
                    "이름 열",
                    columns,
                    index=columns.index(guessed_name) if guessed_name in columns else 0,
                )

            with col3:
                answer_col = st.selectbox(
                    "과제내용 열",
                    columns,
                    index=columns.index(guessed_answer) if guessed_answer in columns else 0,
                )

            st.markdown("##### ⚙️ 처리 옵션")

            opt1, opt2, opt3 = st.columns(3)

            with opt1:
                max_rows = st.number_input(
                    "처리할 최대 인원",
                    min_value=1,
                    max_value=max(1, len(df_input)),
                    value=len(df_input),
                    step=1,
                )

            with opt2:
                batch_pause = st.slider(
                    "학생 1명 처리 후 쉬는 시간(초)",
                    min_value=0.0,
                    max_value=5.0,
                    value=0.5,
                    step=0.5,
                )

            with opt3:
                skip_blank = st.checkbox("빈 답안은 건너뛰기", value=True)

            if st.button("🚀 일괄 채점 시작", use_container_width=True):
                if not api_key:
                    st.error("왼쪽 사이드바에 Google API Key를 입력해 주세요.")

                else:
                    df_work = df_input.head(int(max_rows)).copy()
                    result_rows = []

                    progress = st.progress(0, text="채점 준비 중...")
                    status_area = st.empty()

                    for processed_idx, (_, row) in enumerate(df_work.iterrows(), start=1):
                        std_id = "" if pd.isna(row.get(id_col, "")) else str(row.get(id_col, "")).strip()
                        std_name = "" if pd.isna(row.get(name_col, "")) else str(row.get(name_col, "")).strip()
                        answer_text = "" if pd.isna(row.get(answer_col, "")) else str(row.get(answer_col, "")).strip()

                        status_area.info(f"{processed_idx}/{len(df_work)} 처리 중: {std_id} {std_name}")

                        if skip_blank and not answer_text:
                            result = normalize_evaluation_result(
                                {
                                    "student_id": std_id,
                                    "student_name": std_name,
                                    "total_score": 0,
                                    "overall_level": "확인 필요",
                                    "summary": "과제 내용이 비어 있어 채점하지 않았습니다.",
                                    "criteria": [],
                                    "strengths": [],
                                    "improvements": ["과제 내용을 제출해야 평가할 수 있습니다."],
                                    "next_actions": ["답안을 작성한 뒤 다시 제출하세요."],
                                    "student_feedback": (
                                        f"{std_name + ' 학생, ' if std_name else ''}"
                                        "제출된 과제 내용이 없어 이번에는 평가를 진행하지 못했습니다. "
                                        "답안을 작성한 뒤 다시 확인해 봅시다."
                                    ),
                                    "teacher_review_note": "빈 답안으로 자동 건너뜀",
                                    "possible_concerns": ["빈 답안"],
                                },
                                student_id=std_id,
                                student_name=std_name,
                            )
                            raw_text = json.dumps(result, ensure_ascii=False)

                        else:
                            try:
                                result, raw_text = evaluate_with_gemini(
                                    api_key=api_key,
                                    text_content=answer_text,
                                    uploaded_file_path=None,
                                    rubric=st.session_state.rubric_binder[batch_rubric_name],
                                    student_id=std_id,
                                    student_name=std_name,
                                    assignment_title=assignment_title_batch,
                                )

                            except Exception as exc:
                                result = normalize_evaluation_result(
                                    {
                                        "student_id": std_id,
                                        "student_name": std_name,
                                        "total_score": 0,
                                        "overall_level": "확인 필요",
                                        "summary": f"오류로 채점하지 못했습니다: {exc}",
                                        "criteria": [],
                                        "strengths": [],
                                        "improvements": [],
                                        "next_actions": [],
                                        "student_feedback": "채점 과정에서 오류가 발생하여 교사 확인이 필요합니다.",
                                        "teacher_review_note": f"API 또는 입력 오류 확인 필요: {exc}",
                                        "possible_concerns": ["처리 오류"],
                                    },
                                    student_id=std_id,
                                    student_name=std_name,
                                )
                                raw_text = json.dumps(result, ensure_ascii=False)

                        result_rows.append(
                            result_to_row(
                                result,
                                raw_text,
                                batch_rubric_name,
                                assignment_title_batch,
                            )
                        )

                        progress.progress(
                            int(processed_idx / len(df_work) * 100),
                            text=f"{processed_idx}/{len(df_work)}명 처리 완료",
                        )

                        if batch_pause > 0:
                            time.sleep(batch_pause)

                    result_df = pd.DataFrame(result_rows)
                    merged_df = pd.concat(
                        [
                            df_work.reset_index(drop=True),
                            result_df.reset_index(drop=True),
                        ],
                        axis=1,
                    )

                    progress.progress(100, text="채점 완료")
                    status_area.success("✅ 일괄 채점이 완료되었습니다. 교사 검토 후 확정해 주세요.")

                    st.dataframe(merged_df, use_container_width=True)

                    d1, d2 = st.columns(2)

                    with d1:
                        st.download_button(
                            "📥 결과 CSV 다운로드",
                            data=dataframe_to_csv_bytes(merged_df),
                            file_name="일괄채점결과.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )

                    with d2:
                        st.download_button(
                            "📥 결과 XLSX 다운로드",
                            data=dataframe_to_xlsx_bytes(merged_df),
                            file_name="일괄채점결과.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True,
                        )

        except Exception as exc:
            st.error(f"파일을 읽는 중 오류가 발생했습니다: {exc}")


# ==========================================
# 9. 메뉴 3: AI 루브릭 설계
# ==========================================
elif st.session_state.current_page == "✨ AI 루브릭 설계":
    render_header(
        "✨ AI 루브릭 설계",
        "수행평가 주제 기반 3단계 성취수준 루브릭을 생성하고 바인더에 바로 저장합니다.",
    )
    render_notice("생성된 루브릭은 초안입니다. 학교 평가 계획, 성취기준, 배점 정책에 맞게 교사가 최종 수정해 주세요.")

    st.markdown("#### 🪄 AI 루브릭 설계소")

    col1, col2 = st.columns([2, 1])

    with col1:
        rubric_topic = st.text_input(
            "수행평가 주제",
            placeholder="예: 르샤틀리에 원리 실험 보고서",
        )

    with col2:
        subject = st.text_input("과목/단원", value="화학")

    score_scale = st.number_input(
        "총점",
        min_value=10,
        max_value=100,
        value=100,
        step=5,
    )

    if st.button("🪄 설계 요청", use_container_width=True):
        if not api_key:
            st.error("왼쪽 사이드바에 Google API Key를 입력해 주세요.")

        elif not rubric_topic.strip():
            st.warning("수행평가 주제를 입력해 주세요.")

        else:
            try:
                with st.spinner("루브릭 초안을 설계 중입니다..."):
                    generated = generate_rubric_with_gemini(
                        api_key,
                        rubric_topic,
                        subject,
                        int(score_scale),
                    )

                st.session_state.generated_rubric_text = generated
                st.session_state.generated_rubric_topic = rubric_topic
                st.success("루브릭 초안을 생성했습니다.")

            except Exception as exc:
                st.error(f"루브릭 생성 중 오류가 발생했습니다: {exc}")

    if st.session_state.generated_rubric_text:
        st.markdown("---")
        st.markdown("#### 📌 추천 루브릭")

        safe_rubric_html = escape_text(st.session_state.generated_rubric_text).replace("\n", "<br>")
        st.markdown(
            f"<div class='result-box'>{safe_rubric_html}</div>",
            unsafe_allow_html=True,
        )

        save_name = st.text_input(
            "바인더에 저장할 이름",
            value=st.session_state.generated_rubric_topic or "새 루브릭",
            key="generated_save_name",
        )

        c1, c2 = st.columns(2)

        with c1:
            if st.button("💾 이 루브릭을 바인더에 저장", use_container_width=True):
                if not save_name.strip():
                    st.error("저장할 이름을 입력해 주세요.")
                else:
                    st.session_state.rubric_binder[save_name.strip()] = (
                        st.session_state.generated_rubric_text.strip()
                    )
                    st.success("루브릭 바인더에 저장했습니다. 단일/일괄 채점 메뉴에서 선택할 수 있습니다.")

        with c2:
            st.download_button(
                "📥 루브릭 TXT 다운로드",
                data=st.session_state.generated_rubric_text.encode("utf-8-sig"),
                file_name="AI_추천루브릭.txt",
                mime="text/plain",
                use_container_width=True,
            )