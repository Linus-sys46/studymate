from collections.abc import Iterator
import logging

import streamlit as st
from pypdf import PdfReader
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from streamlit.errors import StreamlitSecretNotFoundError

MODEL_ID = "gemini-3.8-flash"
logger = logging.getLogger(__name__)


class StudyAssistantError(Exception):
    pass

st.set_page_config(page_title="Student AI Study Assistant", layout="wide")

try:
    api_key = st.secrets.get("GEMINI_API_KEY", "")
except StreamlitSecretNotFoundError:
    api_key = ""

# Initialize session memory once per user session
defaults = {
    "file_id": None,
    "file_name": "",
    "pdf_text": "",
    "page_count": 0,
    "pdf_error": "",
    "study_output": "",
    "study_error": "",
    "chat_history": [],
    "completed_tools": [],
}
for key, default_val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = (
            default_val.copy() if isinstance(default_val, list) else default_val
        )
if "upload_generation" not in st.session_state:
    st.session_state.upload_generation = 0


def reset_conversation_state() -> None:
    for key, default_val in defaults.items():
        st.session_state[key] = (
            default_val.copy() if isinstance(default_val, list) else default_val
        )


def user_facing_ai_error(error: Exception, action: str) -> str:
    if isinstance(error, StudyAssistantError):
        return str(error)

    if isinstance(error, genai_errors.APIError):
        logger.warning(
            "Gemini request failed during %s (HTTP %s, %s).",
            action,
            error.code,
            error.status,
        )
        if error.code == 503:
            return (
                "The AI service is busy right now. Your lecture is still here; "
                "please try again in a moment."
            )
        if error.code == 429:
            return (
                "The AI service is handling too many requests right now. "
                "Please wait a moment and try again."
            )
        if error.code in (500, 502, 504):
            return (
                "The AI service is temporarily unavailable. "
                "Please try again shortly."
            )
        if error.code in (401, 403):
            return (
                "The study service could not authorize this request. "
                "Please contact the app administrator."
            )
        return (
            "The AI service couldn't complete this request. Please try again. "
            "If the issue continues, contact the app administrator."
        )

    logger.exception("Unexpected error during %s.", action)
    return (
        "Something went wrong while preparing your response. "
        "Please try again. If the issue continues, contact the app administrator."
    )


with st.sidebar:
    st.header("Setup")
    if st.button("New conversation", use_container_width=True):
        reset_conversation_state()
        st.session_state.upload_generation += 1
        st.rerun()

    uploaded_file = st.file_uploader(
        "Upload Lecture PDF",
        type=["pdf"],
        key=f"lecture_pdf_{st.session_state.upload_generation}",
    )
    if uploaded_file is not None:
        current_file_id = f"{uploaded_file.name}:{uploaded_file.size}"
        if st.session_state.file_id != current_file_id:
            reset_conversation_state()
            st.session_state.file_id = current_file_id
            st.session_state.file_name = uploaded_file.name

            try:
                reader = PdfReader(uploaded_file)
                pages_text = []
                for i, page in enumerate(reader.pages, start=1):
                    text = (page.extract_text() or "").strip()
                    if text:
                        pages_text.append(f"[Page {i}]\n{text}")

                st.session_state.page_count = len(reader.pages)
                st.session_state.pdf_text = "\n\n".join(pages_text)
                if not st.session_state.pdf_text:
                    st.session_state.pdf_error = (
                        "No selectable text could be extracted from this PDF. "
                        "It may be a scanned or image-only document."
                    )
            except Exception as err:
                st.session_state.pdf_error = f"Could not read PDF file: {err}"
    elif st.session_state.file_id is not None:
        reset_conversation_state()

    if st.session_state.pdf_error:
        st.warning(st.session_state.pdf_error)

    document_summary = st.empty()
    activity_summary = st.empty()

st.title("StudyMate")
st.caption("Study tools and Q&A grounded strictly in your uploaded lecture PDF.")

STUDY_SYSTEM_RULE = (
    "You are a university study assistant. Use ONLY the provided lecture PDF text. "
    "Do not invent facts or use outside knowledge. If information is missing, say so clearly."
)

PROMPTS = {
    "summarize": "Summarize these lecture notes clearly with headings and key takeaways:\n\n",
    "key_terms": "Extract the key terms, definitions, and important concepts from these notes:\n\n",
    "quiz": (
        "Generate a 5-question practice quiz based strictly on these lecture notes, "
        "followed by an Answer Key with brief explanations from the notes:\n\n"
    ),
}


def run_gemini(prompt_text: str, system_rule: str) -> Iterator[str]:
    clean_key = (api_key or "").strip()
    if not clean_key:
        raise StudyAssistantError(
            "The study service is not configured yet. Please contact the app administrator."
        )
    if not st.session_state.pdf_text.strip():
        raise StudyAssistantError(
            "Upload a lecture PDF with selectable text before using a study tool."
        )

    client = genai.Client(api_key=clean_key)
    response_stream = client.models.generate_content_stream(
        model=MODEL_ID,
        contents=prompt_text,
        config=types.GenerateContentConfig(
            system_instruction=system_rule,
        ),
    )
    received_text = False
    for chunk in response_stream:
        if chunk.text:
            received_text = True
            yield chunk.text
    if not received_text:
        yield "No response returned."


st.subheader("1. Study Tools")
c1, c2, c3 = st.columns(3)
chosen_tool = None
if c1.button("Summarize Notes", use_container_width=True):
    chosen_tool = "summarize"
if c2.button("Extract Key Terms", use_container_width=True):
    chosen_tool = "key_terms"
if c3.button("Generate 5-Question Quiz", use_container_width=True):
    chosen_tool = "quiz"

if chosen_tool:
    st.session_state.study_error = ""
    st.session_state.study_output = ""
    status_messages = {
        "summarize": "Building a summary from your notes...",
        "key_terms": "Finding key terms in your notes...",
        "quiz": "Preparing a practice quiz from your notes...",
    }
    try:
        with st.spinner(status_messages[chosen_tool], show_time=True):
            full_prompt = PROMPTS[chosen_tool] + st.session_state.pdf_text
            st.session_state.study_output = st.write_stream(
                run_gemini(full_prompt, STUDY_SYSTEM_RULE)
            )
        st.session_state.completed_tools.append(chosen_tool)
    except Exception as err:
        st.session_state.study_error = user_facing_ai_error(
            err,
            "generating study material",
        )

if st.session_state.study_error:
    st.error(st.session_state.study_error)

if st.session_state.study_output and not chosen_tool:
    st.markdown(st.session_state.study_output)

st.divider()
st.subheader("2. Ask Questions About Your Notes")

CHAT_SYSTEM_RULE = (
    "You are a grounded Q&A tutor. Answer ONLY using facts in the uploaded lecture notes. "
    "If the answer is not in the notes, reply: "
    "'I cannot find that information in the uploaded lecture notes.' Do not guess."
)

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_q = st.chat_input("Ask a question about your lecture PDF...")
if user_q:
    st.session_state.chat_history.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    recent_turns = "\n".join(
        f"{m['role'].capitalize()}: {m['content']}"
        for m in st.session_state.chat_history[-7:-1]
    )
    chat_prompt = (
        f"LECTURE NOTES:\n{st.session_state.pdf_text}\n\n"
        f"PRIOR CHAT:\n{recent_turns}\n\n"
        f"QUESTION:\n{user_q}"
    )

    try:
        with st.chat_message("assistant"):
            with st.spinner("Thinking through your notes...", show_time=True):
                reply = st.write_stream(run_gemini(chat_prompt, CHAT_SYSTEM_RULE))
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
    except Exception as err:
        error_msg = user_facing_ai_error(err, "answering a question")
        with st.chat_message("assistant"):
            st.error(error_msg)
        st.session_state.chat_history.append({"role": "assistant", "content": error_msg})

with document_summary.container():
    if st.session_state.file_id and not st.session_state.pdf_error:
        st.subheader("Current lecture")
        st.caption(st.session_state.file_name)
        words = len(st.session_state.pdf_text.split())
        st.metric("Pages", st.session_state.page_count)
        st.metric("Word Count", words)
        with st.expander("Raw Text Preview"):
            st.text(st.session_state.pdf_text[:1200] or "No extractable text found.")
    else:
        st.caption("Upload a lecture PDF to begin.")

with activity_summary.container():
    st.subheader("Activities")
    tool_labels = {
        "summarize": "Summaries generated",
        "key_terms": "Key-term lists generated",
        "quiz": "Quizzes generated",
    }
    has_activity = bool(st.session_state.completed_tools or st.session_state.chat_history)
    if has_activity:
        for tool, label in tool_labels.items():
            count = st.session_state.completed_tools.count(tool)
            if count:
                st.caption(f"{label}: {count}")
        question_count = sum(
            message["role"] == "user" for message in st.session_state.chat_history
        )
        st.caption(f"Questions asked: {question_count}")
    else:
        st.caption("No study activities yet.")