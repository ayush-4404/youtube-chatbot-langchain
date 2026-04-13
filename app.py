from urllib.parse import parse_qs, urlparse
import json
import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from youtube_transcript_api import NoTranscriptFound, TranscriptsDisabled, YouTubeTranscriptApi
import yt_dlp

load_dotenv()


PROMPT_TEMPLATE = """
You are a helpful assistant.
Answer ONLY from the provided transcript context.
If the context is insufficient, just say you don't know.

Conversation history:
{history}

Transcript context:
{context}
Current user question: {question}
""".strip()

PREFERRED_TRANSCRIPT_LANGUAGES = ["hi", "hi-IN", "en"]


def extract_video_id(url_or_id: str) -> str:
    value = url_or_id.strip()
    if not value:
        raise ValueError("Please enter a YouTube URL or video id.")

    if "youtube.com" not in value and "youtu.be" not in value:
        return value

    parsed = urlparse(value)

    if "youtu.be" in parsed.netloc:
        video_id = parsed.path.strip("/")
        if video_id:
            return video_id

    query_video_id = parse_qs(parsed.query).get("v", [""])[0]
    if query_video_id:
        return query_video_id

    path_parts = [part for part in parsed.path.split("/") if part]
    if len(path_parts) >= 2 and path_parts[0] in {"shorts", "embed"}:
        return path_parts[1]

    raise ValueError("Could not detect a valid YouTube video id from that URL.")


def fetch_transcript_text(video_id: str) -> str:
    transcript_list = YouTubeTranscriptApi().fetch(
        video_id,
        languages=PREFERRED_TRANSCRIPT_LANGUAGES,
    )
    return " ".join(chunk.text for chunk in transcript_list)


def fetch_transcript_text_ytdlp(video_id: str) -> str:
    with tempfile.TemporaryDirectory() as tmp_dir:
        ydl_opts = {
            "skip_download": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": PREFERRED_TRANSCRIPT_LANGUAGES,
            "subtitlesformat": "json3",
            "outtmpl": os.path.join(tmp_dir, "%(id)s.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={video_id}"])

        json3_files = [
            os.path.join(tmp_dir, name)
            for name in os.listdir(tmp_dir)
            if name.startswith(video_id) and name.endswith(".json3")
        ]

        if not json3_files:
            raise ValueError("No subtitles found via yt-dlp.")

        combined_lines = []
        for file_path in sorted(json3_files):
            with open(file_path, "r", encoding="utf-8") as fp:
                subtitle_json = json.load(fp)

            for event in subtitle_json.get("events", []):
                segments = event.get("segs", [])
                line = "".join(seg.get("utf8", "") for seg in segments).strip()
                if line:
                    combined_lines.append(line)

        transcript_text = " ".join(combined_lines).strip()
        if not transcript_text:
            raise ValueError("yt-dlp subtitles were empty.")

        return transcript_text


def build_retriever(transcript: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    return retriever, len(chunks)


def format_history(messages, max_turns: int = 8) -> str:
    if not messages:
        return "No previous conversation."

    # Keep only recent turns so prompts stay focused and token usage remains stable.
    recent_messages = messages[-(max_turns * 2) :]
    lines = []
    for message in recent_messages:
        role = "User" if message.get("role") == "user" else "Assistant"
        content = str(message.get("content", "")).strip()
        if content:
            lines.append(f"{role}: {content}")

    return "\n".join(lines) if lines else "No previous conversation."


def answer_question(retriever, question: str, history_messages) -> str:
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    history = format_history(history_messages)

    prompt = PROMPT_TEMPLATE.format(history=history, context=context, question=question)
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0.2)
    response = llm.invoke(prompt)

    return response.content if hasattr(response, "content") else str(response)


st.set_page_config(page_title="YouTube Chatbot", page_icon="🎬", layout="centered")
st.title("YouTube Video Q&A")
st.caption("Ask questions about a YouTube video's transcript using Gemini + LangChain")
st.caption("If YouTube blocks transcript requests on cloud IPs, you can paste a transcript manually below.")

if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "active_video_id" not in st.session_state:
    st.session_state.active_video_id = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

video_input = st.text_input("YouTube URL or video id", placeholder="https://www.youtube.com/watch?v=...")

if st.button("Load Video", use_container_width=True):
    try:
        video_id = extract_video_id(video_input)
        transcript_source = "youtube_transcript_api"
        with st.spinner("Fetching transcript and building vector store..."):
            try:
                transcript_text = fetch_transcript_text(video_id)
            except Exception as primary_exc:
                primary_error = str(primary_exc)
                if (
                    "YouTube is blocking requests from your IP" in primary_error
                    or "RequestBlocked" in primary_error
                    or "IpBlocked" in primary_error
                ):
                    transcript_text = fetch_transcript_text_ytdlp(video_id)
                    transcript_source = "yt-dlp subtitles fallback"
                else:
                    raise

            retriever, chunk_count = build_retriever(transcript_text)

        st.session_state.retriever = retriever
        st.session_state.active_video_id = video_id
        st.session_state.chat_history = []
        st.success(f"Video loaded from {transcript_source}. Created {chunk_count} chunks.")
    except (TranscriptsDisabled, NoTranscriptFound):
        st.error("No Hindi or English transcript found for this video.")
    except Exception as exc:
        error_text = str(exc)
        if (
            "YouTube is blocking requests from your IP" in error_text
            or "RequestBlocked" in error_text
            or "IpBlocked" in error_text
        ):
            st.error("Transcript request was blocked by YouTube from this IP.")
            st.info(
                "Workarounds: run locally on a residential IP, wait and retry later, or paste a transcript manually below."
            )
        else:
            st.error(f"Failed to load video: {exc}")

st.markdown("---")
manual_transcript = st.text_area(
    "Or paste transcript manually",
    placeholder="Paste the video transcript text here, then click 'Use Pasted Transcript'.",
    height=180,
)

if st.button("Use Pasted Transcript", use_container_width=True):
    if not manual_transcript.strip():
        st.warning("Paste transcript text first.")
    else:
        try:
            with st.spinner("Building vector store from pasted transcript..."):
                retriever, chunk_count = build_retriever(manual_transcript)
            st.session_state.retriever = retriever
            st.session_state.active_video_id = "manual-transcript"
            st.session_state.chat_history = []
            st.success(f"Transcript loaded. Created {chunk_count} chunks.")
        except Exception as exc:
            st.error(f"Failed to use pasted transcript: {exc}")

st.markdown("### Chat")
if st.session_state.chat_history and st.button("Clear Chat History"):
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask about the loaded video transcript...")

if user_input:
    if not st.session_state.retriever:
        st.warning("Load a video first.")
    else:
        prior_history = list(st.session_state.chat_history)
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        try:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer = answer_question(st.session_state.retriever, user_input, prior_history)
                st.markdown(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
        except Exception as exc:
            st.error(f"Failed to generate answer: {exc}")

if st.session_state.active_video_id:
    st.caption(f"Current video id: {st.session_state.active_video_id}")
