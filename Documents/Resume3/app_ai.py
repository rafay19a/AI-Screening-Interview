import streamlit as st
import tempfile
import os
import numpy as np
import time

from matcher_ai import ai_parse_and_match
from stt_tts import speech_to_text, text_to_speech
from interview_agent import InterviewAgent

from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
import av


# --------------------------------------------------
# REALTIME AUDIO PROCESSOR (CLOUD FIXED)
# --------------------------------------------------
class InterviewAudioProcessor(AudioProcessorBase):

    def __init__(self):
        self.frames = []

    def recv_audio(self, frame: av.AudioFrame):

        audio = frame.to_ndarray()

        # ✅ CRITICAL NORMALIZATION FIX
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32) / 32768.0

        self.frames.append(audio)

        return frame


# --------------------------------------------------
# SESSION STATE INIT
# --------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "matcher"

if "interview_started" not in st.session_state:
    st.session_state.interview_started = False

if "agent" not in st.session_state:
    st.session_state.agent = None

if "current_question" not in st.session_state:
    st.session_state.current_question = None

if "last_results_sorted" not in st.session_state:
    st.session_state.last_results_sorted = []

if "last_jd" not in st.session_state:
    st.session_state.last_jd = ""

if "processing_audio" not in st.session_state:
    st.session_state.processing_audio = False

if "last_processed_time" not in st.session_state:
    st.session_state.last_processed_time = 0


st.set_page_config(
    page_title="AI Resume Matcher + Interview Scheduling",
    layout="wide"
)


# --------------------------------------------------
# MATCHER SCREEN
# --------------------------------------------------
def matcher_screen():

    st.title("🤖 AI Resume Matcher + Interview Scheduling")

    jd = st.text_area("Paste Job Description (JD)", height=200)

    min_skill_match = st.number_input(
        "Minimum Skill Match %",
        min_value=0,
        max_value=100,
        value=40
    )

    uploaded_files = st.file_uploader(
        "Upload Resume PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("Run Matching"):
        if not jd or not uploaded_files:
            st.warning("Please paste JD & upload resumes.")
        else:
            results = []

            for uploaded in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded.read())
                    tmp_path = tmp.name

                parsed = ai_parse_and_match(tmp_path, jd)
                results.append(parsed)

                os.remove(tmp_path)

            results_sorted = sorted(
                results,
                key=lambda x: x.get("semantic_score", 0),
                reverse=True
            )

            st.session_state.last_results_sorted = results_sorted
            st.session_state.last_jd = jd

    results_sorted = st.session_state.last_results_sorted

    if results_sorted:

        filtered = [
            r for r in results_sorted
            if r.get("skill_match_percent", 0) >= min_skill_match
        ]

        for r in filtered:
            st.write(f"**Candidate:** {os.path.basename(r['resume_path'])}")
            st.write(f"- Semantic Score: {r.get('semantic_score')}")
            st.write(f"- Skill Match: {r.get('skill_match_percent')}%")
            st.write("---")

        if st.button("🎙 Go to Voice Interview"):
            st.session_state.page = "start"


# --------------------------------------------------
# INTERVIEW SCREEN
# --------------------------------------------------
def start_screen():

    st.title("🎙 AI Voice Interview")

    if st.button("⬅ Back to Matcher"):
        st.session_state.page = "matcher"
        st.session_state.interview_started = False
        st.session_state.agent = None
        st.session_state.current_question = None
        return

    if not st.session_state.interview_started:
        if st.button("▶ Start Interview"):

            st.session_state.interview_started = True

            resume_text = ""
            if st.session_state.last_results_sorted:
                resume_text = st.session_state.last_results_sorted[0].get("raw_text", "")

            st.session_state.agent = InterviewAgent(
                jd=st.session_state.last_jd,
                resume_text=resume_text
            )

            first_q = st.session_state.agent.next_question()
            st.session_state.current_question = first_q

            st.audio(text_to_speech(first_q))

    if st.session_state.current_question:
        st.info(st.session_state.current_question)

    if st.session_state.interview_started:

        st.markdown("### 🎤 Speak Naturally")

        ctx = webrtc_streamer(
            key="interview_stream",
            audio_processor_factory=InterviewAudioProcessor,
            media_stream_constraints={"audio": True, "video": False},
            async_processing=True,
        )

        SPEECH_THRESHOLD = 0.01
        COOLDOWN_SECONDS = 2

        if ctx.audio_processor:

            frames = ctx.audio_processor.frames

            st.write("Frame Count:", len(frames))

            if len(frames) > 30 and not st.session_state.processing_audio:

                audio_data = np.concatenate(frames)
                volume = np.abs(audio_data).mean()

                st.write("Detected Volume:", volume)

                now = time.time()

                if (
                    volume > SPEECH_THRESHOLD
                    and now - st.session_state.last_processed_time > COOLDOWN_SECONDS
                ):

                    st.session_state.processing_audio = True
                    st.session_state.last_processed_time = now

                    with st.spinner("Listening..."):
                        answer_text = speech_to_text(audio_data.tobytes())

                    st.write("**You said:**", answer_text)

                    with st.spinner("AI Thinking..."):
                        next_q = st.session_state.agent.next_question(answer_text)

                    st.session_state.current_question = next_q

                    st.audio(text_to_speech(next_q))

                    ctx.audio_processor.frames = []

                    st.session_state.processing_audio = False

        if st.button("Finish Interview"):
            st.session_state.interview_started = False
            st.session_state.agent = None
            st.session_state.current_question = None
            st.success("Interview Completed")


# --------------------------------------------------
# PAGE RENDER
# --------------------------------------------------
if st.session_state.page == "matcher":
    matcher_screen()
else:
    start_screen()
