import streamlit as st
from transformers import pipeline
from gtts import gTTS
import io
import urllib.parse
import time

# --- 1. SETUP ---
st.set_page_config(page_title="AI Creative Engine", layout="wide")
st.title("Cross-Modal Creative Engine")
st.markdown("---")

# --- 2. LOAD MODELS ---
@st.cache_resource
def load_stt():
    # Whisper Tiny is the best for fast transcription on Streamlit
    return pipeline("automatic-speech-recognition", model="openai/whisper-tiny")

stt_pipe = load_stt()

# --- 3. THREE-COLUMN INPUT SECTION ---
col1, col2, col3 = st.columns(3)

with col1:
    st.header("🎙️ Voice to Image")
    audio_image = st.audio_input("Record to generate art:", key="audio_image")

with col2:
    st.header("Speech to Text")
    audio_stt = st.audio_input("Record to transcribe:", key="audio_stt")

with col3:
    st.header("Text to Speech")
    custom_text = st.text_input("Type to hear AI voice:")

# --- 4. PROCESSING LOGIC ---

# Logic for Column 1: Voice to Image
if audio_image:
    with st.spinner("Creating Art..."):
        try:
            image_text = stt_pipe(audio_image.read())["text"]
            st.success(f"AI Prompt: {image_text}")
            encoded_prompt = urllib.parse.quote(image_text.strip())
            # Added a seed to prevent rate limits
            seed = int(time.time())
            image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=1024&model=flux&seed={seed}&nologo=true"
            st.image(image_url, caption="Generated Art")
        except Exception as e:
            st.error(f"Image Error: {e}")

# Logic for Column 2: Speech to Text (Transcriber)
if audio_stt:
    with st.spinner("Transcribing..."):
        try:
            transcription = stt_pipe(audio_stt.read())["text"]
            st.subheader("Transcribed Text:")
            st.code(transcription, language="text") # Displays text in a copyable box
            st.download_button("Download Transcript", transcription, file_name="transcript.txt")
        except Exception as e:
            st.error(f"STT Error: {e}")

# Logic for Column 3: Text to Speech
if custom_text:
    with st.spinner("Synthesizing..."):
        tts = gTTS(text=custom_text, lang='en')
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        st.audio(audio_fp)

# --- 5. DR. CORNER ---
with st.expander("Technical Summary"):
    st.write("- **Automatic Speech Recognition (ASR):** Using OpenAI Whisper Tiny.")
    st.write("- **Text-to-Image:** Latent Diffusion via Flux API.")
    st.write("- **Speech Synthesis:** Concatenative TTS via gTTS.")
    st.write("- **Concurrency:** Independent processing loops for multimodal outputs.")
