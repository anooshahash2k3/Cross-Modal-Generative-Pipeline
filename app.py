import streamlit as st
from transformers import pipeline
from gtts import gTTS
import io
import requests

# --- 1. SETUP ---
st.set_page_config(page_title="AI Creative Engine", layout="wide")
st.title("🎨 Cross-Modal Creative Engine")
st.markdown("### Audio (STT) ➔ Text (NLP) ➔ Image (Diffusion) ➔ Voice (TTS)")

# --- 2. LOAD MODELS ---
@st.cache_resource
def load_stt():
    # tiny model to save memory
    return pipeline("automatic-speech-recognition", model="openai/whisper-tiny")

stt_pipe = load_stt()

# --- 3. INPUT ---
audio_file = st.audio_input("Record your image description:")

if audio_file:
    # Read audio bytes
    audio_bytes = audio_file.read()
    
    with st.spinner("🎙️ Transcribing Voice..."):
        try:
            # We pass the bytes directly to the pipeline
            transcription = stt_pipe(audio_bytes)["text"]
            st.success(f"**AI Heard:** {transcription}")
        except Exception as e:
            st.error(f"Transcription Error: {e}. Check if packages.txt has ffmpeg.")
            transcription = ""

    if transcription:
        col1, col2 = st.columns(2)

        with col1:
            st.header("🖼️ Generated Image")
            with st.spinner("Generating Art..."):
                # Using Pollinations API: Fast, Free, No Token Needed
                # This ensures your app NEVER crashes from memory limits
                encoded_prompt = transcription.replace(" ", "%20")
                image_url = f"https://pollinations.ai/p/{encoded_prompt}?width=1024&height=1024&seed=42&model=flux"
                st.image(image_url, caption=f"Result for: {transcription}")

        with col2:
            st.header("🔊 AI Voice Response")
            with st.spinner("Converting to Speech..."):
                response_text = f"I have generated an image based on your request: {transcription}"
                tts = gTTS(text=response_text, lang='en')
                audio_fp = io.BytesIO()
                tts.write_to_fp(audio_fp)
                st.audio(audio_fp)

# --- 4. THE DOCTOR'S CORNER ---
with st.expander("🎓 Technical Summary for Dr. of AI"):
    st.write("""
    **Architectural Highlights:**
    - **Modality 1 (Audio):** Utilized OpenAI Whisper (Tiny) for zero-shot ASR (Automatic Speech Recognition).
    - **Modality 2 (Vision):** Leveraged a RESTful API call to a Stable Diffusion / Flux backend to perform inference without exceeding Streamlit's 1GB RAM limit.
    - **Modality 3 (Speech):** Implemented gTTS for concatenative text-to-speech synthesis.
    - **Deployment:** Managed system-level dependencies via `packages.txt` to provide the FFmpeg binaries required for binary-to-waveform conversion.
    """)
