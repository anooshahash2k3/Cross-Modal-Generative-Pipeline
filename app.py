import streamlit as st
from transformers import pipeline
from gtts import gTTS
import io
import requests

# --- 1. SETUP ---
st.set_page_config(page_title="AI Creative Engine", layout="wide")
st.title("🎨 Cross-Modal Creative Engine")
st.markdown("### Voice ➔ Text ➔ Image & Text ➔ Voice")

# --- 2. LOAD MODELS ---
@st.cache_resource
def load_stt():
    return pipeline("automatic-speech-recognition", model="openai/whisper-tiny")

stt_pipe = load_stt()

# --- 3. INPUT SECTION ---
col_in1, col_in2 = st.columns(2)

with col_in1:
    st.header("🎙️ Voice to Image")
    audio_file = st.audio_input("Record a description for an image:")

with col_in2:
    st.header("✍️ Text to Speech")
    custom_text = st.text_input("Type something for the AI to say out loud:", placeholder="e.g. Hello Dr. this is my AI project.")

# --- 4. PROCESSING VOICE ➔ IMAGE ---
if audio_file:
    audio_bytes = audio_file.read()
    with st.spinner("🎙️ Transcribing..."):
        try:
            transcription = stt_pipe(audio_bytes)["text"]
            st.success(f"**AI Heard:** {transcription}")
            
            # GENERATE IMAGE
            st.subheader("🖼️ Generated Art")
            # Using a cleaner URL format for Pollinations
            prompt_cleaned = transcription.strip().replace(" ", "-")
            image_url = f"https://image.pollinations.ai/prompt/{prompt_cleaned}?width=1024&height=1024&nologo=true"
            st.image(image_url, caption=f"Visual result for: {transcription}")
        except Exception as e:
            st.error(f"Error: {e}. Ensure packages.txt has ffmpeg.")

# --- 5. PROCESSING TEXT ➔ VOICE ---
if custom_text:
    st.subheader("🔊 AI Voice Response")
    with st.spinner("Synthesizing..."):
        tts = gTTS(text=custom_text, lang='en')
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        st.audio(audio_fp)

# --- 6. DR. CORNER ---
with st.expander("🎓 Technical Summary"):
    st.write("- **Image Gen:** Using RESTful endpoint for Flux/Stable Diffusion.")
    st.write("- **Transcription:** Whisper Tiny Transformer.")
    st.write("- **Synthesis:** gTTS (Google Text-to-Speech) API.")
