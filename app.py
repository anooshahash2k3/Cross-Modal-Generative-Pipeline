import streamlit as st
from transformers import pipeline
from gtts import gTTS
import io
import urllib.parse  # This is the secret for fixing the images!

# --- 1. SETUP ---
st.set_page_config(page_title="AI Creative Engine", layout="wide")
st.title("🎨 Cross-Modal Creative Engine")

# --- 2. LOAD MODELS ---
@st.cache_resource
def load_stt():
    return pipeline("automatic-speech-recognition", model="openai/whisper-tiny")

stt_pipe = load_stt()

# --- 3. INPUT SECTION ---
col_in1, col_in2 = st.columns(2)

with col_in1:
    st.header("🎙️ Voice to Image")
    audio_file = st.audio_input("Record a description:")

with col_in2:
    st.header("✍️ Text to Speech")
    custom_text = st.text_input("Type something for the AI to say:")

# --- 4. PROCESSING VOICE ➔ IMAGE ---
if audio_file:
    audio_bytes = audio_file.read()
    with st.spinner("🎙️ Transcribing..."):
        try:
            transcription = stt_pipe(audio_bytes)["text"]
            st.success(f"**AI Heard:** {transcription}")
            
            # --- FIX: PROPER URL ENCODING ---
            # This converts "mountain house" into "mountain%20house"
            encoded_prompt = urllib.parse.quote(transcription.strip())
            
            # Using the latest Flux model via Pollinations for high accuracy
            image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=1024&model=flux&nologo=true"
            
            st.subheader("🖼️ Generated Art")
            st.image(image_url, caption=f"Visual result for: {transcription}")
        except Exception as e:
            st.error(f"Error: {e}")

# --- 5. PROCESSING TEXT ➔ VOICE ---
if custom_text:
    st.subheader("🔊 AI Voice Response")
    with st.spinner("Synthesizing..."):
        tts = gTTS(text=custom_text, lang='en')
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        st.audio(audio_fp)
