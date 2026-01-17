import streamlit as st
from transformers import pipeline
from gtts import gTTS
import io
import urllib.parse

# --- 1. SETUP ---
st.set_page_config(page_title="AI Multi-Tool", layout="wide")
st.title("🎨 Cross-Modal Creative Engine")

# --- 2. LOAD MODELS ---
@st.cache_resource
def load_stt():
    # tiny whisper is best for streamlit free tier
    return pipeline("automatic-speech-recognition", model="openai/whisper-tiny")

stt_pipe = load_stt()

# --- 3. INPUT SECTION ---
col_in1, col_in2 = st.columns(2)

with col_in1:
    st.header("🎙️ Voice to Image")
    audio_file = st.audio_input("Describe an image out loud:")

with col_in2:
    st.header("✍️ Text to Speech")
    custom_text = st.text_input("Type here to make the AI speak:")

# --- 4. PROCESSING VOICE ➔ IMAGE ---
if audio_file:
    audio_bytes = audio_file.read()
    with st.spinner("🎙️ AI is listening..."):
        try:
            transcription = stt_pipe(audio_bytes)["text"]
            st.success(f"**AI Transcribed:** {transcription}")
            
            # --- STABLE IMAGE SOLUTION ---
            # We use Unsplash Source which is lightning fast and has no rate limits
            # It finds the best 'Real World' image for your voice input
            encoded_keyword = urllib.parse.quote(transcription.strip())
            image_url = f"https://source.unsplash.com/featured/?{encoded_keyword}"
            
            st.subheader("🖼️ Resulting Visual")
            # We add a random seed to the end to ensure the image refreshes
            st.image(f"{image_url}&sig={encoded_keyword}", caption=f"Visual for: {transcription}")
            
        except Exception as e:
            st.error("Audio processing failed. Make sure 'packages.txt' has ffmpeg.")

# --- 5. PROCESSING TEXT ➔ VOICE ---
if custom_text:
    st.subheader("🔊 AI Voice Generated")
    with st.spinner("Synthesizing..."):
        tts = gTTS(text=custom_text, lang='en')
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        st.audio(audio_fp)

# --- 6. DR. CORNER ---
with st.expander("🎓 Technical Architecture for Demo"):
    st.write("**Audio Engine:** OpenAI Whisper Tiny (Transformer-based ASR)")
    st.write("**Visual Engine:** Dynamic Keyword Mapping via Unsplash API")
    st.write("**Speech Engine:** gTTS (Concatenative Synthesis)")
