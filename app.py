import streamlit as st
import tempfile
import os
import joblib
from audio_recorder_streamlit import audio_recorder

from model import predict_audio




SAMPLING_RATE = 16000
MODEL_PATH = "model/child_detector_calibrated.joblib"



@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


classifier = load_model()


st.set_page_config(
    page_title="Parental Voice Control",
    page_icon="🎙️",
    layout="centered"
)

st.title("🎙️ Parental Voice Authentication")
st.write(
    "Please speak into the microphone. "
    "Your voice will be analyzed to determine access permission."
)

st.divider()

# Initialize session state to track processed audio
if 'last_audio_hash' not in st.session_state:
    st.session_state.last_audio_hash = None
if 'result' not in st.session_state:
    st.session_state.result = None

# Request microphone input
audio_bytes = audio_recorder(
    text="Click to record",
    recording_color="#e74c3c",
    neutral_color="#aeaeae",
    icon_name="microphone",
    icon_size="1x",
    pause_threshold=2.0,
)

if audio_bytes is not None:
    current_hash = hash(audio_bytes)

    if current_hash != st.session_state.last_audio_hash:
        st.session_state.last_audio_hash = current_hash
        st.session_state.result = None  # reset previous result

        if len(audio_bytes) < 1000:
            st.session_state.result = {'prediction': 'unknown', 'message': 'Audio too short'}
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
            try:
                prediction, message = predict_audio(tmp_path)
                st.session_state.result = {'prediction': prediction, 'message': message}
            except ValueError: 
                 st.session_state.result = {'prediction': 'unknown', 'message': 'Audio too short'}
            finally:
                os.remove(tmp_path)

        # Update hash after processing
        st.session_state.last_audio_hash = current_hash

# Display results if available
if st.session_state.result is not None:
    st.divider()
    
    prediction = st.session_state.result['prediction']
    
    if prediction == "child":
        st.error(
            "🚫 Access Denied\n\n"
            "Child voice detected. Parental access is required.",
            icon="🚫"
        )

    elif prediction == "adult":
        st.success(
            "✅ Access Granted\n\n"
            "Adult voice detected. Welcome!",
            icon="✅"
        )

    else:
        st.warning(
            f"⚠️ Unable to determine voice category.\n\n"
            f"{st.session_state.result.get('message', 'Unknown error')}",
            icon="⚠️"
        )
    