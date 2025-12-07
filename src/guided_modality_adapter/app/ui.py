import streamlit as st
import os
import tempfile
from src.guided_modality_adapter.services.transcription import DiarizationPipeline
import datetime

st.set_page_config(page_title="Modality Adapter - Audio", layout="wide")

st.title("🗣️ Guided Modality: Speaker Diarization")

# Sidebar Configuration
with st.sidebar:
    st.header("Configuration")
    hf_token = st.text_input("Hugging Face Token", type="password")
    model_size = st.selectbox("Whisper Model", ["tiny", "base", "small", "medium"], index=1)
    num_speakers = st.number_input("Num Speakers (0=Auto)", min_value=0, value=0)

uploaded_file = st.file_uploader("Upload Audio", type=["wav", "mp3", "m4a"])

if uploaded_file and hf_token:
    if st.button("Process Audio"):
        # Create a temporary file to handle the upload
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name

        try:
            with st.spinner("Initializing Pipeline..."):
                service = DiarizationPipeline(hf_token=hf_token, model_size=model_size)
            
            with st.spinner("Processing Audio..."):
                results = service.process(tmp_path, num_speakers=num_speakers if num_speakers > 0 else None)
            
            st.success("Analysis Complete")
            
            # Display Results
            for line in results:
                time_str = str(datetime.timedelta(seconds=int(line['start'])))
                st.markdown(f"**{line['speaker']}** `[{time_str}]`: {line['text']}")
                st.divider()
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
        finally:
            os.remove(tmp_path)
            
elif not hf_token:
    st.info("Please provide a Hugging Face token to enable Pyannote.")