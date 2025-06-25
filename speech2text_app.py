import os
import torch
from transformers import pipeline
import gradio as gr

# Ensure ffmpeg is accessible (if needed)
#os.environ["PATH"] += os.pathsep + "C:/ffmpeg/bin"  # Update this path if needed

# Initialize the speech recognition pipeline once
pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-tiny.en",
    chunk_length_s=30,
)

# Function to transcribe audio using the OpenAI Whisper model
def transcript_audio(audio_file):
    result = pipe(audio_file, batch_size=8)["text"]
    return result

# Set up Gradio interface
audio_input = gr.Audio(sources="upload", type="filepath")
output_text = gr.Textbox()

# Create the Gradio interface
iface = gr.Interface(
    fn=transcript_audio,
    inputs=audio_input,
    outputs=output_text,
    title="🎙️ Audio Transcription App",
    description="Upload an audio file (e.g., MP3 or WAV) to generate a transcription using OpenAI's Whisper model.",
)

# Launch the app — access via http://127.0.0.1:7860
iface.launch(server_name="127.0.0.1", server_port=7860)
