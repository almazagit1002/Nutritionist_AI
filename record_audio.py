import pyaudio
import wave
import threading
import keyboard
from faster_whisper import WhisperModel
from audio_to_txt import audio_to_txt
# Initialize the Whisper model
model_size = "small"
model = WhisperModel(model_size, device="cpu", compute_type="int8")
 

# def audio_to_txt(output_filename):
#     """
#     Transcribes an audio file to text using Faster Whisper.
#     """
#     segments, info = model.transcribe(output_filename, beam_size=5)

#     print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

#     for segment in segments:
#         print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))


def record_audio(output_filename, sample_rate=44100, channels=1, chunk_size=1024):
    """
    Records audio and saves it to a file. Recording stops when 'q' is pressed.
    """
    audio = pyaudio.PyAudio()

    stream = audio.open(
        format=pyaudio.paInt16,
        channels=channels,
        rate=sample_rate,
        input=True,
        frames_per_buffer=chunk_size,
    )

    print("Recording... Press 'q' to stop.")

    frames = []
    recording = True

    def stop_recording():
        nonlocal recording
        while recording:
            if keyboard.is_pressed("q"):
                print("\nStopping recording...")
                recording = False
                break

    stop_thread = threading.Thread(target=stop_recording)
    stop_thread.start()

    while recording:
        data = stream.read(chunk_size)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    audio.terminate()

    with wave.open(output_filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sample_rate)
        wf.writeframes(b"".join(frames))

    print(f"Audio saved to {output_filename}")


if __name__ == "__main__":
    output_file = "recording.wav"

    # Step 1: Record audio
    record_audio(output_file)

    # Step 2: Transcribe audio to text
    print("\nStarting transcription...")
    audio_to_txt(output_file)
