from faster_whisper import WhisperModel
import os

model_size = "small"


#CPU with INT8
model = WhisperModel(model_size, device="cpu", compute_type="int8")



def audio_to_txt(output_filename):
    """
    Transcribes an audio file to text using Faster Whisper and saves it to a .txt file.
    """

    try:
        # Transcribe audio
        print(f"Transcribing audio from: {output_filename}")
        segments, info = model.transcribe(output_filename, beam_size=5)

        # Generate the text file path
        text_filename = output_filename.rsplit('.', 1)[0] + ".txt"
        print(f"Saving transcription to: {text_filename}")

        # Write transcription to a file
        with open(text_filename, "w", encoding="utf-8") as txt_file:
            print(f"Writing transcription to {text_filename}...")  # Debug
            # Write detected language and probability
            txt_file.write(
                "Detected language: '%s' with probability: %f\n\n"
                % (info.language, info.language_probability)
            )
            print(
                "Detected language '%s' with probability %f"
                % (info.language, info.language_probability)
            )

            # Write each segment
            for segment in segments:
                line = "[%.2fs -> %.2fs] %s\n" % (segment.start, segment.end, segment.text)
                txt_file.write(line)
                print(line, end="")  # Print for real-time feedback

        print(f"Transcription successfully saved to: {text_filename}")

    except Exception as e:
        print(f"Error during transcription: {e}")
