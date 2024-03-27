from gtts import gTTS
import os


def text_to_speech(text, lang='en', filename='pcvoice.mp3'):
    tts = gTTS(text=text, lang=lang)
    tts.save(filename)

    if os.name == 'posix':  # For macOS and Linux
        os.system(f"afplay {filename}")
    elif os.name == 'nt':  # For Windows
        os.system(f"start {filename}")
    else:
        print("Unsupported OS. Cannot play the audio.")


# Example usage:
if __name__ == "__main__":
    text_to_speech("This is the PC speaking")
