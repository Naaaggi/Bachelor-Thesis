import pyaudio
import os
import threading
from pydub import AudioSegment


def record_audio(output_filename):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK,
                    input_device_index=0)

    print("Recording...")

    frames = []
    is_recording = True

    def stop_recording():
        nonlocal is_recording
        input("Press 'Enter' to stop: ")
        is_recording = False

    stop_thread = threading.Thread(target=stop_recording)
    stop_thread.start()

    while is_recording:
        data = stream.read(CHUNK)
        frames.append(data)

    print("Recording stopped.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save the recorded audio to an MP3 file
    audio_data = b''.join(frames)
    audio_segment = AudioSegment(
        audio_data,
        sample_width=2,
        frame_rate=RATE,
        channels=CHANNELS
    )
    audio_segment.export(output_filename, format="mp3")


if __name__ == "__main__":
    output_filename = "output.mp3"
    if os.path.exists(output_filename):
        print(f"File '{output_filename}' already exists. Deleting it.")
        os.remove(output_filename)

    record_audio(output_filename)
    print(f"Recording saved as '{output_filename}'.")
