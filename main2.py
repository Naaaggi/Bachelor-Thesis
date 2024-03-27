import time
import tensorflow as tf
import numpy as np
import tensorflow_io as tfio
from record_audio import record_audio
import keras
from openai_helper import ask_gpt
from text_to_speech_helper import text_to_speech


# An integer scalar Tensor. The window length in samples.
frame_length = 256
# An integer scalar Tensor. The number of samples to advance the window per iteration.
frame_step = 160
# An integer scalar Tensor. The FFT length.
fft_length = 384
# A utility function to decode the output of the model


def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(
        pred, input_length=input_len, greedy=True)[0][0]
    # Iterate over the results and get back the text
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text


# The set of characters accepted in the transcription
characters = [x for x in "abcdefghijklmnopqrstuvwxyz'?! "]
# Mappping from character to integer
char_to_num = keras.layers.StringLookup(vocabulary=characters, oov_token="")
# Mappping from integer to character
num_to_char = keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)


def CTCLoss(y_true, y_pred):
    # Compute the CTC loss
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = keras.backend.ctc_batch_cost(
        y_true, y_pred, input_length, label_length)

    return loss


def main():
    # Step 1: Record audio
    print(time.time())
    record_audio("output.mp3")
    print(time.time())
    # Step 2: Load the Keras model with custom loss
    loaded_model = keras.models.load_model(
        "final2.model", custom_objects={"CTCLoss": CTCLoss})
    # Note: You have loaded the model twice in the provided code, so I removed the second load
    print(time.time())
    # Step 3: Load the recorded audio file and preprocess it
    audio_file = "output.mp3"
    file = tf.io.read_file(audio_file)
    audio = tfio.audio.decode_mp3(file)
    print(time.time())

    # Convert stereo audio to mono by taking the average of the channels
    audio = tf.reduce_mean(audio, axis=-1)

    stfts = tf.signal.stft(audio, frame_length=frame_length,
                           frame_step=frame_step, fft_length=fft_length)
    spectrogram = tf.abs(stfts)
    spectrogram = tf.math.pow(spectrogram, 0.5)
    mean = tf.math.reduce_mean(spectrogram, axis=1, keepdims=True)
    stddev = tf.math.reduce_std(spectrogram, axis=1, keepdims=True)
    spectrogram = tf.divide(tf.subtract(spectrogram, mean), stddev + 1e-10)
    input_audio = tf.expand_dims(spectrogram, axis=0)
    print(time.time())

    # Step 4: Make a prediction using the loaded model
    prediction = loaded_model.predict(input_audio)
    print(time.time())

    # Step 5: Decode the prediction
    decoded_prediction = decode_batch_predictions(prediction)
    print(time.time())

    # Step 6: Use GPT to generate a response
    prompt = decoded_prediction[0]
    print("Me: ", prompt)
    response = ask_gpt(prompt)
    print("Bot: ", response)
    text_to_speech(response)
    print(time.time())


if __name__ == "__main__":
    main()
