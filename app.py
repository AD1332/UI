# app2.py
# all 3 models without css

from flask import Flask, render_template, request, jsonify, send_from_directory, send_file
import subprocess
import os
import pyaudio
import wave
import librosa
import numpy as np
from sys import byteorder
from array import array
from struct import pack
from tensorflow.keras.models import load_model 
import time

app = Flask(__name__)

# Load the FNN model here
fnn_model = load_model("results/fnnmodel.h5")  # Replace with the correct path

# Load the CNN model heres
cnn_model = load_model("results/cnnmodel.keras")  # Replace with the correct path

# Load the LSTM model here
lstm_model = load_model("results/lstm_model.h5")
    
THRESHOLD = 500
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
RATE = 16000
SILENCE = 30

def is_silent(snd_data):
    return max(snd_data) < THRESHOLD

def normalize(snd_data):
    MAXIMUM = 16384
    times = float(MAXIMUM)/max(abs(i) for i in snd_data)

    r = array('h')
    for i in snd_data:
        r.append(int(i*times))
    return r

def trim(snd_data):
    def _trim(snd_data):
        snd_started = False
        r = array('h')

        for i in snd_data:
            if not snd_started and abs(i) > THRESHOLD:
                snd_started = True
                r.append(i)
            elif snd_started:
                r.append(i)
        return r

    snd_data = _trim(snd_data)
    snd_data.reverse()
    snd_data = _trim(snd_data)
    snd_data.reverse()
    return snd_data

def add_silence(snd_data, seconds):
    r = array('h', [0 for _ in range(int(seconds*RATE))])
    r.extend(snd_data)
    r.extend([0 for _ in range(int(seconds*RATE))])
    return r

def record():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=1, rate=RATE,
                    input=True, output=True,
                    frames_per_buffer=CHUNK_SIZE)

    num_silent = 0
    snd_started = False
    r = array('h')

    while True:
        snd_data = array('h', stream.read(CHUNK_SIZE))
        if byteorder == 'big':
            snd_data.byteswap()
        r.extend(snd_data)

        silent = is_silent(snd_data)

        if silent and snd_started:
            num_silent += 1
        elif not silent and not snd_started:
            snd_started = True

        if snd_started and num_silent > SILENCE:
            break

    sample_width = p.get_sample_size(FORMAT)
    stream.stop_stream()
    stream.close()
    p.terminate()

    r = normalize(r)
    r = trim(r)
    r = add_silence(r, 0.5)
    return sample_width, r

def record_to_file(path):
    sample_width, data = record()
    data = pack('<' + ('h'*len(data)), *data)

    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()

def extract_feature(file_name, **kwargs):
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")
    X, sample_rate = librosa.core.load(file_name)
    if chroma or contrast:
        stft = np.abs(librosa.stft(X))
    result = np.array([])
    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))
    if chroma:
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        result = np.hstack((result, chroma))
    if mel:
        mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
        result = np.hstack((result, mel))
    if contrast:
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
        result = np.hstack((result, contrast))
    if tonnetz:
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
        result = np.hstack((result, tonnetz))
    return result

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/images/<path:filename>')
def download_file(filename):
    return send_from_directory('images', filename)

@app.route('/audioPlayer')
def playback():
    try:
        return send_file("output.wav", as_attachment=True, mimetype='audio/wav')
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/start_recording', methods=['POST'])
def start_recording():
    try:
        file_path = "output.wav"
        record_to_file(file_path)

        # Extract features and reshape
        features = extract_feature(file_path, mel=True).reshape(1, -1)

        # Predict using the FNN model
        fnn_male_prob = fnn_model.predict(features)[0][0]
        fnn_female_prob = 1 - fnn_male_prob
        fnn_gender = "male" if fnn_male_prob > fnn_female_prob else "female"

        # Predict using the CNN model
        cnn_male_prob = cnn_model.predict(np.expand_dims(features, axis=2))[0][0]
        cnn_female_prob = 1 - cnn_male_prob
        cnn_gender = "male" if cnn_male_prob > cnn_female_prob else "female"

        # Predict using the LSTM model
        lstm_male_prob = lstm_model.predict(features.reshape(1, 1, -1))[0][0]
        lstm_female_prob = 1 - lstm_male_prob
        lstm_gender = "male" if lstm_male_prob > lstm_female_prob else "female"

        # Return the results
        result = {
            "fnn_gender": fnn_gender,
            "fnn_male_prob": fnn_male_prob * 100,
            "fnn_female_prob": fnn_female_prob * 100,
            "cnn_gender": cnn_gender,
            "cnn_male_prob": cnn_male_prob * 100,
            "cnn_female_prob": cnn_female_prob * 100,
            "lstm_gender": lstm_gender,
            "lstm_male_prob": lstm_male_prob * 100,
            "lstm_female_prob": lstm_female_prob * 100
        }

        return jsonify({'success': True, 'result': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)
