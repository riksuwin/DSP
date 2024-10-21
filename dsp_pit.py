from flask import Flask, render_template, request, jsonify
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
import io
import base64
import sounddevice as sd
import logging

# Use the Agg backend for Matplotlib
plt.switch_backend('Agg')

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

dtmf_freqs = {
    '1': (697, 1209), '2': (697, 1336), '3': (697, 1477),
    '4': (770, 1209), '5': (770, 1336), '6': (770, 1477),
    '7': (852, 1209), '8': (852, 1336), '9': (852, 1477),
    '*': (941, 1209), '0': (941, 1336), '#': (941, 1477)
}

def generate_dtmf_tone(key, duration=0.5, fs=8000):
    if key not in dtmf_freqs:
        raise ValueError(f"Invalid key: {key}")
    
    f1, f2 = dtmf_freqs[key]
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    tone = np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t)
    return tone

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/play_tone', methods=['POST'])
def play_tone():
    key = request.json.get('key')
    if key not in dtmf_freqs:
        return jsonify({'error': 'Invalid key'})
    
    tone = generate_dtmf_tone(key)
    sd.play(tone, samplerate=8000)
    sd.wait()  # Wait until the sound has finished playing
    return jsonify({'status': 'success'})

@app.route('/time_domain_analysis', methods=['POST'])
def time_domain_analysis():
    key = request.json.get('key')
    if key not in dtmf_freqs:
        return jsonify({'error': 'Invalid key'})
    
    tone = generate_dtmf_tone(key)
    t = np.linspace(0, len(tone)/8000, len(tone))
    
    plt.figure()
    plt.plot(t[:100], tone[:100])  # Plot the first 100 samples
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title('Time-Domain Analysis')
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return jsonify({'plot_url': plot_url})

@app.route('/frequency_domain_analysis', methods=['POST'])
def frequency_domain_analysis():
    key = request.json.get('key')
    if key not in dtmf_freqs:
        return jsonify({'error': 'Invalid key'})
    
    tone = generate_dtmf_tone(key)
    N = len(tone)
    T = 1.0 / 8000.0
    yf = fft(tone)
    xf = np.fft.fftfreq(N, T)[:N//2]
    
    plt.figure()
    plt.plot(xf, 2.0/N * np.abs(yf[:N//2]))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('Frequency Spectrum')
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return jsonify({'plot_url': plot_url})

if __name__ == '__main__':
    app.run(debug=True)