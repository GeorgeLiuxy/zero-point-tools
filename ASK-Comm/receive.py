import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

def ask_demodulate(signal, freq=1000, sample_rate=44100, symbol_duration=0.1, amplitude_threshold=0.5):
    """解调ASK信号"""
    samples_per_symbol = int(symbol_duration * sample_rate)
    bits = []

    for i in range(0, len(signal), samples_per_symbol):
        symbol = signal[i:i+samples_per_symbol]
        amplitude = np.max(np.abs(symbol))  # 计算符号的最大幅度
        bit = '1' if amplitude > amplitude_threshold else '0'
        bits.append(bit)

    return ''.join(bits)

def bin_to_text(binary_data):
    """将二进制数据转换回文本"""
    text = ''
    for i in range(0, len(binary_data), 8):
        byte = binary_data[i:i+8]
        text += chr(int(byte, 2))
    return text

def record_audio(duration, sample_rate=44100):
    """录制音频信号"""
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float64')
    sd.wait()
    return audio.flatten()

def main():
    # 录制5秒钟的音频信号
    print("Recording audio...")
    recorded_signal = record_audio(5)

    # 可选：绘制录制的音频波形
    plt.plot(recorded_signal[:500])  # 只绘制前500个采样点
    plt.title("Recorded Signal")
    plt.show()

    # 对录制的信号进行解调
    demodulated_bits = ask_demodulate(recorded_signal, freq=1000, symbol_duration=0.1)

    # 解码得到消息
    decoded_message = bin_to_text(demodulated_bits)

    print("Decoded message:", decoded_message)

if __name__ == "__main__":
    main()
