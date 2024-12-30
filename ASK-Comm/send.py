import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

def text_to_bin(text):
    """将文本转换为二进制字符串"""
    return ''.join(format(ord(c), '08b') for c in text)

def ask_modulate(binary_data, freq=1000, sample_rate=44100, symbol_duration=0.1, amplitude_high=1, amplitude_low=0):
    """对二进制数据进行ASK调制"""
    # 时间轴
    t = np.arange(0, symbol_duration, 1/sample_rate)
    # 调制后的信号
    signal = np.array([])

    for bit in binary_data:
        amplitude = amplitude_high if bit == '1' else amplitude_low
        signal = np.concatenate((signal, amplitude * np.sin(2 * np.pi * freq * t)))

    return signal

def save_audio(signal, filename, sample_rate=44100):
    """将信号保存为音频文件"""
    # 范围标准化为 [-1, 1]
    signal = np.int16(signal / np.max(np.abs(signal)) * 32767)
    wav.write(filename, sample_rate, signal)

def main():
    # 发送的消息
    message = "zhangsan"

    # 将消息转换为二进制
    binary_message = text_to_bin(message)

    # 对二进制消息进行ASK调制
    modulated_signal = ask_modulate(binary_message, freq=1000, symbol_duration=0.1)

    # 保存为WAV文件
    save_audio(modulated_signal, 'output.wav')

    # 可选：绘制调制后的信号波形
    plt.plot(modulated_signal[:500])  # 只绘制前500个采样点
    plt.title("ASK Modulated Signal")
    plt.show()

if __name__ == "__main__":
    main()
