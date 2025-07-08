import numpy as np
from scipy.signal import find_peaks
import pywt
import matplotlib.pyplot as plt

class SpikeDetector:
    def __init__(self, signal, fs):
        """
        Parameters:
        - signal: 1D numpy array (neural signal)
        - fs: sampling rate in Hz
        """
        self.signal = signal
        self.fs = fs
        self.spike_indices = None
        self.threshold = None
        self.method_output = None

    def _mad(self, x):
        return np.median(np.abs(x - np.median(x))) / 0.6745

    # ----- Amplitude Thresholding by SNR -----
    def detect_by_snr(self, snr_threshold=5):
        noise_std = self._mad(self.signal)
        self.threshold = -snr_threshold * noise_std
        self.spike_indices, _ = find_peaks(-self.signal, height=-self.threshold)
        self.method_output = self.signal
        return self.spike_indices

    # ----- Nonlinear Energy Operator (NEO) -----
    def _neo(self, signal):
        neo = np.zeros_like(signal)
        neo[1:-1] = signal[1:-1] ** 2 - signal[2:] * signal[:-2]
        return neo

    def detect_by_neo(self, threshold_factor=5):
        neo = self._neo(self.signal)
        noise_std = self._mad(neo)
        self.threshold = threshold_factor * noise_std
        self.spike_indices, _ = find_peaks(neo, height=self.threshold)
        self.method_output = neo
        return self.spike_indices

    # ----- Wavelet Transform Product (WTP) -----
    def _wtp(self, levels=[1, 2, 3], wavelet='db4'):
        coeffs = pywt.wavedec(self.signal, wavelet, level=max(levels))
        details = [coeffs[l] for l in levels]
        upsampled = [pywt.upcoef('d', d, wavelet, level=l, take=len(self.signal)) for d, l in zip(details, levels)]
        return np.prod(upsampled, axis=0)

    def detect_by_wtp(self, threshold_factor=5, levels=[1, 2, 3], wavelet='db4'):
        wtp = self._wtp(levels=levels, wavelet=wavelet)
        noise_std = self._mad(wtp)
        self.threshold = threshold_factor * noise_std
        self.spike_indices, _ = find_peaks(wtp, height=self.threshold)
        self.method_output = wtp
        return self.spike_indices

    # ----- 긴 신호 슬라이딩 윈도우 처리 -----
    def detect_long_signal(self, method='neo', chunk_sec=5, overlap=0, **kwargs):
        """
        긴 신호를 chunk 단위로 나누어 spike 검출 수행

        Parameters:
        - method: 'snr', 'neo', 'wtp'
        - chunk_sec: 청크 단위 길이 (초)
        - overlap: 청크 중첩 길이 (초)
        - kwargs: 각 검출 메서드로 전달할 추가 파라미터

        Returns:
        - np.array: 전체 신호 기준 spike 인덱스 리스트
        """
        chunk_size = int(chunk_sec * self.fs)
        step = chunk_size - int(overlap * self.fs)
        all_spikes = []

        for start in range(0, len(self.signal), step):
            end = min(start + chunk_size, len(self.signal))
            chunk = self.signal[start:end]
            sub_detector = SpikeDetector(chunk, self.fs)

            if method == 'snr':
                spikes = sub_detector.detect_by_snr(**kwargs)
            elif method == 'neo':
                spikes = sub_detector.detect_by_neo(**kwargs)
            elif method == 'wtp':
                spikes = sub_detector.detect_by_wtp(**kwargs)
            else:
                raise ValueError("method must be one of: 'snr', 'neo', 'wtp'")

            global_spikes = [start + idx for idx in spikes]
            all_spikes.extend(global_spikes)

        self.spike_indices = np.array(all_spikes)
        return self.spike_indices

    # ----- Spike waveform 추출 (음의 peak 정렬 포함) -----
    def extract_spike_waveforms(self, window_ms=5, align_peak=True, peak_search_ms=1.0):
        """
        spike index를 기준으로 음의 피크에 맞춰 spike waveform 추출

        Parameters:
        - window_ms: 파형 윈도우 크기 (ms, default 5ms)
        - align_peak: True면 local 음의 peak 기준 정렬 수행
        - peak_search_ms: spike index 주변 peak 검색 폭 (ms)

        Returns:
        - np.array: (스파이크 개수 x 윈도우 샘플 수) 2D 배열
        """
        if self.spike_indices is None:
            raise ValueError("No spikes detected. Run detection first.")

        half_window = int((window_ms / 1000) * self.fs // 2)
        peak_search_half = int((peak_search_ms / 1000) * self.fs // 2)

        waveforms = []
        for idx in self.spike_indices:
            search_start = max(0, idx - peak_search_half)
            search_end = min(len(self.signal), idx + peak_search_half)

            if align_peak:
                local_idx = np.argmin(self.signal[search_start:search_end])
                true_idx = search_start + local_idx
            else:
                true_idx = idx

            start = true_idx - half_window
            end = true_idx + half_window

            if start < 0 or end > len(self.signal):
                continue

            snippet = self.signal[start:end]
            waveforms.append(snippet)

        return np.array(waveforms)

    # ----- Spike waveform 시각화 -----
    def plot_spike_waveforms(self, window_ms=5, max_waveforms=100):
        """
        검출된 spike들의 waveform을 플롯 (최대 max_waveforms 개)

        Parameters:
        - window_ms: 파형 윈도우 크기 (ms)
        - max_waveforms: 최대 플롯 개수 (가독성 위해 제한)
        """
        waveforms = self.extract_spike_waveforms(window_ms=window_ms, align_peak=True)
        n = min(len(waveforms), max_waveforms)

        t = np.linspace(-window_ms/2, window_ms/2, waveforms.shape[1])

        plt.figure(figsize=(10, 5))
        for i in range(n):
            plt.plot(t, waveforms[i], alpha=0.5, color='blue')
        plt.title(f'{n} Aligned Spike Waveforms (±{window_ms/2} ms)')
        plt.xlabel('Time (ms)')
        plt.ylabel('Amplitude (uV)')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # ----- 결과 접근용 메서드 -----
    def get_spikes(self):
        return self.spike_indices

    def get_threshold(self):
        return self.threshold

    def get_method_output(self):
        return self.method_output
