import numpy as np


class UWAChannel:
    def __init__(self, n_subcarriers=64, cp_length=16, bandwidth=12e3, carrier_freq=12e3):
        self.n_subcarriers = n_subcarriers
        self.cp_length = cp_length
        self.bandwidth = bandwidth
        self.carrier_freq = carrier_freq
        self.subcarrier_spacing = bandwidth / n_subcarriers

    def generate_multipath_channel(self, n_taps=6, max_delay_spread=0.01):
        delays = np.sort(np.random.uniform(0, max_delay_spread, n_taps))
        delays_samples = np.round(delays * self.bandwidth).astype(int)
        delays_samples = np.clip(delays_samples, 0, self.cp_length - 1)

        amplitudes = np.exp(-delays / max_delay_spread) * (
            np.random.randn(n_taps) + 1j * np.random.randn(n_taps)
        ) / np.sqrt(2)

        h = np.zeros(self.n_subcarriers, dtype=complex)
        for i, (tap, amp) in enumerate(zip(delays_samples, amplitudes)):
            h[tap] += amp

        h = h / np.sqrt(np.sum(np.abs(h) ** 2))
        return h

    def generate_frequency_response(self, n_taps=6, max_delay_spread=0.01):
        h = self.generate_multipath_channel(n_taps, max_delay_spread)
        H = np.fft.fft(h, self.n_subcarriers)
        return H

    def apply_channel(self, tx_signal, H):
        h_time = np.fft.ifft(H, self.n_subcarriers)
        h_padded = np.zeros((tx_signal.shape[1],), dtype=complex)
        h_padded[: min(len(h_time), self.n_subcarriers)] = h_time[
            : min(len(h_time), self.n_subcarriers)
        ]

        h_padded_cp = np.concatenate(
            [h_padded[-self.cp_length :], h_padded]
        )

        convolved = np.zeros_like(tx_signal, dtype=complex)
        for i in range(tx_signal.shape[0]):
            convolved[i] = np.convolve(tx_signal[i], h_padded_cp, mode="same")[
                : tx_signal.shape[1]
            ]

        return convolved

    def apply_channel_freq(self, freq_symbols, H):
        return freq_symbols * H

    def estimate_channel_lms(self, rx_freq, pilot_tx, pilot_indices, mu=0.01, n_iter=50):
        H_est = np.ones(self.n_subcarriers, dtype=complex)
        H_est[pilot_indices] = 0.1

        for _ in range(n_iter):
            for idx in pilot_indices:
                y = rx_freq[idx]
                x = pilot_tx[idx]
                h = H_est[idx]
                e = y - h * x
                H_est[idx] = h + mu * np.conj(x) * e

        pilot_H = H_est[pilot_indices]
        all_indices = np.arange(self.n_subcarriers)
        non_pilot = np.setdiff1d(all_indices, pilot_indices)
        H_est[non_pilot] = np.interp(
            non_pilot, pilot_indices, np.abs(pilot_H)
        ) + 1j * np.interp(non_pilot, pilot_indices, np.angle(pilot_H))

        return H_est

    def estimate_channel_mmse(self, rx_freq, pilot_tx, pilot_indices, snr_db=20):
        H_est = np.zeros(self.n_subcarriers, dtype=complex)
        snr_linear = 10 ** (snr_db / 10)

        for idx in pilot_indices:
            H_est[idx] = rx_freq[idx] / pilot_tx[idx]

        pilot_H = H_est[pilot_indices]
        all_indices = np.arange(self.n_subcarriers)
        non_pilot = np.setdiff1d(all_indices, pilot_indices)

        H_est[non_pilot] = np.interp(
            non_pilot, pilot_indices, np.abs(pilot_H)
        ) + 1j * np.interp(non_pilot, pilot_indices, np.angle(pilot_H))

        return H_est
