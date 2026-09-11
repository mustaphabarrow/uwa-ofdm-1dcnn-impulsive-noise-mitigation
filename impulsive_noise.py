import numpy as np


class ImpulsiveNoiseGenerator:
    def __init__(self, n_subcarriers=64, cp_length=16):
        self.n_subcarriers = n_subcarriers
        self.cp_length = cp_length

    def middleton_class_a(self, n_samples, A=0.1, Gamma=0.1):
        M = 10
        pmf = np.exp(-A) * (A ** np.arange(M)) / np.arange(1, M + 1)
        pmf = pmf / np.sum(pmf)

        sigma_g2 = 1.0
        sigma_i2 = Gamma * sigma_g2
        noise = np.zeros(n_samples, dtype=complex)

        for m in range(M):
            variance = m * sigma_i2 + sigma_g2
            n_gaussian = np.sqrt(variance / 2) * (
                np.random.randn(n_samples) + 1j * np.random.randn(n_samples)
            )
            count = int(pmf[m] * n_samples)
            indices = np.random.choice(n_samples, count, replace=False)
            noise[indices] += n_gaussian[indices] * np.sqrt(pmf[m])

        return noise

    def bernoulli_gaussian(self, n_samples, p=0.1, sigma_i2=10.0):
        mask = np.random.random(n_samples) < p
        noise = np.zeros(n_samples, dtype=complex)
        noise[mask] = np.sqrt(sigma_i2 / 2) * (
            np.random.randn(np.sum(mask)) + 1j * np.random.randn(np.sum(mask))
        )
        return noise

    def alpha_stable(self, n_samples, alpha=1.5, scale=0.1):
        u = np.random.uniform(-np.pi / 2, np.pi / 2, n_samples)
        w = np.random.exponential(1, n_samples)
        theta = (1 - alpha) * u
        x = (
            np.sin(alpha * theta)
            / (np.cos(theta) ** (1 / alpha))
            * (np.cos(theta - alpha * theta) / w) ** ((1 - alpha) / alpha)
        )
        return scale * (x + 1j * np.random.randn(n_samples) * scale)

    def add_impulsive_noise(self, signal, snr_db=20, noise_type="middleton_class_a", **kwargs):
        signal_power = np.mean(np.abs(signal) ** 2)
        snr_linear = 10 ** (snr_db / 10)
        noise_power_desired = signal_power / snr_linear

        if signal.ndim > 1:
            avg_noise = np.zeros_like(signal)
            for i in range(signal.shape[0]):
                n_samples = signal.shape[1]
                if noise_type == "middleton_class_a":
                    noise = self.middleton_class_a(n_samples, **kwargs)
                elif noise_type == "bernoulli_gaussian":
                    noise = self.bernoulli_gaussian(n_samples, **kwargs)
                elif noise_type == "alpha_stable":
                    noise = self.alpha_stable(n_samples, **kwargs)
                else:
                    raise ValueError(f"Unknown noise type: {noise_type}")

                current_noise_power = np.mean(np.abs(noise) ** 2)
                if current_noise_power > 0:
                    noise = noise * np.sqrt(noise_power_desired / current_noise_power)
                avg_noise[i] = noise
            return signal + avg_noise
        else:
            n_samples = len(signal)
            if noise_type == "middleton_class_a":
                noise = self.middleton_class_a(n_samples, **kwargs)
            elif noise_type == "bernoulli_gaussian":
                noise = self.bernoulli_gaussian(n_samples, **kwargs)
            elif noise_type == "alpha_stable":
                noise = self.alpha_stable(n_samples, **kwargs)
            else:
                raise ValueError(f"Unknown noise type: {noise_type}")

            current_noise_power = np.mean(np.abs(noise) ** 2)
            if current_noise_power > 0:
                noise = noise * np.sqrt(noise_power_desired / current_noise_power)
            return signal + noise

    def add_impulsive_noise_block(self, signal, snr_db=20, impulsive_ratio=0.1, impulsive_power_ratio=100):
        signal_power = np.mean(np.abs(signal) ** 2)
        snr_linear = 10 ** (snr_db / 10)

        awgn_power = signal_power / snr_linear
        awgn = np.sqrt(awgn_power / 2) * (
            np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape)
        )

        n_impulsive = int(impulsive_ratio * signal.shape[1])
        impulsive = np.zeros_like(signal)
        for i in range(signal.shape[0]):
            indices = np.random.choice(signal.shape[1], n_impulsive, replace=False)
            impulsive_amp = np.sqrt(awgn_power * impulsive_power_ratio) * (
                np.random.randn(n_impulsive) + 1j * np.random.randn(n_impulsive)
            )
            impulsive[i, indices] = impulsive_amp

        return signal + awgn + impulsive

    def estimate_impulsive_power(self, signal, threshold_factor=3):
        magnitude = np.abs(signal)
        median_mag = np.median(magnitude)
        mad = np.median(np.abs(magnitude - median_mag))
        threshold = median_mag + threshold_factor * mad

        impulsive_mask = magnitude > threshold
        impulsive_power = np.mean(np.abs(signal[impulsive_mask]) ** 2) if np.any(impulsive_mask) else 0
        total_power = np.mean(np.abs(signal) ** 2)

        return impulsive_power / total_power if total_power > 0 else 0

    def detect_impulsive_samples(self, signal, threshold_factor=3):
        magnitude = np.abs(signal)
        median_mag = np.median(magnitude)
        mad = np.median(np.abs(magnitude - median_mag))
        threshold = median_mag + threshold_factor * mad
        return (magnitude > threshold).astype(float)
