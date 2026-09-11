import numpy as np


class OFDMSystem:
    def __init__(self, n_subcarriers=64, cp_length=16, modulation="QPSK"):
        self.n_subcarriers = n_subcarriers
        self.cp_length = cp_length
        self.modulation = modulation
        self.bits_per_symbol = 2 if modulation == "QPSK" else 4

        n_null = max(2, n_subcarriers // 16)
        self.null_subcarriers = np.array(
            list(range(n_null // 2)) + list(range(n_subcarriers - n_null + n_null // 2, n_subcarriers)),
            dtype=int,
        )
        self.data_subcarriers = np.array(
            [i for i in range(n_subcarriers) if i not in self.null_subcarriers],
            dtype=int,
        )
        self.null_mask = np.zeros(n_subcarriers, dtype=bool)
        self.null_mask[self.null_subcarriers] = True

    def generate_random_bits(self, n_symbols):
        return np.random.randint(
            0, 2, size=(n_symbols, self.n_subcarriers * self.bits_per_symbol)
        )

    def modulate(self, bits):
        n_symbols = bits.shape[0]
        symbols_per_sub = self.bits_per_symbol
        bit_groups = bits.reshape(n_symbols, self.n_subcarriers, symbols_per_sub)

        if self.modulation == "QPSK":
            constellation = np.array(
                [1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]
            ) / np.sqrt(2)
            indices = bit_groups[:, :, 0] * 2 + bit_groups[:, :, 1]
            symbols = constellation[indices]
            symbols[:, self.null_subcarriers] = 0
            return symbols
        elif self.modulation == "16QAM":
            constellation = np.array([
                -3-3j, -3-1j, -3+1j, -3+3j,
                -1-3j, -1-1j, -1+1j, -1+3j,
                 1-3j,  1-1j,  1+1j,  1+3j,
                 3-3j,  3-1j,  3+1j,  3+3j,
            ]) / np.sqrt(10)
            indices = (bit_groups[:, :, 0] * 8 + bit_groups[:, :, 1] * 4 +
                       bit_groups[:, :, 2] * 2 + bit_groups[:, :, 3])
            symbols = constellation[indices]
            symbols[:, self.null_subcarriers] = 0
            return symbols

    def demodulate(self, received_symbols):
        if self.modulation == "QPSK":
            bits = np.zeros(
                (received_symbols.shape[0], self.n_subcarriers, 2), dtype=int
            )
            bits[:, :, 0] = np.real(received_symbols) < 0
            bits[:, :, 1] = np.imag(received_symbols) < 0
            return bits.reshape(received_symbols.shape[0], -1)
        elif self.modulation == "16QAM":
            bits = np.zeros(
                (received_symbols.shape[0], self.n_subcarriers, 4), dtype=int
            )
            bits[:, :, 0] = np.real(received_symbols) < 0
            bits[:, :, 1] = np.abs(np.real(received_symbols)) < 2 / np.sqrt(10)
            bits[:, :, 2] = np.imag(received_symbols) < 0
            bits[:, :, 3] = np.abs(np.imag(received_symbols)) < 2 / np.sqrt(10)
            return bits.reshape(received_symbols.shape[0], -1)

    def ifft(self, symbols):
        return np.fft.ifft(symbols, axis=-1)

    def add_cp(self, ofdm_signal):
        return np.concatenate(
            [ofdm_signal[:, -self.cp_length :], ofdm_signal], axis=1
        )

    def remove_cp(self, received):
        return received[:, self.cp_length :]

    def fft(self, time_signal):
        return np.fft.fft(time_signal, axis=-1)

    def transmit(self, bits):
        symbols = self.modulate(bits)
        time_signal = self.ifft(symbols)
        tx_signal = self.add_cp(time_signal)
        return tx_signal, symbols

    def receive(self, rx_signal):
        without_cp = self.remove_cp(rx_signal)
        freq_signal = self.fft(without_cp)
        bits = self.demodulate(freq_signal)
        return bits, freq_signal

    def calculate_ber(self, tx_bits, rx_bits):
        errors = np.sum(tx_bits != rx_bits)
        return errors / tx_bits.size
