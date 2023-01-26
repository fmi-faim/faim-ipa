import numpy as np
from matplotlib import pyplot as plt


class UIntHistogram:
    def __init__(self, data=None):
        """
        An unsigned integer histogram class that can be updated with new data.

        :param data: array[uint]
            Optional data of which the initial histogram is built.
        """
        if data is not None:
            assert data.min() >= 0, "Negative data is not supported."
            self.offset, self.bins, self.frequencies = self._get_hist(data)
        else:
            self.offset, self.bins, self.frequencies = None, None, None

    @staticmethod
    def _add(list_a, list_b):
        """
        Add two lists element-wise.
        :param list_a: list(numeric)
        :param list_b: list(numeric)
        :return: list(numeric)
        """
        return list(map(lambda e: e[0] + e[1], zip(list_a, list_b)))

    @staticmethod
    def _get_hist(data):
        """
        Compute histogram with integer bins.
        :param data: array[numeric]
        :return: offset, bins, frequencies
        """
        offset = int(data.min())
        bins = int(data.max()) + 2 - offset
        freq = np.histogram(data, np.arange(offset, offset + bins))[0].tolist()
        return offset, bins, freq

    def _aggregate_histograms(self, offset_data, bins, freq):
        """
        Integrate new frequencies into existing histogram.

        :param offset_data:
        :param bins:
        :param freq:
        :return:
        """
        assert isinstance(freq, list), "freq must be of type List."
        assert isinstance(self.frequencies, list), (
            "frequencies must be of " "type List."
        )
        lower_shift = offset_data - self.offset
        upper_shift = self.offset + self.bins - (offset_data + bins)

        if lower_shift == 0 and upper_shift == 0:
            # Old and new frequencies cover the same range:
            # [old frequencies]
            # [new frequencies]
            self.frequencies = self._add(self.frequencies, freq)
        elif (
            lower_shift < 0
            and (offset_data + bins - 1 >= self.offset)
            and (offset_data + bins - 1 <= self.offset + self.bins - 1)
        ):
            # New frequencies have additional lower ones.
            #     [old frequencies]
            # [new frequencies]
            frequencies_to = offset_data + bins - self.offset - 1
            freq_from = self.offset - offset_data
            self.frequencies[:frequencies_to] = self._add(
                self.frequencies[:frequencies_to],
                freq[freq_from:],
            )
            self.frequencies = freq[:freq_from] + self.frequencies
            self.offset = offset_data
            self.bins = len(self.frequencies) + 1
        elif lower_shift < 0 and (offset_data + bins - 1 < self.offset):
            # New frequencies only have additional lower ones.
            #                       [old frequencies]
            # [new frequencies]
            gap_freq = [
                0,
            ] * (self.offset - offset_data - bins + 1)
            self.frequencies = freq + gap_freq + self.frequencies
            self.offset = offset_data
            self.bins = len(self.frequencies) + 1
        elif (
            offset_data >= self.offset
            and offset_data <= (self.offset + self.bins - 2)
            and offset_data + bins - 2 > (self.offset + self.bins - 2)
        ):
            # New frequencies have additional upper ones.
            #     [old frequencies]
            #            [new frequencies]
            from_frequencies = self.offset + self.bins - offset_data - 1
            to_freq = self.offset + self.bins - offset_data
            self.frequencies[-from_frequencies:] = self._add(
                self.frequencies[-from_frequencies:],
                freq[:to_freq],
            )
            self.frequencies = self.frequencies + freq[from_frequencies:]
            self.bins = len(self.frequencies) + 1
        elif (offset_data) > (self.offset + self.bins - 2):
            # New frequencies have only additional upper ones.
            #     [old frequencies]
            #                           [new frequencies]
            gap_freq = [
                0,
            ] * (offset_data - self.offset - self.bins + 1)
            self.frequencies = self.frequencies + gap_freq + freq
            self.bins = len(self.frequencies) + 1
        elif lower_shift >= 0 and upper_shift >= 0:
            # New frequencies are completely covered.
            # [          old frequencies          ]
            #           [new frequencies]
            if upper_shift == 0:
                self.frequencies[lower_shift:] = self._add(
                    self.frequencies[lower_shift:], freq
                )
            else:
                self.frequencies[lower_shift:-upper_shift] = self._add(
                    self.frequencies[lower_shift:-upper_shift], freq
                )
        else:
            # Old frequencies are completely covered.
            #     [old frequencies]
            # [    new frequencies    ]
            from_ = self.offset - offset_data
            to = from_ + self.bins - 1
            self.frequencies = self._add(
                self.frequencies,
                freq[from_:to],
            )
            self.frequencies = freq[:from_] + self.frequencies + freq[to:]
            self.offset = offset_data
            self.bins = len(self.frequencies) + 1

    def combine(self, histogram):
        """
        Combine this histogram with another UIntHistogram.

        :param histogram: to merge with this
        """
        if self.frequencies is None:
            self.frequencies = histogram.frequencies
            self.bins = histogram.bins
            self.offset = histogram.offset
        else:
            self._aggregate_histograms(
                offset_data=histogram.offset,
                bins=histogram.bins,
                freq=histogram.frequencies,
            )

    def update(self, data):
        """
        Update histogram by adding more data.

        :param data: array(uint)
            Data to be added to the histogram.
        """
        assert data.min() >= 0, "Negative data is not supported."

        if self.frequencies is None:
            self.offset, self.bins, self.frequencies = self._get_hist(data)
        else:
            offset_data, bins, freq = self._get_hist(data)
            self._aggregate_histograms(offset_data=offset_data, bins=bins, freq=freq)

    def plot(self, width=1):
        """
        Plot histogram.
        """
        if width > 1:
            heights = []
            for i in range(self.offset, self.offset + self.bins - 1, width):
                heights.append(np.sum(self.frequencies[i : i + width]))
        else:
            heights = self.frequencies

        plt.bar(
            np.arange(self.offset, self.offset + self.bins - 1, width),
            heights,
            width=width,
        )
        plt.show()

    def mean(self):
        """
        Get histogram mean.
        :return: float
        """
        return np.sum(
            np.arange(self.offset, self.offset + self.bins - 1) * self.frequencies
        ) / np.sum(self.frequencies)

    def std(self):
        """
        Get histogram standard deviation.
        :return: float
        """
        return np.sqrt(
            np.sum(
                (np.arange(self.offset, self.offset + self.bins - 1) - self.mean()) ** 2
                * self.frequencies
            )
            / np.sum(self.frequencies)
        )

    def quantile(self, q):
        """
        Get quantile `q`.
        :param q: uint
        :return: quantile
        """
        assert q >= 0 and q <= 1
        return self.offset + np.argmax(
            np.cumsum(self.frequencies) / np.sum(self.frequencies) >= q
        )

    def min(self):
        """
        Get minimum.
        :return: uint
        """
        return self.offset

    def max(self):
        """
        Get maximum.
        :return: uint
        """
        return self.offset + self.bins - 2

    def save(self, path):
        np.savez(path, frequencies=self.frequencies, offset=self.offset, bins=self.bins)

    @staticmethod
    def load(path):
        storage = np.load(path)
        hist = UIntHistogram()
        hist.frequencies = storage["frequencies"].tolist()
        hist.offset = storage["offset"]
        hist.bins = storage["bins"]
        return hist
