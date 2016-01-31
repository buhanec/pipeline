from typing import Iterator, Union, Tuple, List
from abc import ABCMeta, abstractmethod
import numpy as np
import scipy
import scipy.signal
import scipy.fftpack


def val_split(a, partitions, range_max, range_min=0, size=True):
    if size:
        n = int(np.ceil(range_max / partitions))
        splits = partitions
    else:
        n = partitions
        splits = (range_max - range_min) // partitions

    it = iter(a)
    it_current = next(it)
    ret_val = [[] for _ in range(n)]

    try:
        if isinstance(it_current, (tuple, list, np.ndarray)):
            it_current, it_value = it_current
            for i in range(n):
                for j in range(splits):
                    split_current = (partitions + 1) * i + j
                    while it_current <= split_current:
                        ret_val[i].append((it_current, it_value))
                        it_current, it_value = next(it)
                    continue
            return list(map(np.array, ret_val))
        for i in range(n):
            for j in range(splits):
                split_current = (partitions + 1) * i + j
                while it_current <= split_current:
                    ret_val[i].append(it_current)
                    it_current = next(it)
                continue
    except StopIteration:
        return list(map(np.array, ret_val))


def rint(a):
    return np.round(a).astype(int)


def cyclic_d(values, lim):
    return min([(n, np.minimum(np.square((n - values) % lim),
                               np.square((values + n) % lim)).mean())
                for n in range(lim)], key=lambda n: n[1])[0]


def c2p(v):
    if not isinstance(v, np.ndarray):
        v = np.array(v)
    if len(v.shape) == 1:
        v = np.array([v])
    return np.array([np.sqrt(np.square(v).sum(axis=1)) / np.sqrt(2), np.arctan2(v[:,1], v[:,0]) % (2*np.pi)]).T


def p2c(v):
    if not isinstance(v, np.ndarray):
        v = np.array(v)
    if len(v.shape) == 1:
        v = np.array([v])
    return (np.array([np.cos(v[:,1]), np.sin(v[:,1])]) * v[:,0] * np.sqrt(2)).T


class BitStream(np.ndarray):

    def __new__(cls, input_obj, symbolwidth=1):
        obj = np.array(input_obj, dtype=int).view(cls)
        obj.symbolwidth = symbolwidth
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.symbolwidth = getattr(obj, 'symbolwidth', None)

    def __repr__(self):
        if self.symbolwidth > 1:
            source = super().__repr__().splitlines()
            extra = ', symbolwidth={}'.format(self.symbolwidth)
            if len(source[-1]) + len(extra) < 72:
                source[-1] = '{}{})'.format(source[-1][:-1], extra)
            else:
                source[-1] = source[-1][:-1] + ','
                source.append(extra[2:] + ')')
            return '\n'.join(source)
        return super().__repr__()

    def assymbolwidth(self, symbolwidth: int) -> 'BitStream':
        if self.symbolwidth != 1:
            src = BitStream(list(''.join(map(
                    lambda n: bin(n)[2:].zfill(self.symbolwidth), self))))
        else:
            src = self
        extra = int(np.ceil(len(src) / symbolwidth) * symbolwidth) - len(src)
        padded = np.append(src, [0] * extra)
        iters = [padded[i:] for i in range(symbolwidth)]
        new = np.array(list(zip(*iters))[::symbolwidth]).astype(int).astype(str)
        return type(src)([int(''.join(s), 2) for s in new],
                         symbolwidth=symbolwidth)


class WavStream(np.ndarray):

    DEFAULT_PEAK_WIDTH = np.arange(20, 21)

    def __new__(cls, input_obj, rate, symbol_len):
        obj = np.asarray(input_obj, dtype=float).view(cls)
        obj.rate = rate
        obj.symbol_len = symbol_len
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.rate = getattr(obj, 'rate', None)
        self.symbol_len = getattr(obj, 'symbol_len', None)

    def symbols(self) -> Tuple['WavStream']:
        return (self[self.symbol_len * n:self.symbol_len * (n + 1)]
                for n in range(rint(len(self) / self.symbol_len)))

    def fft(self) -> Tuple[np.ndarray, np.ndarray]:
        d = len(self)
        src = self
        while src.rate is None:
            src = src.base
        r = src.rate
        xf = np.linspace(0, r/2, d/2)
        yf = 2/d * np.abs(scipy.fftpack.fft(self)[:d/2])
        return xf, yf

    def filter(self, window_size, shape, std) -> 'WavStream':
        window = scipy.signal.general_gaussian(window_size, p=shape, sig=std)
        stream_f = type(self)(scipy.signal.fftconvolve(window, self),
                              rate=self.rate,
                              symbol_len=self.symbol_len)
        # TODO: Better rebalancing so it can be used for ASK
        stream_f = ((np.abs(self).mean() / np.abs(stream_f).mean()).item() *
                    stream_f)
        stream_f = np.roll(stream_f, -window_size//2)
        return stream_f

    # TODO: implement frequency search
    # TODO: better search for peak, fit to sine
    # TODO: counter outliers
    # TODO: look at https://gist.github.com/endolith/250860 (alt)
    # TODO: recheck relocate_peak value
    @classmethod
    def _peaks(cls, stream, peak_width=None, threshold=5.0e-2, ref=None,
               relocate_peak=False):
        # Results list
        peaks = []
        # Prepare some args because binding objects in defaults
        if peak_width is None:
            peak_width = cls.DEFAULT_PEAK_WIDTH
        pw_pad = peak_width.mean()/2
        # TODO: time scipy peaks
        scipy_peaks = scipy.signal.find_peaks_cwt(stream, peak_width)
        for n in scipy_peaks:
            if relocate_peak:
                min_n = max(n - pw_pad, 0)
                m = int(min_n + (stream[min_n:n + pw_pad]).argmax())
            else:
                m = n
            if stream[m] > threshold:
                if ref is not None:
                    peaks.append((m, stream[m], ref[m]))
                else:
                    peaks.append((m, stream[m]))
        return peaks

    def fft_peaks(self, peak_width=None, threshold=5.0e-2, debug=False) \
            -> List[Tuple[int, float, float]]:
        xf, yf = self.fft()
        if debug:
            return xf, yf
        return self._peaks(yf, peak_width, threshold, relocate_peak=True,
                           ref=xf)

    def peaks(self, peak_width=None, threshold=5.0e-2) \
            -> List[Tuple[int, float]]:
        results = (self._peaks(self, peak_width, threshold) +
                   list(map(lambda v: (v[0], -v[1]),
                            self._peaks(-self, peak_width, threshold))))
        results.sort(key=lambda v: v[0])
        return results

    # TODO: optimize and improve
    def zeroes(self, zero_width=20, threshold=0.25):
        results = np.where(np.logical_and(self > -threshold,
                                          self < threshold))[0]
        if not len(results):
            return results
        merged = []
        current = []
        a = results[0]
        for b in results[1:]:
            if (b - a) <= zero_width:
                current.append(a)
            elif len(current):
                current.append(a)
                merged.append(int(round(np.mean(current))))
                current = []
            else:
                merged.append(a)
            a = b
        if len(current):
            current.append(a)
            merged.append(int(round(np.mean(current))))
        return merged


class Parameter(object):

    def __init__(self, start, stop=None, num=None, default=np.nan):
        self._start = start
        self._stop = stop or start
        self._num = num or 1
        self._range, self._step = np.linspace(self._start, self._stop,
                                              self._num, retstep=True)
        if default in self._range:  # wait for == None to return array
            self._current = default
        else:
            self._current = self._range[len(self._range) // 2]
        if self._num > 1:
            if float(int(self._step)) == self._step:
                self._type = int
            else:
                self._type = float
        elif float(int(self._start)) == self._start:
            self._type = int
        else:
            self._type = float

    def __iter__(self) -> Iterator[int]:
        for value in self._range:
            self._current = value
            yield value

    @property
    def type(self) -> type:
        return self._type

    @property
    def start(self) -> Union[int, float]:
        if self._type == int:
            return int(self._start)
        return self._start

    @property
    def stop(self) -> Union[int, float]:
        if self._type == int:
            return int(self._stop)
        return self._stop

    @property
    def current(self) -> Union[int, float]:
        if self._type == int:
            return int(self._current)
        return self._current

    @property
    def step(self) -> Union[int, float]:
        if self._type == int:
            return int(self._step)
        return self._step


class Encoder(object, metaclass=ABCMeta):

    symbol_width = Parameter(2)
    symbol_duration = Parameter(0.2)
    frequency = Parameter(75)
    rate = Parameter(5000)

    filter_window_base = Parameter(20)
    filter_window_scale = Parameter(0.1)
    filter_shape = Parameter(0.5)
    filter_std_base = Parameter(10)
    filter_std_scale = Parameter(0.05)

    peak_width_start = Parameter(0.2)
    peak_width_span = Parameter(0.0)
    peak_threshold = Parameter(5.0e-3)

    def __init__(self):
        # Main vars
        self.symbol_width = self.symbol_width.current
        self.symbol_size = 2**self.symbol_width
        self.symbol_duration = self.symbol_duration.current
        self.f = self.frequency.current
        self.r = self.rate.current
        print('Main vars:', self.f, self.r, self.symbol_width,
              self.symbol_duration)

        # Secondary vars
        self.λ = self.r / self.f
        self.symbol_len = rint(self.r * self.symbol_duration)
        print('Secondary vars:', round(self.λ), self.symbol_len)

        # Filter vars
        self.filter_window = rint(self.filter_window_base.current +
                                  self.filter_window_scale.current * self.λ)
        self.filter_shape = self.filter_shape.current
        self.filter_std = rint(self.filter_std_base.current +
                               self.filter_std_scale.current * self.λ)
        print('Filter vars:', self.filter_window, self.filter_shape,
              self.filter_std)

        # Peak vars
        p_start = rint(self.peak_width_start.current * self.λ)
        p_span = max(1, rint(self.peak_width_span.current * self.λ))
        self.peak_range = np.arange(p_start, p_start + p_span)
        self.peak_threshold = self.peak_threshold.current
        print('Peak vars:', self.peak_range, self.peak_threshold)

    def filter(self, stream):
        return stream.filter(self.filter_window, self.filter_shape,
                             self.filter_std)

    @abstractmethod
    def encode(self, stream):
        pass

    @abstractmethod
    def decode(self, rate, stream):
        pass


# TODO: omit first/last peak
class SimpleASK(Encoder):

    low_amplitude = Parameter(0.2)
    high_amplitude = Parameter(1)

    def __init__(self):
        super().__init__()
        self.low_amp = self.low_amplitude.current
        self.high_amp = self.high_amplitude.current
        self.step_amp = (self.high_amp - self.low_amp) / (self.symbol_size - 1)

    def encode(self, stream: BitStream):
        stream = stream.assymbolwidth(self.symbol_width)
        stream_len = len(stream) * self.symbol_len
        stream_max = self.f * 2 * np.pi * len(stream) * self.symbol_len / self.r

        base = np.linspace(0, stream_max, stream_len)
        reshape = (stream * self.step_amp) + self.low_amp
        print('Reshape:', reshape.round(2))
        return np.sin(base) * reshape.repeat(self.symbol_len)

    def decode(self, rate, stream):
        symbol_len = rint(rate * self.symbol_duration)
        levels = np.linspace(self.low_amp, self.high_amp, self.symbol_size)
        stream = self.filter(WavStream(stream, rate, symbol_len))

        retval = []
        for symbol in stream.symbols():
            peaks = np.array(symbol.peaks(self.peak_range, self.peak_threshold))
            peak = np.abs(peaks[:, 1]).mean()
            value = np.square(levels - peak).argmin()
            print('>', value, round(peak, 2))
            retval.append(value)
        return BitStream(retval, symbolwidth=self.symbol_width)


# TODO: omit first/last peak
class SimplePSK(Encoder):

    zeroes_width = Parameter(0.2)
    zeroes_threshold = Parameter(0.25)

    def __init__(self):
        super().__init__()
        self.zeroes_width = rint(self.zeroes_width.current * self.λ)
        self.zeroes_threshold = self.zeroes_threshold.current

    def encode(self, stream: BitStream):
        stream = stream.assymbolwidth(self.symbol_width)
        stream_len = len(stream) * self.symbol_len
        stream_max = self.f * 2 * np.pi * len(stream) * self.symbol_len / self.r

        base = np.linspace(0, stream_max, stream_len)
        shifts = (stream * 2 * np.pi / self.symbol_size)
        print('Shifts:', shifts.round(2))
        return np.sin(base - shifts.repeat(self.symbol_len))

    def decode(self, rate, stream):
        λ = rate / self.f
        symbol_len = rint(rate * self.symbol_duration)
        stream_len = len(stream)
        stream = self.filter(WavStream(stream, rate, symbol_len))

        peaks = np.array(stream.peaks(self.peak_range, self.peak_threshold))
        positives = peaks[:,0][peaks[:,1] > 0]
        negatives = peaks[:,0][peaks[:,1] < 0]
        positives2 = val_split(positives, symbol_len, stream_len, size=True)
        negatives2 = val_split(negatives, symbol_len, stream_len, size=True)
        negatives_stream = (np.array(negatives2) % λ / λ + 0.25)
        positives_stream = (np.array(positives2) % λ / λ + 0.75)
        peaks_stream = np.array([np.concatenate(s) for s in zip(negatives_stream, positives_stream)]) % 1 * self.symbol_size

        # Zeroes detection
        # TODO: zeroes reinforcement
        zeroes = stream.zeroes(self.zeroes_width, self.zeroes_threshold)
        zeroes = val_split(zeroes, symbol_len, stream_len, size=True)

        return BitStream(list(map(lambda v: cyclic_d(v, self.symbol_size), peaks_stream)), symbolwidth=self.symbol_width)

class SimpleFSK(Encoder):

    frequency_dev = Parameter(20)

    def __init__(self):
        super().__init__()
        self.f_low = self.f - self.frequency_dev.current
        self.f_high = self.f + self.frequency_dev.current
        self.f_step = (self.f_high - self.f_low) / (self.symbol_size - 1)

    def encode(self, stream: BitStream):
        stream = stream.assymbolwidth(self.symbol_width)
        stream_len = len(stream) * self.symbol_len
        stream_max = 2 * np.pi * len(stream) * self.symbol_len / self.r

        base = np.linspace(0, stream_max, stream_len)
        f_map = (stream * self.f_step) + self.f_low
        print('Frequency map:', f_map.round(2))
        return np.sin(base * f_map.repeat(self.symbol_len))

    def decode(self, rate, stream):
        λ = rate / self.f
        symbol_len = rint(rate * self.symbol_duration)
        levels = np.linspace(self.f_low, self.f_high, self.symbol_size)
        stream = self.filter(WavStream(stream, rate, symbol_len))

        retval = []
        for symbol in stream.symbols():
            peaks = np.array(symbol.fft_peaks(self.peak_range, self.peak_threshold))
            peak = np.average(peaks[:,2], weights=peaks[:,1])
            value = np.square(levels - peak).argmin()
            print('>', value, peak.round(2))
            retval.append(value)
        return BitStream(retval, symbolwidth=self.symbol_size)
