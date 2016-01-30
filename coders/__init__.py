from typing import Iterator, Union, Tuple, List
from abc import ABCMeta, abstractclassmethod
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
        if isinstance(it_current, tuple):
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


class BitStream(np.ndarray):

    def __new__(cls, input_obj, symbolsize=1):
        obj = np.array(input_obj, dtype=int).view(cls)
        obj.symbolsize = symbolsize
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.symbolsize = getattr(obj, 'symbolsize', None)

    def __repr__(self):
        if self.symbolsize > 1:
            source = super().__repr__().splitlines()
            extra = ', symbolsize={}'.format(self.symbolsize)
            if len(source[-1]) + len(extra) < 72:
                source[-1] = '{}{})'.format(source[-1][:-1], extra)
            else:
                source[-1] = source[-1][:-1] + ','
                source.append(extra[2:] + ')')
            return '\n'.join(source)
        return super().__repr__()

    def assymbolsize(self, symbolsize: int) -> 'BitStream':
        if self.symbolsize != 1:
            src = BitStream(list(''.join(map(
                    lambda n: bin(n)[2:].zfill(self.symbolsize), self))))
        else:
            src = self
        extra = int(np.ceil(len(src) / symbolsize) * symbolsize) - len(src)
        padded = np.append(src, [0] * extra)
        iters = [padded[i:] for i in range(symbolsize)]
        new = np.array(list(zip(*iters))[::symbolsize]).astype(int).astype(str)
        return type(src)([int(''.join(s), 2) for s in new],
                         symbolsize=symbolsize)


class WavStream(np.ndarray):

    DEFAULT_PEAK_WIDTH = np.arange(20, 21)

    def __new__(cls, input_obj, rate, symbol_duration):
        obj = np.asarray(input_obj, dtype=float).view(cls)
        obj.rate = rate
        obj.symbol_duration = symbol_duration
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.rate = getattr(obj, 'rate', None)
        self.symbol_duration = getattr(obj, 'symbol_duration', None)

    def symbols(self) -> Tuple['WavStream']:
        return (self[self.symbol_duration*n:self.symbol_duration*(n+1)]
                for n in range(round(len(self) / self.symbol_duration)))

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
                              symbol_duration=self.symbol_duration)
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

    @abstractclassmethod
    def encode(cls, stream):
        pass

    @abstractclassmethod
    def decode(cls, rate, stream):
        pass


# TODO: omit first/last peak
class SimpleASK(Encoder):

    low_amplitude = Parameter(0.2)
    high_amplitude = Parameter(1)
    symbol_size = Parameter(2)
    symbol_duration = Parameter(0.2)
    frequency = Parameter(100)
    rate = Parameter(44100)

    peak_width_start = Parameter(0.2)
    peak_width_span = Parameter(0.0)
    peak_threshold = Parameter(5.0e-3)

    filter_window_base = Parameter(40)
    filter_window_scale = Parameter(0.1)
    filter_shape = Parameter(0.5)
    filter_std_base = Parameter(20)
    filter_std_scale = Parameter(0.05)

    @classmethod
    def encode(cls, stream: BitStream):
        f = cls.frequency.current
        r = cls.rate.current
        low_amp = cls.low_amplitude.current
        high_amp = cls.high_amplitude.current
        symbol_size = cls.symbol_size.current
        symbol_len = int(round(r * cls.symbol_duration.current))

        stream = stream.assymbolsize(symbol_size)
        stream_len = len(stream) * symbol_len
        levels = 2**symbol_size
        step_amp = (high_amp - low_amp) / (levels - 1)

        base = np.linspace(0, f * 2 * np.pi * len(stream) * symbol_len / r,
                           stream_len)
        reshape = ((stream * step_amp) + low_amp)
        print('Reshape:', reshape.round(2))
        wave = np.multiply(np.sin(base), reshape.repeat(symbol_len))

        return wave

    @classmethod
    def decode(cls, rate, stream):
        # Prepare main variables
        f = cls.frequency.current
        r = rate
        period = r/f
        low_amp = cls.low_amplitude.current
        high_amp = cls.high_amplitude.current
        symbol_size = cls.symbol_size.current
        symbol_len = int(round(rate * cls.symbol_duration.current))
        levels = np.linspace(low_amp, high_amp, 2**symbol_size)  # type: np.ndarray  # noqa
        stream = WavStream(stream, rate, symbol_len)
        print('Main vars:', symbol_len, len(stream), round(period), levels.round(2))

        # Filter stream
        f_window = cls.filter_window_base.current + \
                   int(round(r * cls.filter_window_scale.current / f))
        f_shape = cls.filter_shape.current  # type: float
        f_std = cls.filter_std_base.current + \
                int(round(r * cls.filter_std_scale.current / f))
        stream = stream.filter(f_window, f_shape, f_std)
        print('Filter vars:', f_window, f_shape, f_std)

        # Peak detection
        pw_start = int(round(r * cls.peak_width_start.current / f))
        pw_span = max(1, int(round(r * cls.peak_width_span.current / f)))
        pw_range = np.arange(pw_start, pw_start + pw_span)
        peak_thresh = cls.peak_threshold.current
        print('Peak vars:', pw_start, pw_span, pw_range)

        # Create solution
        retval = []
        for symbol in stream.symbols():
            peaks = np.array(list(map(lambda v: v[1],
                                      symbol.peaks(pw_range, peak_thresh))))
            peak = np.abs(peaks).mean()
            print('peak', peak.round(2))
            retval.append(np.abs(levels - peak).argmin())
        return BitStream(retval, symbolsize=symbol_size)


# TODO: omit first/last peak
class SimplePSK(Encoder):

    symbol_size = Parameter(2)
    symbol_duration = Parameter(0.2)
    frequency = Parameter(50)
    rate = Parameter(5000)

    peak_width_start = Parameter(0.2)
    peak_width_span = Parameter(0.0)
    peak_threshold = Parameter(5.0e-3)

    filter_window_base = Parameter(40)
    filter_window_scale = Parameter(0.1)
    filter_shape = Parameter(0.5)
    filter_std_base = Parameter(20)
    filter_std_scale = Parameter(0.05)

    zeroes_width = Parameter(0.2)
    zeroes_threshold = Parameter(0.25)

    @classmethod
    def encode(cls, stream: BitStream):
        f = cls.frequency.current
        r = cls.rate.current
        symbol_size = cls.symbol_size.current
        symbol_len = int(round(r * cls.symbol_duration.current))

        stream = stream.assymbolsize(symbol_size)
        stream_len = len(stream) * symbol_len
        levels = 2**symbol_size

        base = np.linspace(0, f * 2 * np.pi * len(stream) * symbol_len / r,
                           stream_len)
        shifts = (stream * 2 * np.pi / levels)
        print('Shifts:', shifts.round(2))
        wave = np.sin(base - shifts.repeat(symbol_len))

        return wave

    @classmethod
    def decode(cls, rate, stream):
        # Prepare main variables
        f = cls.frequency.current
        r = rate
        period = r/f
        symbol_size = cls.symbol_size.current
        symbol_len = int(round(rate * cls.symbol_duration.current))
        stream_len = len(stream)
        levels = 2**symbol_size
        shifts = np.linspace(0, 1, levels+1)
        stream = WavStream(stream, r, symbol_len)
        print('Main vars:', symbol_len, len(stream), round(period))

        # Filter stream
        f_window = cls.filter_window_base.current + \
                   int(round(r * cls.filter_window_scale.current / f))
        f_shape = cls.filter_shape.current  # type: float
        f_std = cls.filter_std_base.current + \
                int(round(r * cls.filter_std_scale.current / f))
        stream = stream.filter(f_window, f_shape, f_std)
        print('Filter vars:', f_window, f_shape, f_std)

        # Peak detection
        pw_start = int(round(r * cls.peak_width_start.current / f))
        pw_span = max(1, int(round(r * cls.peak_width_span.current / f)))
        pw_range = np.arange(pw_start, pw_start + pw_span)
        peak_thresh = cls.peak_threshold.current
        peaks = stream.peaks(pw_range, peak_thresh)
        positives = [i for i, v in peaks if v > 0]
        negatives = [i for i, v in peaks if v < 0]
        positives = val_split(positives, symbol_len, stream_len, size=True)
        negatives = val_split(negatives, symbol_len, stream_len, size=True)
        negatives_stream = [((n % period / period + 0.25) % 1 * levels)
                            for n in negatives]
        positives_stream = [((p % period / period + 0.75) % 1 * levels)
                            for p in positives]
        peaks_stream = [np.concatenate(s)
                        for s in zip(negatives_stream, positives_stream)]
        print('Peak vars:', pw_start, pw_span, pw_range)

        # Zeroes detection
        # TODO: zeroes reinforcement
        zeroes_width = int(round(r * cls.zeroes_width.current / f))
        zeroes_threshold = cls.zeroes_threshold.current
        zeroes = stream.zeroes(zeroes_width, zeroes_threshold)
        zeroes = val_split(zeroes, symbol_len, stream_len, size=True)
        print('Zeroes vars:', zeroes_width)

        return BitStream([(p.round().astype(int) % levels).mean().astype(int) for p in peaks_stream])


class SimpleFSK(Encoder):

    symbol_size = Parameter(2)
    symbol_duration = Parameter(0.2)
    frequency = Parameter(50)
    frequency_dev = Parameter(20)
    rate = Parameter(5000)

    peak_width_start = Parameter(0.2)
    peak_width_span = Parameter(0.0)
    peak_threshold = Parameter(5.0e-3)

    filter_window_base = Parameter(20)
    filter_window_scale = Parameter(0.05)
    filter_shape = Parameter(0.5)
    filter_std_base = Parameter(10)
    filter_std_scale = Parameter(0.05)

    @classmethod
    def encode(cls, stream: BitStream):
        f = cls.frequency.current
        f_low = f - cls.frequency_dev.current
        f_high = f + cls.frequency_dev.current
        r = cls.rate.current
        symbol_size = cls.symbol_size.current
        symbol_len = int(round(r * cls.symbol_duration.current))

        stream = stream.assymbolsize(symbol_size)
        stream_len = len(stream) * symbol_len
        levels = 2**symbol_size
        f_step = (f_high - f_low) / (levels - 1)

        base = np.linspace(0, 2 * np.pi * len(stream) * symbol_len / r,
                           stream_len)
        f_map = ((stream * f_step) + f_low)
        print('Frequency map:', f_map.round(2))
        wave = np.sin(np.multiply(base, f_map.repeat(symbol_len)))

        return wave

    @classmethod
    def decode(cls, rate, stream):
        # Prepare main variables
        f = cls.frequency.current
        r = rate
        period = r/f
        f_low = f - cls.frequency_dev.current
        f_high = f + cls.frequency_dev.current
        symbol_size = cls.symbol_size.current
        symbol_len = int(round(rate * cls.symbol_duration.current))
        levels = np.linspace(f_low, f_high, 2**symbol_size)  # type: np.ndarray  # noqa
        stream = WavStream(stream, rate, symbol_len)
        print('Main vars:', symbol_len, len(stream), round(period), levels.round(2))

        # Filter stream
        f_window = cls.filter_window_base.current + \
                   int(round(r * cls.filter_window_scale.current / f))
        f_shape = cls.filter_shape.current  # type: float
        f_std = cls.filter_std_base.current + \
                int(round(r * cls.filter_std_scale.current / f))
        stream = stream.filter(f_window, f_shape, f_std)
        print('Filter vars:', f_window, f_shape, f_std)

        # Peak detection
        pw_start = int(round(r * cls.peak_width_start.current / f))
        pw_span = max(1, int(round(r * cls.peak_width_span.current / f)))
        pw_range = np.arange(pw_start, pw_start + pw_span)
        peak_thresh = cls.peak_threshold.current
        print('Peak vars:', pw_start, pw_span, pw_range)

        # Create solution
        retval = []
        for symbol in stream.symbols():
            peaks = np.array(symbol.fft_peaks(pw_range, peak_thresh))
            peak = np.average(peaks[:,2], weights=peaks[:,1])
            print('peak', peak.round(2))
            retval.append(np.abs(levels - peak).argmin())
        return BitStream(retval, symbolsize=symbol_size)
