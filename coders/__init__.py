from typing import Union, Tuple, List, Optional, Dict, Any, Iterable
from numbers import Number
from abc import ABCMeta, abstractmethod
import numpy as np
import scipy
import scipy.signal
import scipy.fftpack
import scipy.stats


def val_split(a: Iterable, partitions: int, range_max: int, range_min: int=0,
              size: bool=True) -> List[np.ndarray]:
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
                        ret_val[i].append([it_current, it_value])
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


def rint(a: Number) -> int:
    return np.round(a).astype(int)


def cyclic_d(values: np.ndarray, lim: int) -> int:
    values %= lim
    lim_case = np.array([np.minimum(np.square(values - lim),
                                    np.square(values))])
    m = np.array([np.minimum(np.square(n - values),
                             np.square(n + values))
                  for n in np.arange(1, 4)])
    return np.concatenate((lim_case, m), axis=0).mean(axis=1).argmin()


def c2p(v: Union[Tuple, List, np.ndarray]) -> np.ndarray:
    if not isinstance(v, np.ndarray):
        v = np.array(v)
    if len(v.shape) == 1:
        v = np.array([v])
    return np.array([np.sqrt(np.square(v).sum(axis=1)) / np.sqrt(2),
                     np.arctan2(v[:, 1], v[:, 0]) % (2*np.pi)]).T


def p2c(v: Union[Tuple, List, np.ndarray]) -> np.ndarray:
    if not isinstance(v, np.ndarray):
        v = np.array(v)
    if len(v.shape) == 1:
        v = np.array([v])
    return (np.array([np.cos(v[:, 1]),
                      np.sin(v[:, 1])]) * v[:, 0] * np.sqrt(2)).T


def sync_padding(coder: 'Encoder', duration: float=0.4) -> 'WavStream':
    transition = rint(duration * 0.15 * coder.r)
    total = rint(duration * coder.r)

    base = np.sin(np.linspace(0, coder.f * 2 * np.pi * duration, total))
    transform = np.concatenate((np.zeros(transition),
                                ease(np.linspace(0, 1, transition)),
                                np.ones(total - 4 * transition),
                                ease(np.linspace(1, 0, transition)),
                                np.zeros(transition)))

    return WavStream(base * transform, coder.r, total)


def ease(x: Union[Number, np.ndarray], a: Number=2) \
        -> Union[Number, np.ndarray]:
    return x**a / (x**a + (1 - x)**a)


def min_error(a: np.ndarray, b: np.ndarray, shift: int, l: Optional[int]=None,
              w: int=0) -> int:
    if l is None:
        l = len(a)
    shifts = np.arange(-shift, shift)
    errors = np.array([np.square(a[w:l-w] - b[w+n:l-w+n]).sum().item()
                       for n in shifts])
    return shifts[errors.argmin()]


def fitness(bit_rate: float, error_rate: float) -> float:
    return bit_rate/error_rate**2


class BitStream(np.ndarray):

    def __new__(cls, input_obj, symbolwidth: int=1):
        obj = np.array(input_obj, dtype=int).view(cls)
        obj.symbolwidth = symbolwidth
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.symbolwidth = getattr(obj, 'symbolwidth', None)

    def __repr__(self) -> str:
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
        new = np.array(list(zip(*iters))[::symbolwidth]) \
            .astype(int).astype(str)
        return type(src)([int(''.join(s), 2) for s in new],
                         symbolwidth=symbolwidth)

    def __eq__(self, other: Any) -> Union[bool, List[bool]]:
        self_ = self
        other_ = other
        if isinstance(other_, BitStream):
            other_ = other_.assymbolwidth(1)
            if self_.symbolwidth != 1:
                self_ = self_.assymbolwidth(1)
            clip = len(self_) - len(other_)
            if 0 < clip < self.symbolwidth and all(self_[-clip:] == 0):
                self_ = self_[:len(other_)]
        return super(BitStream, self_).__eq__(other_)


class WavStream(np.ndarray):

    DEFAULT_PEAK_WIDTH = np.arange(20, 21)

    def __new__(cls, input_obj, rate: int, symbol_len: int):
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

    def filter(self, window_size: int, shape: float, std: float) \
            -> 'WavStream':
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
    def _peaks(cls, stream: np.ndarray, peak_width: Optional[np.ndarray]=None,
               threshold: float=5.0e-2, ref: List=None,
               relocate_peak: bool=False) \
            -> List[Union[Tuple[int, float], Tuple[int, float, float]]]:
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

    def fft_peaks(self, peak_width: Optional[np.ndarray]=None,
                  threshold: float=5.0e-2, axis: bool=False) \
            -> List[Tuple[int, float, float]]:
        xf, yf = self.fft()
        if axis:
            return xf, yf
        return self._peaks(yf, peak_width, threshold, relocate_peak=True,
                           ref=xf)

    def peaks(self, peak_width: Optional[np.ndarray]=None,
              threshold: float=5.0e-2) -> List[Tuple[int, float]]:
        results = (self._peaks(self, peak_width, threshold) +
                   list(map(lambda v: (v[0], -v[1]),
                            self._peaks(-self, peak_width, threshold))))
        results.sort(key=lambda v: v[0])
        return results

    # TODO: optimize and improve
    def zeroes(self, zero_width: int=20, threshold: float=0.25) -> List[int]:
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

    def __init__(self, start: Union[int, float],
                 stop: Union[int, float, None]=None,
                 default: Union[int, float, None]=None,
                 scale: Union[int, float, None]=None, log: bool=False,
                 shift: Union[int, float]=0, poly: Union[int, float]=1,
                 forced_type: Optional[type]=None):
        self.start = start
        self.stop = stop or start
        if self.start > self.stop:
            self.start, self.stop = self.stop, self.start

        # Type and is log
        if forced_type is not None:
            self.type = forced_type
        elif (isinstance(self.start, int) and isinstance(self.stop, int) and
                (isinstance(default, int) or default is None)):
            self.type = int
        else:
            self.type = float
        self.poly = poly
        self.log = log
        self.shift = shift

        # Set default
        if default is None:
            self._current = self._random()
        else:
            self._current = default

        # Set scale
        if isinstance(scale, Number):
            self.scale = scale
        else:
            self.scale = abs(self.start - self.stop) / 3

    def _random(self) -> float:
        if self.log:
            r = np.log(self.stop) - np.log(self.start)
            x = np.e**(np.random.random() * r + np.log(self.start))
        else:
            r = self.stop - self.start
            x = np.random.random() * r + self.start
        if x < self.start:
            return self.start
        if x > self.stop:
            return self.stop
        return x

    def random(self) -> 'Parameter':
        return type(self)(self.start, self.stop, self._random(),
                          scale=self.scale, log=self.log,
                          forced_type=self.type)

    def _mutate(self, scale: float=1) -> float:
        c = (self._current + self.shift) ** (1/self.poly)
        s = (self.scale * scale) ** (1/self.poly)
        if self.log:
            v = np.random.lognormal(mean=np.log(c), sigma=np.log(s))
        else:
            v = np.random.normal(loc=c, scale=s)
        return (v - self.shift)**self.poly

    def mutate(self, scale: float=1) -> 'Parameter':
        # Get value within bounds
        v = self._mutate(scale)
        while not self.start <= v <= self.stop:
            v = self._mutate(scale)

        # Create a new object
        return type(self)(self.start, self.stop, v, scale=self.scale,
                          log=self.log, forced_type=self.type)

    def cross(self, other: 'Parameter', strength: float=1) \
            -> 'Parameter':
        c = (self._current + self.shift) ** (1/self.poly)
        o = (other._current + self.shift) ** (1/self.poly)
        if self.log:
            c = np.log(c)
            o = np.log(o)
        if c < o:
            c, o = o, c
        if c-o <= 0:
            v = c
        else:
            v = np.random.normal(loc=(o+c)/2, scale=(c-o)*strength/3)
        while not self.start <= v <= self.stop:
            v = np.random.normal(loc=(o+c)/2, scale=(c-o)*strength/3)
        if self.log:
            v = np.e**v

        # Create a new object
        return type(self)(self.start, self.stop, v, scale=self.scale,
                          log=self.log, forced_type=self.type)

    def set(self, value: Union[int, float]) -> Union[int, float]:
        self._current = value
        return self.current

    def copy(self) -> 'Parameter':
        return type(self)(self.start, self.stop, self._current,
                          scale=self.scale, log=self.log,
                          forced_type=self.type)

    @property
    def current(self) -> Union[int, float]:
        if self.type == int:
            return rint(self._current)
        return self._current

    @property
    def c(self) -> Union[int, float]:
        return self.current

    def __repr__(self) -> str:
        base = ('Parameter({}, {}, {}, scale={}'
                .format(self.start, self.stop, round(self._current, 2),
                        round(self.scale, 2), self.log))
        if self.log:
            base += ', log=True'
        if self.shift != 0:
            base += ', shift={}'.format(round(self.shift, 2))
        if self.poly != 1:
            base += ', poly={}'.format(round(self.poly, 2))
        return base + ', forced_type={})'.format(self.type.__name__)


class Encoder(object, metaclass=ABCMeta):

    symbol_width = Parameter(1, 3, 2)
    symbol_duration = Parameter(0.001, 1.0, 0.2)
    frequency = Parameter(1, 10000, 400)
    rate = Parameter(8000, 44100, 16000)

    filter_window_base = Parameter(1, 500, 20)
    filter_window_scale = Parameter(0.0, 1.0, 0.1)
    filter_shape = Parameter(0.5, 1.0, 0.5)
    filter_std_base = Parameter(1, 250, 10)
    filter_std_scale = Parameter(0.0, 0.5, 0.05)

    peak_width_start = Parameter(0.0, 1.0, 0.2)
    peak_width_span = Parameter(0.0, 1.0, 0.0)
    peak_threshold = Parameter(0.0, 1.0, 5.0e-3)

    def __init__(self):
        # Clone parameters
        for p, v in self.parameters.items():
            setattr(self, p, v.copy())

        # Main vars
        self.symbol_size = 2**self.symbol_width.c
        self.f = self.frequency.c
        self.r = self.rate.c
        print('Main vars:', self.f, self.r, self.symbol_width.c,
              self.symbol_duration.c)

        # Secondary vars
        self.λ = self.r / self.f
        self.symbol_len = rint(self.r * self.symbol_duration.c)
        print('Secondary vars:', round(self.λ), self.symbol_len)

        # Filter vars
        self.filter_window = rint(self.filter_window_base.c +
                                  self.filter_window_scale.c * self.λ)
        self.filter_std = rint(self.filter_std_base.c +
                               self.filter_std_scale.c * self.λ)
        print('Filter vars:', self.filter_window, self.filter_shape.c,
              self.filter_std)

        # Peak vars
        p_start = max(1, rint(self.peak_width_start.c * self.λ))
        p_span = max(1, rint(self.peak_width_span.c * self.λ))
        p_range = np.arange(p_start, p_start + p_span)
        p_num = max(1, len(p_range) // 5)
        self.peak_range = p_range[::p_num]
        print('Peak vars:', self.peak_range, self.peak_threshold.c)

    def __repr__(self) -> str:
        return '{}:\n    {}'.format(type(self).__name__, '\n    '.join(
            '{}: {}'.format(p, v.c) for p, v in self.parameters.items()))

    @property
    def parameters(self) -> Dict[str, Parameter]:
        return {p: getattr(self, p) for p in dir(self)
                if p != 'parameters' and
                isinstance(getattr(self, p), Parameter)}

    @classmethod
    def random(cls) -> 'Encoder':
        new = cls()
        for p, v in new.parameters.items():
            setattr(new, p, v.random())
        return new

    def mutate(self, amount: float=1/3, scale=1) -> 'Encoder':
        new = type(self)()
        for p, v in self.parameters.items():
            if np.random.random() <= amount:
                print('mutating', p)
                v = v.mutate(scale)
            getattr(new, p).set(v.c)
        return new

    def cross(self, other: 'Encoder', crossed_amount: float=1/3) -> 'Encoder':
        new = type(self)()
        for p, v in self.parameters.items():
            if np.random.random() <= crossed_amount:
                print('crossing', p)
                v = v.cross(getattr(other, p))
            elif np.random.randint(2):
                print('stealing', p)
                v = getattr(other, p)
            getattr(new, p).set(v.c)
        return new

    def filter(self, stream: WavStream) -> WavStream:
        return stream.filter(self.filter_window, self.filter_shape.c,
                             self.filter_std)

    @abstractmethod
    def encode(self, stream: BitStream) -> WavStream:
        pass

    @abstractmethod
    def decode(self, stream: WavStream) -> BitStream:
        pass


# TODO: second levels parameter for decoding balance
class SimpleASK(Encoder):

    high_amplitude = Parameter(0.0, 1.0, 0.8)
    low_amplitude = Parameter(0.0, 0.9, 0.25)

    def __init__(self):
        super().__init__()
        self.low_amp = self.low_amplitude.c * self.high_amplitude.c
        self.step_amp = ((self.high_amplitude.c * (1 - self.low_amplitude.c)) /
                         (self.symbol_size - 1))

    def encode(self, stream: BitStream) -> WavStream:
        stream = stream.assymbolwidth(self.symbol_width.c)
        stream_len = len(stream) * self.symbol_len
        stream_max = (self.f * 2 * np.pi * len(stream) * self.symbol_len /
                      self.r)

        base = np.linspace(0, stream_max, stream_len)
        reshape = (stream * self.step_amp) + self.low_amp
        print('Reshape:', reshape.round(2))
        return WavStream(np.sin(base) * reshape.repeat(self.symbol_len),
                         self.r, self.symbol_len)

    def decode(self, stream: WavStream) -> BitStream:
        symbol_len = rint(stream.rate * self.symbol_duration.c)
        levels, _ = np.linspace(self.low_amp, self.high_amplitude.c,
                                self.symbol_size, retstep=True)
        stream = self.filter(WavStream(stream, stream.rate, symbol_len))

        retval = []
        for symbol in stream.symbols():
            peaks = np.array(symbol.peaks(self.peak_range,
                                          self.peak_threshold.c))
            # TODO: repeat peaks and do square distance from levels
            peak = np.abs(peaks[:, 1]).mean()
            value = np.square(levels - peak).argmin()
            print('>', value, round(peak, 2))
            retval.append(value)
        return BitStream(retval, symbolwidth=self.symbol_width.c)


class SimplePSK(Encoder):

    zeroes_width = Parameter(0.05, 1.0, 0.2)
    zeroes_threshold = Parameter(0.0, 1.0, 0.25)

    def encode(self, stream: BitStream) -> WavStream:
        stream = stream.assymbolwidth(self.symbol_width.c)
        stream_len = len(stream) * self.symbol_len
        stream_max = (self.f * 2 * np.pi * len(stream) * self.symbol_len /
                      self.r)

        base, _ = np.linspace(0, stream_max, stream_len, retstep=True)
        shifts = (stream * 2 * np.pi / self.symbol_size)
        print('Shifts:', shifts.round(2))
        return WavStream(np.sin(base - shifts.repeat(self.symbol_len)),
                         self.r, self.symbol_len)

    def decode(self, stream: WavStream) -> BitStream:
        λ = stream.rate / self.f
        symbol_len = rint(stream.rate * self.symbol_duration.c)
        stream = self.filter(WavStream(stream, stream.rate, symbol_len))

        retval = []
        for s in stream.symbols():
            peaks = np.array(s.peaks(self.peak_range,
                                     self.peak_threshold.c))
            positives = peaks[:, 0][peaks[:, 1] > 0]
            negatives = peaks[:, 0][peaks[:, 1] < 0]
            negatives2 = np.array(negatives) % λ / λ + 0.25
            positives2 = np.array(positives) % λ / λ + 0.75
            peaks_stream = (np.concatenate((negatives2, positives2)) % 1 *
                            self.symbol_size)
            value = cyclic_d(peaks_stream, self.symbol_size)
            retval.append(value)
            print('>', value, peaks_stream.mean().round(2))
        return BitStream(retval, symbolwidth=self.symbol_width.c)

    # TODO: reevaluate with better peaks implementation
    def decode_(self, stream: WavStream) -> BitStream:
        λ = stream.rate / self.f
        symbol_len = rint(stream.rate * self.symbol_duration.c)
        stream_len = len(stream)
        stream = self.filter(WavStream(stream, stream.rate, symbol_len))

        peaks = np.array(stream.peaks(self.peak_range, self.peak_threshold.c))
        positives = peaks[:, 0][peaks[:, 1] > 0]
        negatives = peaks[:, 0][peaks[:, 1] < 0]
        positives2 = val_split(positives, symbol_len, stream_len, size=True)
        negatives2 = val_split(negatives, symbol_len, stream_len, size=True)
        negatives_stream = (np.array(negatives2) % λ / λ + 0.25)
        positives_stream = (np.array(positives2) % λ / λ + 0.75)
        peaks_stream = (np.array([np.concatenate(s) for s in
                                  zip(negatives_stream, positives_stream)]) %
                        1 * self.symbol_size)
        peaks_stream2 = (np.array([np.concatenate(s) for s in
                                   zip(negatives_stream, positives_stream)]) %
                         1 * 2 * np.pi)
        for p in peaks_stream2:
            print('>', p.mean().round(2))

        # TODO: zeroes reinforcement
        # zeroes = stream.zeroes(rint(self.zeroes_width.c * self.λ),
        #                        self.zeroes_threshold.c)
        # zeroes = val_split(zeroes, symbol_len, stream_len, size=True)

        return BitStream(list(map(lambda v: cyclic_d(v, self.symbol_size),
                                  peaks_stream)),
                         symbolwidth=self.symbol_width.c)


class SimpleFSK(Encoder):

    frequency_dev = Parameter(0.01, 1.0, 0.25)

    def __init__(self):
        super().__init__()
        self.f_low = self.f * (1 - self.frequency_dev.c)
        self.f_high = self.f * (1 + self.frequency_dev.c)
        self.f_step = (self.f_high - self.f_low) / (self.symbol_size - 1)

    def encode(self, stream: BitStream) -> WavStream:
        stream = stream.assymbolwidth(self.symbol_width.c)
        stream_len = len(stream) * self.symbol_len
        stream_max = 2 * np.pi * len(stream) * self.symbol_len / self.r

        base = np.linspace(0, stream_max, stream_len)
        f_map = (stream * self.f_step) + self.f_low
        print('Frequency map:', f_map.round(2))
        return WavStream(np.sin(base * f_map.repeat(self.symbol_len)),
                         self.r, self.symbol_len)

    def decode(self, stream: WavStream) -> BitStream:
        symbol_len = rint(stream.rate * self.symbol_duration.c)
        levels, _ = np.linspace(self.f_low, self.f_high, self.symbol_size,
                                retstep=True)
        stream = self.filter(WavStream(stream, stream.rate, symbol_len))

        retval = []
        for symbol in stream.symbols():
            peaks = np.array(symbol.fft_peaks(self.peak_range,
                                              self.peak_threshold.c))
            peak = np.average(peaks[:, 2], weights=peaks[:, 1])
            value = np.square(levels - peak).argmin()
            print('>', value, peak.round(2))
            retval.append(value)
        return BitStream(retval, symbolwidth=self.symbol_width.c)


class SimpleQAM(Encoder):

    symbol_width = Parameter(4)

    def __init__(self):
        super().__init__()
        # TODO: generate constellation based on symbol width
        self.cartesian = np.array([[x, y] for x in np.linspace(-1, 1, 4)
                                   for y in np.linspace(-1, 1, 4)])
        self.polar = c2p(self.cartesian)

    def encode(self, stream: BitStream) -> WavStream:
        stream = stream.assymbolwidth(self.symbol_width.c)
        stream_len = len(stream) * self.symbol_len
        stream_max = (self.f * 2 * np.pi * len(stream) * self.symbol_len /
                      self.r)

        base, _ = np.linspace(0, stream_max, stream_len, retstep=True)
        qam_map = self.polar[stream]
        print('Shifts:', qam_map[:, 1].round(2))
        print('Reshape:', qam_map[:, 0].round(2))
        return WavStream(np.sin(base - qam_map[:, 1].repeat(self.symbol_len)) *
                         qam_map[:, 0].repeat(self.symbol_len), self.r,
                         self.symbol_len)

    def decode(self, stream: WavStream) -> BitStream:
        λ = stream.rate / self.f
        symbol_len = rint(stream.rate * self.symbol_duration.c)
        stream = self.filter(WavStream(stream, stream.rate, symbol_len))

        retval = []
        for s in stream.symbols():
            peaks = np.array(s.peaks(self.peak_range, self.peak_threshold.c))
            # TODO: squared distance from levels
            amp = np.abs(peaks)[:, 1].mean()
            positives = peaks[:, 0][peaks[:, 1] > 0]
            negatives = peaks[:, 0][peaks[:, 1] < 0]
            negatives2 = np.array(negatives) % λ / λ + 0.25
            positives2 = np.array(positives) % λ / λ + 0.75
            peaks_stream = np.concatenate((negatives2, positives2))
            bad_mean = (peaks_stream % 1 * 2 * np.pi).mean()

            polar = np.array([[amp, bad_mean]])
            cartesian = p2c(polar)

            temp = np.square(self.cartesian - cartesian)
            temp2 = temp[:, 0] + temp[:, 1]
            value = temp2.argmin()
            retval.append(value)
            print('>', value, cartesian.round(2), polar.round(2))
        return BitStream(retval, symbolwidth=self.symbol_width.c)

    # TODO: reevaluate with better peaks implementation
    def decode_(self, stream: WavStream) -> BitStream:
        λ = stream.rate / self.f
        symbol_len = rint(stream.rate * self.symbol_duration.c)
        stream_len = len(stream)
        stream = self.filter(WavStream(stream, stream.rate, symbol_len))

        peaks = np.array(stream.peaks(self.peak_range, self.peak_threshold.c))
        amp = val_split(np.abs(peaks), symbol_len, stream_len, size=True)
        amp2 = [v[:, 1].mean() for v in amp]
        positives = peaks[:, 0][peaks[:, 1] > 0]
        negatives = peaks[:, 0][peaks[:, 1] < 0]
        positives2 = val_split(positives, symbol_len, stream_len, size=True)
        negatives2 = val_split(negatives, symbol_len, stream_len, size=True)
        negatives_stream = (np.array(negatives2) % λ / λ + 0.25)
        positives_stream = (np.array(positives2) % λ / λ + 0.75)
        peaks_stream = (np.array([np.concatenate(s) for s in
                                 zip(negatives_stream, positives_stream)]) %
                        1 * 2 * np.pi)
        bad_mean = np.array([v.mean() for v in peaks_stream])

        polar = np.array(list(zip(amp2, bad_mean)))
        cartesian = p2c(polar)

        retval = []
        for c in cartesian:
            temp = np.square(self.cartesian - c)
            temp2 = temp[:, 0] + temp[:, 1]
            value = temp2.argmin()
            print('>', value, c.round(2))
            retval.append(value)
        return BitStream(retval, symbolwidth=self.symbol_width.c)
