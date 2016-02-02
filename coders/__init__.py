from typing import Union, Tuple, List
from numbers import Number
from abc import ABCMeta, abstractmethod
import numpy as np
import scipy
import scipy.signal
import scipy.fftpack
import scipy.stats


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


def rint(a):
    return np.round(a).astype(int)


def cyclic_d(values, lim):
    values = values % lim
    lim_case = np.array([np.minimum(np.square(values - lim),
                                    np.square(values))])
    m = np.array([np.minimum(np.square(n - values),
                             np.square(n + values))
                  for n in np.arange(1, 4)])
    return np.concatenate((lim_case, m), axis=0).mean(axis=1).argmin()


def c2p(v):
    if not isinstance(v, np.ndarray):
        v = np.array(v)
    if len(v.shape) == 1:
        v = np.array([v])
    return np.array([np.sqrt(np.square(v).sum(axis=1)) / np.sqrt(2),
                     np.arctan2(v[:, 1], v[:, 0]) % (2*np.pi)]).T


def p2c(v):
    if not isinstance(v, np.ndarray):
        v = np.array(v)
    if len(v.shape) == 1:
        v = np.array([v])
    return (np.array([np.cos(v[:, 1]),
                      np.sin(v[:, 1])]) * v[:, 0] * np.sqrt(2)).T


def sync_padding(coder, duration=0.4):
    transition = rint(duration * 0.15 * coder.r)
    total = rint(duration * coder.r)

    base = np.sin(np.linspace(0, coder.f * 2 * np.pi * duration, total))
    transform = np.concatenate((np.zeros(transition),
                                ease(np.linspace(0, 1, transition)),
                                np.ones(total - 4 * transition),
                                ease(np.linspace(1, 0, transition)),
                                np.zeros(transition)))

    return WavStream(base * transform, coder.r, total)


# TODO: replace with window = scipy.signal.gaussian(51, std=7)
def ease(x: Union[Number, np.ndarray], a=2):
    return x**a / (x**a + (1 - x)**a)


def min_error(a, b, shift, l=None, w=0):
    if l is None:
        l = len(a)
    shifts = np.arange(-shift, shift)
    errors = np.array([np.square(a[w:l-w] - b[w+n:l-w+n]).sum().item()
                       for n in shifts])
    return shifts[errors.argmin()]


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
            extra = ', symbolwidth={}'.format(str(self.symbolwidth))
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

    def __eq__(self, other):
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

    def __init__(self, start, stop=None, default=None, scale=None, log=False,
                 shift=0, poly=1):
        self.start = start
        self.stop = stop or start

        # Type and is log
        if (isinstance(self.start, int) and isinstance(self.stop, int) and
                (isinstance(default, int) or default is None)):
            self.type = int
        else:
            self.type = float
        self.poly = poly
        self.log = log
        self.shift = shift

        # Set default
        if default is None:
            self._current = self.type((self.start + self.stop) / 2)
        else:
            self._current = default

        # Set scale
        if isinstance(scale, Number):
            self.scale = scale
        else:
            self.scale = abs(self.start - self.stop) / 3

    def _mutate(self, scale=1) -> float:
        c = (self._current + self.shift) ** (1/self.poly)
        s = (self.scale * scale) ** (1/self.poly)
        if self.log:
            v = np.random.lognormal(mean=np.log(c), sigma=np.log(s))
        else:
            v = np.random.normal(loc=c, scale=s)
        return (v - self.shift)**self.poly

    def mutate(self, scale=1) -> 'Parameter':
        # Get value within bounds
        v = self._mutate(scale)
        while not self.start <= v <= self.stop:
            v = self._mutate(scale)

        # Create a new object
        return type(self)(self.start, self.stop, v, scale=self.scale,
                          log=self.log)

    def cross(self, other: 'Parameter', strength=1, x=None) -> 'Parameter':
        c = (self._current + self.shift) ** (1/self.poly)
        o = (other._current + self.shift) ** (1/self.poly)
        if self.log:
            c = np.log(c)
            o = np.log(o)
        if c < o:
            c, o = o, c
        v = np.random.normal(loc=(o+c)/2, scale=(c-o)/3)
        while not self.start < v < self.stop:
            v = np.random.normal(loc=(o+c)/2, scale=(c-o)/3)
        if self.log:
            v = np.e**v

        # Create a new object
        return type(self)(self.start, self.stop, v, scale=self.scale,
                          log=self.log)

    @property
    def current(self) -> Union[int, float]:
        return self.type(self._current)

    def __repr__(self):
        base = ('Parameter({}, {}, {}, scale={}'
                .format(self.start, self.stop, round(self._current, 2),
                        self.scale, self.log))
        if self.log:
            base += ', log=True'
        if self.shift != 0:
            base += ', shift={}'.format(self.shift)
        if self.poly != 1:
            base += ', poly={}'.format(self.poly)
        return base + ')'


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
        p_start = max(1, rint(self.peak_width_start.current * self.λ))
        p_span = max(1, rint(self.peak_width_span.current * self.λ))
        p_range = np.arange(p_start, p_start + p_span)
        p_num = max(1, len(p_range) // 5)
        self.peak_range = p_range[::p_num]
        self.peak_threshold = self.peak_threshold.current
        print('Peak vars:', self.peak_range, self.peak_threshold)

    def _post_init(self):
        cls = type(self)
        parameters = {p: getattr(self, p) for p in dir(cls)
                      if isinstance(getattr(cls, p), Parameter)}
        self.parameters = {p: v.current if isinstance(v, Parameter) else v
                           for p, v in parameters.items()}

    def filter(self, stream):
        return stream.filter(self.filter_window, self.filter_shape,
                             self.filter_std)

    @abstractmethod
    def encode(self, stream):
        pass

    @abstractmethod
    def decode(self, stream):
        pass


# TODO: omit first/last peak
# TODO: separate levels for decoding
class SimpleASK(Encoder):

    high_amplitude = Parameter(0.0, 1.0, 0.8)
    low_amplitude = Parameter(0.0, 0.9, 0.25)

    def __init__(self):
        super().__init__()
        self.high_amp = self.high_amplitude.current
        self.low_amp = self.low_amplitude.current * self.high_amp
        self.step_amp = ((self.high_amp - self.low_amp) /
                         (self.symbol_size - 1))
        super()._post_init()

    def encode(self, stream: BitStream):
        stream = stream.assymbolwidth(self.symbol_width)
        stream_len = len(stream) * self.symbol_len
        stream_max = (self.f * 2 * np.pi * len(stream) * self.symbol_len /
                      self.r)

        base = np.linspace(0, stream_max, stream_len)
        reshape = (stream * self.step_amp) + self.low_amp
        print('Reshape:', reshape.round(2))
        return WavStream(np.sin(base) * reshape.repeat(self.symbol_len),
                         self.r, self.symbol_len)

    def decode(self, stream):
        symbol_len = rint(stream.rate * self.symbol_duration)
        levels, _ = np.linspace(self.low_amp, self.high_amp, self.symbol_size,
                                retstep=True)
        stream = self.filter(WavStream(stream, stream.rate, symbol_len))

        retval = []
        for symbol in stream.symbols():
            peaks = np.array(symbol.peaks(self.peak_range,
                                          self.peak_threshold))
            # TODO: repeat peaks and do square distance from levels
            peak = np.abs(peaks[:, 1]).mean()
            value = np.square(levels - peak).argmin()
            print('>', value, round(peak, 2))
            retval.append(value)
        return BitStream(retval, symbolwidth=self.symbol_width)


# TODO: omit first/last peak
class SimplePSK(Encoder):

    zeroes_width = Parameter(0.05, 1.0, 0.2)
    zeroes_threshold = Parameter(0.0, 1.0, 0.25)

    def __init__(self):
        super().__init__()
        self.zeroes_width = rint(self.zeroes_width.current * self.λ)
        self.zeroes_threshold = self.zeroes_threshold.current
        super()._post_init()

    def encode(self, stream: BitStream):
        stream = stream.assymbolwidth(self.symbol_width)
        stream_len = len(stream) * self.symbol_len
        stream_max = (self.f * 2 * np.pi * len(stream) * self.symbol_len /
                      self.r)

        base, _ = np.linspace(0, stream_max, stream_len, retstep=True)
        shifts = (stream * 2 * np.pi / self.symbol_size)
        print('Shifts:', shifts.round(2))
        return WavStream(np.sin(base - shifts.repeat(self.symbol_len)),
                         self.r, self.symbol_len)

    def decode(self, stream):
        λ = stream.rate / self.f
        symbol_len = rint(stream.rate * self.symbol_duration)
        stream = self.filter(WavStream(stream, stream.rate, symbol_len))

        retval = []
        for s in stream.symbols():
            peaks = np.array(s.peaks(self.peak_range, self.peak_threshold))
            positives = peaks[:, 0][peaks[:, 1] > 0]
            negatives = peaks[:, 0][peaks[:, 1] < 0]
            negatives2 = np.array(negatives) % λ / λ + 0.25
            positives2 = np.array(positives) % λ / λ + 0.75
            peaks_stream = (np.concatenate((negatives2, positives2)) % 1 *
                            self.symbol_size)
            value = cyclic_d(peaks_stream, self.symbol_size)
            retval.append(value)
            print('>', value, peaks_stream.mean().round(2))
        return BitStream(retval, symbolwidth=self.symbol_width)

    # TODO: reevaluate with better peaks implementation
    def decode_(self, stream):
        λ = stream.rate / self.f
        symbol_len = rint(stream.rate * self.symbol_duration)
        stream_len = len(stream)
        stream = self.filter(WavStream(stream, stream.rate, symbol_len))

        peaks = np.array(stream.peaks(self.peak_range, self.peak_threshold))
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
        # zeroes = stream.zeroes(self.zeroes_width, self.zeroes_threshold)
        # zeroes = val_split(zeroes, symbol_len, stream_len, size=True)

        return BitStream(list(map(lambda v: cyclic_d(v, self.symbol_size),
                                  peaks_stream)),
                         symbolwidth=self.symbol_width)


class SimpleFSK(Encoder):

    frequency_dev = Parameter(0.01, 1.0, 0.25)

    def __init__(self):
        super().__init__()
        self.f_low = self.f * (1 - self.frequency_dev.current)
        self.f_high = self.f * (1 + self.frequency_dev.current)
        self.f_step = (self.f_high - self.f_low) / (self.symbol_size - 1)
        super()._post_init()

    def encode(self, stream: BitStream):
        stream = stream.assymbolwidth(self.symbol_width)
        stream_len = len(stream) * self.symbol_len
        stream_max = 2 * np.pi * len(stream) * self.symbol_len / self.r

        base = np.linspace(0, stream_max, stream_len)
        f_map = (stream * self.f_step) + self.f_low
        print('Frequency map:', f_map.round(2))
        return WavStream(np.sin(base * f_map.repeat(self.symbol_len)),
                         self.r, self.symbol_len)

    def decode(self, stream):
        symbol_len = rint(stream.rate * self.symbol_duration)
        levels, _ = np.linspace(self.f_low, self.f_high, self.symbol_size,
                                retstep=True)
        stream = self.filter(WavStream(stream, stream.rate, symbol_len))

        retval = []
        for symbol in stream.symbols():
            peaks = np.array(symbol.fft_peaks(self.peak_range,
                                              self.peak_threshold))
            peak = np.average(peaks[:, 2], weights=peaks[:, 1])
            value = np.square(levels - peak).argmin()
            print('>', value, peak.round(2))
            retval.append(value)
        return BitStream(retval, symbolwidth=self.symbol_width)


class SimpleQAM(Encoder):

    symbol_width = Parameter(4)

    def __init__(self):
        super().__init__()
        self.symbol_width = self.symbol_width  # type: int
        # TODO: generate constellation based on symbol width
        self.cartesian = np.array([[x, y] for x in np.linspace(-1, 1, 4)
                                   for y in np.linspace(-1, 1, 4)])
        self.polar = c2p(self.cartesian)
        super()._post_init()

    def encode(self, stream):
        stream = stream.assymbolwidth(self.symbol_width)
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

    def decode(self, stream):
        λ = stream.rate / self.f
        symbol_len = rint(stream.rate * self.symbol_duration)
        stream = self.filter(WavStream(stream, stream.rate, symbol_len))

        retval = []
        for s in stream.symbols():
            peaks = np.array(s.peaks(self.peak_range, self.peak_threshold))
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
        return BitStream(retval, symbolwidth=self.symbol_width)

    def decode_(self, stream):
        λ = stream.rate / self.f
        symbol_len = rint(stream.rate * self.symbol_duration)
        stream_len = len(stream)
        stream = self.filter(WavStream(stream, stream.rate, symbol_len))

        peaks = np.array(stream.peaks(self.peak_range, self.peak_threshold))
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
        return BitStream(retval, symbolwidth=self.symbol_width)
