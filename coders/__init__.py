"""Project core module."""

from typing import Union, Tuple, List, Optional, Dict, Any
from numbers import Number
from abc import ABCMeta, abstractmethod
import numpy as np
import scipy
import scipy.signal
import scipy.fftpack
import scipy.stats

from .util import (rint, c2p, p2c, min_error, lin_trim_mean,
                   lin_trim_error, sq_cyclic_align_error, ease,
                   trim_mean, infs, smooth5)
from .ga import Individual


def sync_padding(coder: 'Encoder', duration: float = 0.4) -> 'WavStream':
    """
    Create synchronisation pattern for a specific waveform.

    :param coder: coder used to generate waveform
    :param duration: synchronisation duration
    :return: synchronisation waveform
    """
    transition = rint(duration * 0.15 * coder.r)
    total = rint(duration * coder.r)

    base = np.sin(np.linspace(0, coder.f * 2 * np.pi * duration, total,
                              endpoint=False))
    transform = np.concatenate((
        np.zeros(transition),
        ease(np.linspace(0, 1, transition, endpoint=False)),
        np.ones(total - 4 * transition),
        ease(np.linspace(1, 0, transition, endpoint=False)),
        np.zeros(transition)))

    return WavStream(base * transform, coder.r, total)


class BitStream(np.ndarray):
    """Represents a stream of data in a base of a power of 2."""

    def __new__(cls, input_obj, symbolwidth: int = 1) -> np.ndarray:
        """
        Create new object.

        :param input_obj: input values
        :param symbolwidth: symbol width
        :return: array with input values
        """
        obj = np.array(input_obj, dtype=int).view(cls)
        obj.symbolwidth = symbolwidth
        return obj

    def __array_finalize__(self, obj):
        """
        Finalise array creation.

        :param obj: object to finalise
        """
        if obj is None:
            return
        self.symbolwidth = getattr(obj, 'symbolwidth', None)

    def __repr__(self) -> str:
        """
        Get object representation.

        :return: representation
        """
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
        """
        Convert `BitStream` to a different base.

        Symbol width represents the binary width of the symbol. The
        final `BitStream` base is `2 ** symbolwidth`.

        :param symbolwidth: symbol width
        :return: `BitSream` with new base
        """
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

    def __and__(self, other: Any) -> Union[bool, List[bool]]:
        """
        Compare original binary representation of two `BitStream`s.

        :param other: `BitStream` object to compare to
        :return: comparison result
        """
        self_ = self
        other_ = other
        if isinstance(other_, BitStream):
            other_ = other_.assymbolwidth(1)
            if self_.symbolwidth != 1:
                self_ = self_.assymbolwidth(1)
            clip = len(self_) - len(other_)
            clipped = self_[-clip:] == 0  # type: np.ndarray
            if 0 < clip < self.symbolwidth and all(clipped):
                self_ = self_[:len(other_)]
        return super(BitStream, self_).__and__(other_)

    def __len__(self) -> int:
        """
        Object length.

        :return: length
        """
        return super().__len__()


class WavStream(np.ndarray):
    """Represents a waveform."""

    DEFAULT_PEAK_WIDTH = np.arange(20, 21)

    def __new__(cls, input_obj, rate: int, symbol_len: int):
        """
        Create new object.

        :param input_obj: input values
        :param rate: waveform rate
        :param symbol_len: waveform symbol len
        :return: array with input values
        """
        obj = np.asarray(input_obj, dtype=float).view(cls)
        obj.rate = rate
        obj.symbol_len = symbol_len
        return obj

    def __array_finalize__(self, obj):
        """
        Finalise array creation.

        :param obj: object to finalise
        """
        if obj is None:
            return
        self.rate = getattr(obj, 'rate', None)
        self.symbol_len = getattr(obj, 'symbol_len', None)

    def symbols(self) -> Tuple['WavStream']:
        """
        Split waveform into symbols.

        :return: symbols
        """
        return (self[self.symbol_len * n:self.symbol_len * (n + 1)]
                for n in range(rint(len(self) / self.symbol_len)))

    def fft(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform FFT on waveform.

        :return: result
        """
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
        """
        Convolve stream with a Gaussian window to filter out noise.

        Uses FFT to speed up processing on large arrays.

        :param window_size: Gaussian window size
        :param shape: Gaussian window shape
        :param std: Gaussian window standard deviation
        :return: Filtered stream
        """
        kernel = scipy.signal.general_gaussian(window_size, p=shape, sig=std)
        self_ = self  # type: np.ndarray

        shape = len(self) + window_size - 1
        fft_shape = (smooth5(shape),)
        fft_self = np.fft.rfftn(self_, fft_shape)
        fft_kernel = np.fft.rfftn(kernel, fft_shape)
        start = (window_size - 1) // 2
        fft_slice = slice(start, len(self) + start)
        ifft_ret = np.fft.irfftn(fft_self * fft_kernel, fft_shape)[fft_slice]

        stream_f = type(self)(ifft_ret, rate=self.rate,
                              symbol_len=self.symbol_len)
        stream_f = stream_f  # type: np.ndarray
        stream_f = ((np.abs(self_).mean() / np.abs(stream_f).mean()).item() *
                    stream_f)
        return stream_f

    def filter2(self, conv_size: int, conv_scale: int) -> 'WavStream':
        """
        Convolve stream with a rectangular window to filter out noise.

        :param conv_size: window size
        :param conv_scale: window scale
        :return: Filtered stream
        """
        kernel = np.ones(conv_size) / conv_scale
        stream_f = type(self)(np.convolve(self, kernel, mode='same'),
                              rate=self.rate,
                              symbol_len=self.symbol_len)
        return stream_f

    def filter3(self, window_size: int, shape: float, std: float) \
            -> 'WavStream':
        """
        Convolve stream with a Gaussian window to filter out noise.

        :param window_size: Gaussian window size
        :param shape: Gaussian window shape
        :param std: Gaussian window standard deviation
        :return: Filtered stream
        """
        kernel = scipy.signal.general_gaussian(window_size, p=shape, sig=std)
        stream_f = type(self)(np.convolve(self, kernel, mode='same'),
                              rate=self.rate,
                              symbol_len=self.symbol_len)
        stream_f = stream_f  # type: np.ndarray
        self_ = self  # type: np.ndarray
        stream_f = ((np.abs(self_).mean() / np.abs(stream_f).mean()).item() *
                    stream_f)
        return stream_f

    @classmethod
    def _peaks(cls, stream: np.ndarray,
               peak_width: Optional[np.ndarray] = None,
               threshold: float = 5.0e-2, ref: List = None,
               relocate_peak: bool = False) \
            -> List[Union[Tuple[int, float], Tuple[int, float, float]]]:
        """
        Find peaks in NumPy array.

        :param stream: array to search for peaks in
        :param peak_width: peak widths
        :param threshold: lower threshold for peaks
        :param ref: reference values for peaks
        :param relocate_peak: relocate peak to local min or max
        :return: found peaks
        """
        peaks = []
        if peak_width is None:
            peak_width = cls.DEFAULT_PEAK_WIDTH
        pw_pad = peak_width.mean() / 2
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

    def fft_peaks(self, peak_width: Optional[np.ndarray] = None,
                  threshold: float = 5.0e-2, axis: bool = False) \
            -> Union[Tuple[np.ndarray, np.ndarray],
                     List[Tuple[int, float, float]]]:
        """
        Calculate FFT and use peak detection to find FFT peaks.

        :param peak_width: peak widths
        :param threshold: lower threshold for peaks
        :param axis: if `True` return FFT data
        :return: FFT data or FFT peaks
        """
        xf, yf = self.fft()
        if axis:
            return xf, yf
        return self._peaks(yf, peak_width, threshold, relocate_peak=True,
                           ref=xf)

    def peaks(self, peak_width: Optional[np.ndarray] = None,
              threshold: float = 5.0e-2, relocate_peak: bool = True) \
            -> List[Tuple[int, float]]:
        """
        Find peaks in waveform.

        :param peak_width: peak widths
        :param threshold: lower threshold for peaks
        :param relocate_peak: relocate peak to local maxima or minima
        :return: peak data
        """
        results = (self._peaks(self, peak_width, threshold,
                               relocate_peak=relocate_peak) +
                   list(map(lambda v: (v[0], -v[1]),
                            self._peaks(-self, peak_width, threshold))))
        results.sort(key=lambda v: v[0])
        return results

    def zeroes(self, zero_width: int = 20, threshold: float = 0.25) \
            -> List[int]:
        """
        Find zeros in WavStream.

        :param zero_width: zero searching width
        :param threshold: zero threshold
        :return: list of zeros
        """
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

    @property
    def duration(self) -> float:
        """
        Return waveform duration..

        :return: duration
        """
        return len(self) / self.rate

    def __len__(self) -> int:
        """
        Return waveform length.

        :return: length
        """
        return super().__len__()


class Parameter(Individual):
    """Encoder parameter object."""

    def __init__(self, start: Union[int, float],
                 stop: Union[int, float, None]=None,
                 default: Union[int, float, None]=None,
                 scale: Union[int, float, None]=None,
                 log: bool=False,
                 shift: Union[int, float]=0,
                 poly: Union[int, float]=1,
                 forced_type: Optional[type]=None):
        """
        Initialise parameter.

        :param start: start value
        :param stop: stop value
        :param default: default value
        :param scale: scale
        :param log: log scale
        :param shift: shift
        :param poly: polynomial scale
        :param forced_type: force type
        """
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
        """
        Get random valid parameter value.

        :return: random value
        """
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
        """
        Randomise parameter.

        :return: randomised parameter
        """
        return type(self)(self.start, self.stop, self._random(),
                          scale=self.scale, log=self.log,
                          forced_type=self.type)

    def _mutate(self, scale: float = 1) -> float:
        """
        Get random mutated parameter value.

        :param scale: mutation scale
        :return: random value
        """
        c = (self._current + self.shift) ** (1/self.poly)
        s = (self.scale * scale) ** (1/self.poly)
        if self.log:
            v = np.random.lognormal(mean=np.log(c), sigma=np.log(s))
        else:
            v = np.random.normal(loc=c, scale=s)
        return (v - self.shift)**self.poly

    def mutate(self, amount: float = 1, scale: float = 1) -> 'Parameter':
        """
        Get random mutated parameter.

        :param amount: mutation amount
        :param scale: mutation scale
        :return: random mutated parameter
        """
        # Get value within bounds
        if amount > 0:
            v = self._mutate(scale)
            while not self.start <= v <= self.stop:
                v = self._mutate(scale)
        else:
            v = self.current

        # Create a new object
        return type(self)(self.start, self.stop, v, scale=self.scale,
                          log=self.log, forced_type=self.type)

    def cross(self, other: 'Parameter', amount: float = 1) \
            -> 'Parameter':
        """
        Cross-mutate two parameters.

        :param other: parameter to mutate with
        :param amount: crossing strength
        :return: crossed parameter
        """
        c = (self._current + self.shift) ** (1 / self.poly)
        o = (other._current + self.shift) ** (1 / self.poly)
        if self.log:
            c = np.log(c)
            o = np.log(o)
        if c < o:
            c, o = o, c
        s = max((c - o) * amount / 3, self.scale / 4)
        mid = (o + c) / 2
        v = np.random.normal(loc=mid, scale=s)
        while not self.start <= v <= self.stop:
            v = np.random.normal(loc=mid, scale=s)
        if self.log:
            v = np.e ** v

        # Create a new object
        return type(self)(self.start, self.stop, v, scale=self.scale,
                          log=self.log, forced_type=self.type)

    def set(self, value: Union[int, float]) -> Union[int, float]:
        """
        Set parameter value.

        :param value: value to set
        :return: new value
        """
        self._current = value
        return self.current

    def copy(self) -> 'Parameter':
        """
        Get object copy.

        :return: copied object
        """
        return type(self)(self.start, self.stop, self._current,
                          scale=self.scale, log=self.log,
                          forced_type=self.type)

    @property
    def current(self) -> Union[int, float]:
        """
        Get current value of parameter.

        :return: current value
        """
        if self.type == int:
            return rint(self._current)
        return self._current

    @property
    def c(self) -> Union[int, float]:
        """
        Get current value of parameter.

        :return: current value
        """
        return self.current

    def __repr__(self) -> str:
        """
        Get object representation.

        :return: representation
        """
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


class Encoder(Individual, metaclass=ABCMeta):
    """Base encoder and decoder object."""

    symbol_width = Parameter(1, 3, 2)
    symbol_duration = Parameter(0.001, 0.1, 0.005)
    frequency = Parameter(1, 10000, 1000)
    rate = Parameter(8000, 44100, 16000)
    amplitude = Parameter(0.1, 1, 0.9)

    filter_type = Parameter(0, 3, 0)
    filter_window_base = Parameter(1, 500, 20)
    filter_window_scale = Parameter(0.0, 1.0, 0.1)
    filter_shape = Parameter(0, 1.0, 0.5)
    filter_std_base = Parameter(1, 250, 10)
    filter_std_scale = Parameter(0.0, 0.5, 0.05)

    peak_width_start = Parameter(0.0, 1.0, 0.4)
    peak_width_span = Parameter(0.0, 1.0, 0.0)
    peak_threshold = Parameter(0.0, 1.0, 5.0e-3)

    sqe_start = Parameter(0, 0.75, 0.5)
    sqe_start_v = Parameter(0, 1, 0)
    sqe_end = Parameter(0, 0.25, 0.1)
    sqe_end_v = Parameter(0, 1, 0.5)

    def __init__(self):
        """Initialise object."""
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
        """
        Get object representation.

        :return: representation
        """
        return '{}:\n    {}'.format(type(self).__name__, '\n    '.join(
            '{}: {}'.format(p, v.c) for p, v in self.parameters.items()))

    @property
    def parameters(self) -> Dict[str, Parameter]:
        """
        Get all parameters.

        :return: parameters
        """
        return {p: getattr(self, p) for p in dir(self)
                if p != 'parameters' and
                isinstance(getattr(self, p), Parameter)}

    @classmethod
    def random(cls) -> 'Encoder':
        """
        Randomise encoder and decoder.

        :return: randomised encoder and decoder
        """
        new = cls()
        for p, v in new.parameters.items():
            setattr(new, p, v.random())
        return new

    def reinit(self):
        """Perform initialisation again."""
        self.__init__()

    def mutate(self, amount: float = 1/3, scale: float = 1) -> 'Encoder':
        """
        Get random mutated encoder and decoder.

        :param amount: mutation amount
        :param scale: mutation scale
        :return: random mutated encoder and decoder
        """
        new = type(self)()
        for p, v in self.parameters.items():
            if amount > np.random.random():
                print('mutating', p)
                v = v.mutate(scale)
            getattr(new, p).set(v.c)
        return new

    def cross(self, other: 'Encoder', amount: float = 1/3) -> 'Encoder':
        """
        Cross-mutate two encoder and decoder pairs.

        :param other: encoder and decoder to mutate with
        :param amount: crossing strength
        :return: crossed encoder and decoder
        """
        new = type(self)()
        for p, v in self.parameters.items():
            if amount > np.random.random():
                print('crossing', p)
                v = v.cross(getattr(other, p))
            elif np.random.randint(2):
                print('stealing', p)
                v = getattr(other, p)
            getattr(new, p).set(v.c)
        return new

    def filter(self, stream: WavStream) -> WavStream:
        """
        Perform filtering on waveform.

        :param stream: waveform to filter
        :return: filtered waveform
        """
        if self.filter_type.c == 0:
            return stream
        elif self.filter_type.c == 1:
            return stream.filter(self.filter_window, self.filter_shape.c,
                                 self.filter_std)
        elif self.filter_type.c == 2:
            return stream.filter2(self.filter_window, self.filter_window)
        elif self.filter_type.c == 3:
            return stream.filter3(self.filter_window, self.filter_shape.c,
                                  self.filter_std)

    @abstractmethod
    def encode(self, stream: BitStream) -> WavStream:
        """
        Encode binary data.

        :param stream: binary data
        :return: waveform
        """
        pass

    @abstractmethod
    def decode(self, stream: WavStream, filter_stream: bool = True,
               retcert: bool = False) \
            -> Union[BitStream, Tuple[BitStream, List[float]]]:
        """
        Decode waveform.

        :param stream: waveform
        :param filter_stream: apply filter to waveform
        :param retcert: return certainties
        :return: binary data and optionally certainties
        """
        pass

    def _base(self, stream: BitStream, frequency: bool = True) -> np.ndarray:
        """
        Provide base waveform to transform with sine function.

        :param stream: binary data
        :param frequency: base frequency
        :return: base waveform
        """
        if frequency:
            v_max = self.f * 2 * np.pi * len(stream) * self.symbol_len / self.r
        else:
            v_max = 2 * np.pi * len(stream) * self.symbol_len / self.r
        return np.linspace(0, v_max, len(stream) * self.symbol_len,
                           endpoint=False)


class FeatureASK(Encoder):
    """Peak-based ASK encoder and decoder."""

    high_amplitude = Parameter(0.0, 1.0, 0.9)
    low_amplitude = Parameter(0.0, 0.9, 0.1)

    d_high_amplitude = Parameter(0.0, 1.0, 0.9)
    d_low_amplitude = Parameter(0.0, 0.9, 0.1)

    def __init__(self):
        """Initialise object."""
        super().__init__()
        self.low_amp = self.low_amplitude.c * self.high_amplitude.c
        self.step_amp = ((self.high_amplitude.c * (1 - self.low_amplitude.c)) /
                         (self.symbol_size - 1))
        self.d_low_amp = self.d_low_amplitude.c * self.d_high_amplitude.c
        self.d_step_amp = ((self.d_high_amplitude.c *
                            (1 - self.d_low_amplitude.c)) /
                           (self.symbol_size - 1))

    def encode(self, stream: BitStream) -> WavStream:
        """
        Encode binary data.

        :param stream: binary data
        :return: waveform
        """
        stream = stream.assymbolwidth(self.symbol_width.c)

        base = self._base(stream)

        reshape = (stream * self.step_amp) + self.low_amp
        print('Reshape:', reshape.round(2))
        return WavStream(np.sin(base) * reshape.repeat(self.symbol_len),
                         self.r, self.symbol_len)

    def decode(self, stream: WavStream, filter_stream: bool = True,
               retcert: bool = False) \
            -> Union[BitStream, Tuple[BitStream, List[float]]]:
        """
        Decode waveform.

        :param stream: waveform
        :param filter_stream: apply filter to waveform
        :param retcert: return certainties
        :return: binary data and optionally certainties
        """
        symbol_len = rint(stream.rate * self.symbol_duration.c)
        levels, _ = np.linspace(self.d_low_amp, self.d_high_amplitude.c,
                                self.symbol_size, retstep=True)

        if filter_stream:
            stream = self.filter(WavStream(stream, stream.rate, symbol_len))

        retval = []
        certainties = []
        for symbol in stream.symbols():
            peaks = np.array(symbol.peaks(self.peak_range,
                                          self.peak_threshold.c))

            if len(peaks) == 0:
                retval.append(0)
                certainties.append(infs(self.symbol_size))
                print('> no peaks')
                continue

            peak = trim_mean(np.abs(peaks[:, 1]))
            values = np.square(levels - peak)
            value = values.argmin()
            retval.append(value)
            certainties.append(values)

        retval = BitStream(retval, symbolwidth=self.symbol_width.c)
        if retcert:
            return retval, certainties
        return retval


class FeatureIntegralASK(FeatureASK):
    """Sum-based ASK encoder and decoder."""

    def decode(self, stream: WavStream, filter_stream: bool = True,
               retcert: bool = False) \
            -> Union[BitStream, Tuple[BitStream, List[float]]]:
        """
        Decode waveform.

        :param stream: waveform
        :param filter_stream: apply filter to waveform
        :param retcert: return certainties
        :return: binary data and optionally certainties
        """
        symbol_len = rint(stream.rate * self.symbol_duration.c)
        sums = self._sine_sums()
        print('Sums:', sums)

        if filter_stream:
            stream = self.filter(WavStream(stream, stream.rate, symbol_len))

        retval = []
        certainties = []
        for symbol in stream.symbols():
            symbol = symbol  # type: np.ndarray
            s_value = lin_trim_mean(np.abs(symbol),
                                    start=self.sqe_start.c,
                                    start_v=self.sqe_start_v.c,
                                    end=self.sqe_end.c,
                                    end_v=self.sqe_end_v.c)
            distance = np.square(sums - s_value)
            value = distance.argmin()
            certainty = distance
            print('>', value, round(s_value, 2), certainty)
            retval.append(value)
            certainties.append(distance)

        retval = BitStream(retval, symbolwidth=self.symbol_width.c)
        if retcert:
            return retval, certainties
        return retval

    def _sine_sums(self, sample_n: int = 100) -> np.ndarray:
        """
        Calculate symbol sums for comparison.

        :param sample_n: number of symbols to average over
        :return: sums for comparison
        """
        retval = []
        symbol_max = self.f * 2 * np.pi * self.symbol_len / self.r
        base_symbol = np.linspace(0, symbol_max * sample_n,
                                  self.symbol_len * sample_n)
        for n in range(self.symbol_size):
            retval.append(np.abs(np.sin(base_symbol) * (n * self.d_step_amp +
                          self.d_low_amp)).sum().item() / sample_n)
        return np.array(retval) / self.symbol_len


class FeaturePSK(Encoder):
    """Peak-based PSK encoder and decoder."""

    def encode(self, stream: BitStream) -> WavStream:
        """
        Encode binary data.

        :param stream: binary data
        :return: waveform
        """
        stream = stream.assymbolwidth(self.symbol_width.c)

        base = self._base(stream)

        shifts = (stream * 2 * np.pi / self.symbol_size)
        print('Shifts:', shifts.round(2))
        return WavStream(np.sin(base - shifts.repeat(self.symbol_len)) *
                         self.amplitude.c, self.r, self.symbol_len)

    def decode(self, stream: WavStream, filter_stream: bool = True,
               retcert: bool = False) \
            -> Union[BitStream, Tuple[BitStream, List[float]]]:
        """
        Decode waveform.

        :param stream: waveform
        :param filter_stream: apply filter to waveform
        :param retcert: return certainties
        :return: binary data and optionally certainties
        """
        λ = stream.rate / self.f
        symbol_len = rint(stream.rate * self.symbol_duration.c)

        if filter_stream:
            stream = self.filter(WavStream(stream, stream.rate, symbol_len))

        retval = []
        certainties = []
        for k, s in enumerate(stream.symbols()):
            shift = λ - (k * symbol_len) % λ

            peaks = np.array(s.peaks(self.peak_range,
                                     self.peak_threshold.c))

            if len(peaks) == 0:
                retval.append(0)
                certainties.append(infs(self.symbol_size))
                print('> no peaks')
                continue

            # Drop first or last space peaks
            if peaks[0, 1] == 0:
                del peaks[0, 1]
            if peaks[-1, 1] == symbol_len - 1:
                del peaks[-1]

            if len(peaks) == 0:
                retval.append(0)
                certainties.append(infs(self.symbol_size))
                print('> no peaks')
                continue

            positives = peaks[:, 0][peaks[:, 1] > 0] - shift
            negatives = peaks[:, 0][peaks[:, 1] < 0] - shift

            values = sq_cyclic_align_error(positives, negatives, λ,
                                           self.symbol_size,
                                           start=self.sqe_start.c,
                                           start_v=self.sqe_start_v.c,
                                           end=self.sqe_end.c,
                                           end_v=self.sqe_end_v.c)
            value = values.argmin()

            retval.append(value)
            certainties.append(values)
            print('>', value, values.round(2))

        retval = BitStream(retval, symbolwidth=self.symbol_width.c)
        if retcert:
            return retval, certainties
        return retval


class FeatureFSK(Encoder):
    """FFT-based FSK encoder and decoder."""

    frequency_dev = Parameter(0.01, 1.0, 0.25)

    peak_threshold = Parameter(0.0, 1.0, 0.15)

    def __init__(self):
        """Initialise object."""
        super().__init__()
        self.f_low = self.f * (1 - self.frequency_dev.c)
        self.f_high = self.f * (1 + self.frequency_dev.c)
        self.f_step = (self.f_high - self.f_low) / (self.symbol_size - 1)

    def encode(self, stream: BitStream) -> WavStream:
        """
        Encode binary data.

        :param stream: binary data
        :return: waveform
        """
        stream = stream.assymbolwidth(self.symbol_width.c)
        f_map = (stream * self.f_step) + self.f_low
        print('Frequency map:', f_map.round(2))

        base = []
        symbol_base, _ = np.linspace(0, 2 * np.pi * self.symbol_len / self.r,
                                     self.symbol_len, endpoint=False,
                                     retstep=True)
        shift = 0
        for f in f_map:
            λ = self.r / f
            base.append(symbol_base * f + shift * 2 * np.pi)
            shift += (self.symbol_len % λ) / λ

        base = np.concatenate(base)

        return WavStream(np.sin(base) * self.amplitude.c, self.r,
                         self.symbol_len)

    def decode(self, stream: WavStream, filter_stream: bool = True,
               retcert: bool = False) \
            -> Union[BitStream, Tuple[BitStream, List[float]]]:
        """
        Decode waveform.

        :param stream: waveform
        :param filter_stream: apply filter to waveform
        :param retcert: return certainties
        :return: binary data and optionally certainties
        """
        symbol_len = rint(stream.rate * self.symbol_duration.c)
        levels, _ = np.linspace(self.f_low, self.f_high, self.symbol_size,
                                retstep=True)
        if filter_stream:
            stream = self.filter(WavStream(stream, stream.rate, symbol_len))

        retval = []
        certainties = []
        for symbol in stream.symbols():
            peaks = np.array(symbol.fft_peaks(self.peak_range,
                                              self.peak_threshold.c))

            if len(peaks) == 0:
                retval.append(0)
                certainties.append(infs(self.symbol_size))
                print('> no peaks')
                continue

            peak = np.average(peaks[:, 2], weights=peaks[:, 1])
            values = np.square(levels - peak)
            value = values.argmin()
            print('>', value, peak.round(2))
            retval.append(value)
            certainties.append(values)

        retval = BitStream(retval, symbolwidth=self.symbol_width.c)
        if retcert:
            return retval, certainties
        return retval


class FeatureFSK2(FeatureFSK):
    """Peak-based FSK encoder and decoder."""

    def decode(self, stream: WavStream, filter_stream: bool = True,
               retcert: bool = False) \
            -> Union[BitStream, Tuple[BitStream, List[float]]]:
        """
        Decode waveform.

        :param stream: waveform
        :param filter_stream: apply filter to waveform
        :param retcert: return certainties
        :return: binary data and optionally certainties
        """
        symbol_len = rint(stream.rate * self.symbol_duration.c)
        levels, _ = np.linspace(self.f_low, self.f_high, self.symbol_size,
                                retstep=True)
        peak_dist_ref = np.array(self.r) / levels / 2

        if filter_stream:
            stream = self.filter(WavStream(stream, stream.rate, symbol_len))

        retval = []
        certainties = []
        for symbol in stream.symbols():
            peaks = np.array(symbol.peaks(self.peak_range,
                                          self.peak_threshold.c))

            if len(peaks) == 0:
                retval.append(0)
                certainties.append(infs(self.symbol_size))
                print('> no peaks')
                continue

            peaks_dist = peaks[:, 0][1:] - peaks[:, 0][:-1]  # type: np.ndarray

            if len(peaks_dist) == 0:
                retval.append(0)
                certainties.append(infs(self.symbol_size))
                print('> no peaks')
                continue

            peaks_result = np.array([np.mod(d, peak_dist_ref)
                                     for d in peaks_dist])
            peaks_result2 = np.minimum(peak_dist_ref - peaks_result,
                                       peaks_result)
            values = peaks_result2.sum(axis=0)
            value = values.argmin()
            print('>', value, values.round(2))
            retval.append(value)
            certainties.append(values)

        retval = BitStream(retval, symbolwidth=self.symbol_width.c)
        if retcert:
            return retval, certainties
        return retval


class FeatureQAM(Encoder):
    """Peak-based QAM encoder and decoder."""

    symbol_shifts = Parameter(4)
    symbol_levels = Parameter(2)
    symbol_width = Parameter(3)

    high_amplitude = Parameter(0.0, 1.0, 0.9)
    low_amplitude = Parameter(0.0, 0.9, 0.5)

    d_high_amplitude = Parameter(0.0, 1.0, 0.8)
    d_low_amplitude = Parameter(0.0, 0.9, 0.5)

    d_symbol_shifts_scale = Parameter(4)
    d_comparison_type = Parameter(0, 1, 0)

    def __init__(self):
        """Initialise object."""
        super().__init__()
        # Symbol param check
        width = np.log2(self.symbol_shifts.c * self.symbol_levels.c)
        if width != self.symbol_width.c:
            raise ValueError('incorrect parameter settings')

        # ASK Setup
        self.low_amp = self.low_amplitude.c * self.high_amplitude.c
        self.step_amp = ((self.high_amplitude.c * (1 - self.low_amplitude.c)) /
                         (self.symbol_size - 1))
        self.d_low_amp = self.d_low_amplitude.c * self.d_high_amplitude.c
        self.d_step_amp = ((self.d_high_amplitude.c *
                            (1 - self.d_low_amplitude.c)) /
                           (self.symbol_size - 1))

        shifts, step = np.linspace(0, 2 * np.pi, self.symbol_shifts.c,
                                   endpoint=False, retstep=True)
        self.shift_step = step / 2
        shifts_alt = shifts + self.shift_step

        levels = np.linspace(self.low_amp, self.high_amplitude.c,
                             self.symbol_levels.c)
        levels_ = np.linspace(self.d_low_amp, self.d_high_amplitude.c,
                              self.symbol_levels.c)

        patterns = [[l, theta] for theta in shifts for l in levels[::2]]
        patterns += [[l, theta] for theta in shifts_alt for l in levels[1::2]]

        patterns_ = [[l, theta] for theta in shifts for l in levels_[::2]]
        patterns_ += [[l, theta] for theta in shifts_alt
                      for l in levels_[1::2]]
        patterns_ += patterns_[:self.symbol_levels.c // 2]

        self.polar = np.array(patterns)
        self.polar_ = np.array(patterns_)
        self.polar_[-(self.symbol_levels.c // 2):, 1] = 2 * np.pi

        self.cartesian = p2c(self.polar)
        self.cartesian_ = p2c(self.polar_[:self.symbol_size])

        print('Polar:\n{}'.format(self.polar.round(2)))
        print('Cartesian:\n{}'.format(self.cartesian.round(2)))

    def encode(self, stream: BitStream) -> WavStream:
        """
        Encode binary data.

        :param stream: binary data
        :return: waveform
        """
        stream = stream.assymbolwidth(self.symbol_width.c)

        base = self._base(stream)

        qam_map = self.polar[stream]
        print('Shifts:', qam_map[:, 1].round(2))
        print('Reshape:', qam_map[:, 0].round(2))
        return WavStream(np.sin(base - qam_map[:, 1].repeat(self.symbol_len)) *
                         qam_map[:, 0].repeat(self.symbol_len), self.r,
                         self.symbol_len)

    def decode(self, stream: WavStream, filter_stream: bool = True,
               retcert: bool = False) \
            -> Union[BitStream, Tuple[BitStream, List[float]]]:
        """
        Decode waveform.

        :param stream: waveform
        :param filter_stream: apply filter to waveform
        :param retcert: return certainties
        :return: binary data and optionally certainties
        """
        λ = stream.rate / self.f
        symbol_len = rint(stream.rate * self.symbol_duration.c)

        if filter_stream:
            stream = self.filter(WavStream(stream, stream.rate, symbol_len))

        retval = []
        certainties = []
        for k, symbol in enumerate(stream.symbols()):
            shift = λ - (k * symbol_len) % λ
            peaks = np.array(symbol.peaks(self.peak_range,
                                          self.peak_threshold.c))

            if len(peaks) == 0:
                retval.append(0)
                certainties.append(infs(self.symbol_size))
                print('> no peaks')
                continue

            amp = trim_mean(np.abs(peaks[:, 1]))

            # Drop first or last space peaks
            if peaks[0, 1] == 0:
                del peaks[0, 1]
            if peaks[-1, 1] == symbol_len - 1:
                del peaks[-1]

            if len(peaks) == 0:
                retval.append(0)
                certainties.append(infs(self.symbol_size))
                print('> no peaks')
                continue

            positives = peaks[:, 0][peaks[:, 1] > 0] - shift
            negatives = peaks[:, 0][peaks[:, 1] < 0] - shift

            n_shifts = self.symbol_shifts.c * 2 * self.d_symbol_shifts_scale.c
            shifts = sq_cyclic_align_error(positives, negatives, λ, n_shifts,
                                           start=self.sqe_start.c,
                                           start_v=self.sqe_start_v.c,
                                           end=self.sqe_end.c,
                                           end_v=self.sqe_end_v.c)
            shift = (shifts.argmin() * self.shift_step /
                     self.d_symbol_shifts_scale.c)

            print(amp, shift / (2 * np.pi))

            polar = np.array([amp, shift])
            cartesian = p2c(polar)

            if self.d_comparison_type.c == 0:
                temp = np.square(self.cartesian_ - cartesian)
                values = temp[:, 0] + temp[:, 1]
                value = values.argmin()
            else:
                weights = np.array([self.d_step_amp, self.shift_step])
                temp = np.square((self.polar_ - polar) / weights)
                values = temp[:, 0] + temp[:, 1]
                even_shells = self.symbol_levels.c // 2
                lim_cases = np.min([values[:even_shells],
                                    values[-even_shells:]], axis=0)
                other_cases = values[even_shells:-even_shells]
                values = np.concatenate((lim_cases, other_cases))
                value = values.argmin()
            retval.append(value)
            certainties.append(values)
            print('>', k, value, polar)
            print('Peaks:', peaks[:, 0])

        retval = BitStream(retval, symbolwidth=self.symbol_width.c)
        if retcert:
            return retval, certainties
        return retval


class RMSEncoder(Encoder, metaclass=ABCMeta):
    """RMSE encoder and decoder."""

    def _samples(self, stream_len: int) -> np.ndarray:
        """
        Create samples to compare to.

        :param stream_len: binary data length
        :return: comparison samples
        """
        retval = []
        for v in range(self.symbol_size):
            stream = BitStream([v] * stream_len,
                               symbolwidth=self.symbol_width.c)
            retval.append([s for s in self._encode(stream).symbols()])
        return np.swapaxes(np.array(retval), 0, 1)

    def _encode(self, stream: BitStream) -> WavStream:
        """
        Encode binary data for comparison.

        :param stream: binary data
        :return: comparison waveform
        """
        return self.encode(stream)

    def decode(self, stream: WavStream, filter_stream: bool = True,
               retcert: bool = False) \
            -> Union[BitStream, Tuple[BitStream, List[float]]]:
        """
        Decode waveform.

        :param stream: waveform
        :param filter_stream: apply filter to waveform
        :param retcert: return certainties
        :return: binary data and optionally certainties
        """
        symbol_len = rint(stream.rate * self.symbol_duration.c)

        if filter_stream:
            stream = self.filter(WavStream(stream, stream.rate, symbol_len))

        symbols = [s for s in stream.symbols()]
        samples = self._samples(len(symbols))

        retval = []
        certainties = []
        for i, symbol in enumerate(symbols):
            values = lin_trim_error(samples[i], symbol,
                                    start=self.sqe_start.c,
                                    start_v=self.sqe_start_v.c,
                                    end=self.sqe_end.c,
                                    end_v=self.sqe_end_v.c)
            value = values.argmin()
            print('>', value, values)
            retval.append(value)
            certainties.append(values)

        retval = BitStream(retval, symbolwidth=self.symbol_width.c)
        if retcert:
            return retval, certainties
        return retval


class RMSASK(RMSEncoder, FeatureASK):
    """RMSE ASK encoder and decoder."""

    def _encode(self, stream: BitStream) -> WavStream:
        """
        Encode binary data for comparison.

        :param stream: binary data
        :return: comparison waveform
        """
        stream = stream.assymbolwidth(self.symbol_width.c)

        base = self._base(stream)

        reshape = (stream * self.d_step_amp) + self.d_low_amp
        print('Reshape:', reshape.round(2))
        return WavStream(np.sin(base) * reshape.repeat(self.symbol_len),
                         self.r, self.symbol_len)


class RMSPSK(RMSEncoder, FeaturePSK):
    """RMSE PSK encoder and decoder."""

    pass


class RMSQAM(RMSEncoder, FeatureQAM):
    """RMSE QAM encoder and decoder."""

    def _encode(self, stream: BitStream) -> WavStream:
        """
        Encode binary data for comparison.

        :param stream: binary data
        :return: comparison waveform
        """
        stream = stream.assymbolwidth(self.symbol_width.c)

        base = self._base(stream)

        qam_map = self.polar_[stream]
        print('Shifts:', qam_map[:, 1].round(2))
        print('Reshape:', qam_map[:, 0].round(2))
        return WavStream(np.sin(base - qam_map[:, 1].repeat(self.symbol_len)) *
                         qam_map[:, 0].repeat(self.symbol_len), self.r,
                         self.symbol_len)
