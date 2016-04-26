"""Project utility module."""

from typing import Union, Tuple, List, Optional, Iterable
from numbers import Number
import numpy as np


def rint(a: Number) -> int:
    """
    Round a number and return as int.

    :param a: number
    :return: rounded int
    """
    return np.round(a).astype(int)


def val_split(a: Iterable, partitions: int, range_max: int, range_min: int = 0,
              size: bool = True) -> List[np.ndarray]:
    """
    Split `a` into partitions. Unused in final code.

    :param a: source to split
    :param partitions: number of partitions
    :param range_max: range max
    :param range_min: range min
    :param size: if `True` use `range_max` as size rather than range
    :return: partitioned array
    """
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


def c2p(v: Union[Tuple, List, np.ndarray]) -> np.ndarray:
    """
    Convert Cartesian to polar coordinates.

    :param v: single coordinate or list of coordinates
    :return: converted coordinate or coordinates
    """
    if not isinstance(v, np.ndarray):
        v = np.array(v)
    if len(v.shape) == 1:
        v = np.array([v])
    return np.array([np.sqrt(np.square(v).sum(axis=1)),
                     np.arctan2(v[:, 1], v[:, 0]) % (2*np.pi)]).T


def p2c(v: Union[Tuple, List, np.ndarray]) -> np.ndarray:
    """
    Convert polar to Cartesian coordinates.

    :param v: single coordinate or list of coordinates
    :return: converted coordinate or coordinates
    """
    if not isinstance(v, np.ndarray):
        v = np.array(v)
    if len(v.shape) == 1:
        v = np.array([v])
    return (np.array([np.cos(v[:, 1]), np.sin(v[:, 1])]) * v[:, 0]).T


def sq_error(a: Union[Number, np.ndarray], b: Union[Number, np.ndarray]):
    """
    Get squared error.

    :param a: value or values
    :param b: value or values
    :return: error or errors
    """
    return np.square(a - b).sum().item()


def min_error(a: np.ndarray, b: np.ndarray, shift: int,
              l: Optional[int] = None, w: int = 0) -> int:
    """
    Find shift between two arrays.

    Best shift is based on the smallest RMS error when comparing shifts
    between `a` and `b`.

    :param a: array to compare
    :param b: array to compare
    :param shift: maximum shift
    :param l: length to compare
    :param w: starting offset of comparison
    :return: best shift found
    """
    if l is None:
        l = len(a)
    shifts = np.arange(-shift, shift)
    errors = np.array([sq_error(a[w:l-w], b[w+n:l-w+n]) for n in shifts])
    if not len(errors):
        return 0
    return shifts[errors.argmin()]


def lin_trim_mean(a: np.ndarray, start: float = 0.5, end: float = 0.1,
                  start_v: float = 0, end_v: float = 0.5) -> float:
    """
    Calculate mean of array.

    Scaling used to apply lower weights the start and end of the array.
    Weights are calculated linearly from the edges.

    :param a: input
    :param start: amount of inputs to scale at the start
    :param end: amount of inputs to scale at the end
    :param start_v: starting scale value at the start
    :param end_v: starting scale value at the end
    :return: trimmed weighted mean
    """
    start_w = np.linspace(start_v, 1, start * len(a), endpoint=False)
    end_w = np.linspace(end_v, 1, end * len(a), endpoint=False)[::-1]
    mid_w = np.ones(len(a) - len(start_w) - len(end_w))
    weights = np.concatenate((start_w, mid_w, end_w))
    return ((a * weights).sum() / weights.sum()).item()


def lin_trim_error(a: np.ndarray, b: np.ndarray, start: float = 0.5,
                   end: float = 0.1, start_v: float = 0,
                   end_v: float = 0.5) -> np.ndarray:
    """
    Calculate squared error between two arrays.

    Scaling used to apply lower weights the start and end of the array.
    Weights are calculated linearly from the edges.

    :param a: reference array
    :param b: second array
    :param start: amount of inputs to scale at the start
    :param end: amount of inputs to scale at the end
    :param start_v: starting scale value at the start
    :param end_v: starting scale value at the end
    :return: squared error
    """
    start_w = np.linspace(start_v, 1, start * len(b), endpoint=False)
    end_w = np.linspace(end_v, 1, end * len(b), endpoint=False)[::-1]
    mid_w = np.ones(len(b) - len(start_w) - len(end_w))
    weights = np.concatenate((start_w, mid_w, end_w))
    return (np.square(a - b) * weights).sum(axis=1)


def sq_cyclic_align_error(positives: Union[List, np.ndarray],
                          negatives: Union[List, np.ndarray],
                          wavelength: Number, lim: int,
                          start: float = 0.5, end: float = 0.1,
                          start_v: float = 0.1, end_v: float = 0.1,
                          start_min: int = 2, end_min: int = 2) -> np.ndarray:
    """
    Determine weights corresponding to all possible shifts.

    Scaling used to apply lower weights the start and end of the array.
    Weights are calculated linearly from the edges.

    :param positives: positive peaks
    :param negatives: negative peaks
    :param wavelength: wavelength of the sine wave
    :param lim: number of possible shifts
    :param start: amount of inputs to scale at the start
    :param end: amount of inputs to scale at the end
    :param start_v: starting scale value at the start
    :param end_v: starting scale value at the end
    :param start_min: minimum number of scaled samples at the start
    :param end_min: minimum number of scaled samples at the end
    :return: weights corresponding to alignment probabilities
    """
    l = len(positives) + len(negatives)
    start_w = np.linspace(start_v, 1, int(start * l) or start_min,
                          endpoint=False)
    end_w = np.linspace(end_v, 1, int(end * l) or end_min,
                        endpoint=False)[::-1]
    l2 = l - len(start_w) - len(end_w)
    if l2 > 0:
        mid_w = np.ones(l2)
    elif l2 == 0:
        mid_w = np.array([])
    else:
        mid_w = np.array(start_w[l2:] + end_w[:-l2]) / 2
        start_w = start_w[:l2]
        end_w = end_w[-l2:]
    weights = np.concatenate((start_w, mid_w, end_w))

    p_k = 0
    n_k = 0

    peaks = []
    while True:
        e = False
        try:
            p = positives[p_k]
        except IndexError:
            p = np.Infinity
            e = True
        try:
            n = negatives[n_k]
        except IndexError:
            n = np.Infinity
            if e:
                break
        if p < n:
            c = p % wavelength / wavelength + 0.75
            p_k += 1
        else:
            c = n % wavelength / wavelength + 0.25
            n_k += 1
        peaks.append(c)
    peaks = (np.array(peaks) % 1 * lim) % lim

    lim_case = [np.minimum(np.square(peaks - lim), np.square(peaks))]
    m = [np.minimum(np.square(n - peaks), np.square(n + peaks))
         for n in np.arange(1, lim)]
    cases = (np.array(lim_case + m) * weights).sum(axis=1) / weights.sum()

    return np.array(cases)


def ease(x: Union[Number, np.ndarray], a: Number = 2) \
        -> Union[Number, np.ndarray]:
    """
    Function used to create easing pattern for synchronisation.

    :param x: input or inputs
    :param a: easing parameter
    :return: value or values for given `x`
    """
    return x ** a / (x ** a + (1 - x) ** a)


def trim_mean(a: np.ndarray, min_num: int = 1, percent: float = 0.2,
              strength: float = 0.2) -> float:
    """
    Simpler version of `lin_trim_mean`.

    Provides option to set minimum number of trimmed samples.

    :param a: array to trim
    :param min_num: minimum number to trim
    :param percent: trim amount
    :param strength: trim strength
    :return: trimmed mean
    """
    if len(a) == 1:
        return a[0]
    num = max(min_num, rint(len(a) * percent))
    weight = len(a) - 2 * num * (1 - strength)
    s = sum(a[:num]) * strength + sum(a[num:-num]) + sum(a[-num:]) * strength
    return s / weight


def add_noise(a: np.ndarray, noise: float) -> np.ndarray:
    """
    Add noise to array.

    :param a: source array
    :param noise: noise strength
    :return: noisy array
    """
    return np.clip(a + np.random.normal(0, noise, len(a)), -1, 1)


def infs(n: int) -> np.ndarray:
    """
    Create NumPy array of `inf`.

    :param n: number of elements
    :return: `n` `inf` element array
    """
    return np.ones(n) * np.inf


def smooth5(size: int) -> int:
    """
    Find smallest 5-smooth number equal to or larger than size.

    Based on SciPy implementation.

    :param size: starting value
    :return: 5-smooth number
    """
    if size < 6:
        return size
    if not size % 2:
        return size

    new = np.inf
    power5 = 1
    while power5 < size:
        power35 = power5
        while power35 < size:
            power2 = 2 ** ((-int(-size // power35) - 1).bit_length())
            n = power2 * power35
            if n == size:
                return new
            elif n < new:
                new = n
            power35 *= 3
            if power35 == size:
                return new
        if power35 < new:
            new = power35
        power5 *= 5
        if power5 == size:
            return new
    if power5 < new:
        new = power5
    return new
