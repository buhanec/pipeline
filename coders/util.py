from typing import Union, Tuple, List, Optional, Dict, Any, Iterable
from numbers import Number
from abc import ABCMeta, abstractmethod
import numpy as np
import scipy
import scipy.signal
import scipy.fftpack
import scipy.stats


def rint(a: Number) -> int:
    return np.round(a).astype(int)


def val_split(a: Iterable, partitions: int, range_max: int, range_min: int = 0,
              size: bool = True) -> List[np.ndarray]:
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


def cyclic_d(values: np.ndarray, lim: int) -> int:
    values %= lim
    lim_case = np.array([np.minimum(np.square(values - lim),
                                    np.square(values))])
    m = np.array([np.minimum(np.square(n - values),
                             np.square(n + values))
                  for n in np.arange(1, lim)])
    return np.concatenate((lim_case, m), axis=0).mean(axis=1).argmin()


def sq_error(a: Union[Number, np.ndarray], b: Union[Number, np.ndarray]):
    return np.square(a - b).sum().item()


def min_error(a: np.ndarray, b: np.ndarray, shift: int,
              l: Optional[int] = None, w: int = 0) -> int:
    if l is None:
        l = len(a)
    shifts = np.arange(-shift, shift)
    errors = np.array([sq_error(a[w:l-w], b[w+n:l-w+n]) for n in shifts])
    if not len(errors):
        return 0
    return shifts[errors.argmin()]


def lin_trim_mean(a: np.ndarray, start: float = 0.5, end: float = 0.1,
                  start_v: float = 0, end_v: float = 0.5) -> float:
    start_w = np.linspace(start_v, 1, start * len(a), endpoint=False)
    end_w = np.linspace(end_v, 1, end * len(a), endpoint=False)[::-1]
    mid_w = np.ones(len(a) - len(start_w) - len(end_w))
    weights = np.concatenate((start_w, mid_w, end_w))
    return ((a * weights).sum() / weights.sum()).item()


def lin_trim_error(ref: np.ndarray, a: np.ndarray, start: float = 0.5,
                   end: float = 0.1, start_v: float = 0, end_v: float = 0.5) \
        -> np.ndarray:
    start_w = np.linspace(start_v, 1, start * len(a), endpoint=False)
    end_w = np.linspace(end_v, 1, end * len(a), endpoint=False)[::-1]
    mid_w = np.ones(len(a) - len(start_w) - len(end_w))
    weights = np.concatenate((start_w, mid_w, end_w))
    return (np.square(ref - a) * weights).sum(axis=1)


def sq_cyclic_align_error(positives: Union[List, np.ndarray],
                          negatives: Union[List, np.ndarray],
                          wavelength: Number, lim: int,
                          start: float = 0.5, end: float = 0.1,
                          start_v: float = 0.1, end_v: float = 0.1,
                          start_min: int = 2, end_min: int = 2) -> np.ndarray:
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
    return x ** a / (x ** a + (1 - x) ** a)


def trim_mean(a, min_num: int = 1, percent: float = 0.2,
              strength: float = 0.2) -> float:
    if len(a) == 1:
        return a[0]
    num = max(min_num, rint(len(a) * percent))
    weight = len(a) - 2 * num * (1 - strength)
    s = sum(a[:num]) * strength + sum(a[num:-num]) + sum(a[-num:]) * strength
    return s / weight
