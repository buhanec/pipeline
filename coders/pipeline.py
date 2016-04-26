"""Pipeline."""

from typing import Tuple, Union, Optional
import numpy as np
import scipy.io.wavfile
import fasteners
import os
import subprocess

from . import WavStream, BitStream, Encoder, sync_padding
from .util import rint, min_error, add_noise


def ffmpeg(ffmpeg_bin: str):
    def ffmpeg_call(in_file: str, out_file: str, sample_rate: int,
                    bit_rate: Union[str, int, None]=None,
                    codec: Optional[str]=None):
        if codec is None and out_file.split('.')[-1] == 'wav':
            codec = 'pcm_s16le'
        cmd = [ffmpeg_bin, '-i', in_file, '-ar', str(sample_rate), '-y']
        if bit_rate:
            cmd += ['-b:a', str(sample_rate)]
        if codec:
            cmd += ['-acodec', codec]
        cmd.append(out_file)
        subprocess.run(cmd)

    return ffmpeg_call


class Pipeline:

    def __init__(self, working_dir: str,
                 ffmpeg_path: str = 'ffmpeg', noise_in: float = 0.1,
                 noise_out: float = 0.1, sync_duration: float = 0.05,
                 filter_resync: bool = False, no_lock: bool = True):
        self.working_dir = working_dir
        self.ffmpeg = ffmpeg(ffmpeg_path)
        self.noise_in = noise_in
        self.noise_out = noise_out
        self.sync_duration = sync_duration
        self.filter_resync = filter_resync
        self._id = 1
        self._lock = fasteners.InterProcessLock(self.lock_path)
        self.no_lock = no_lock
        if not self.no_lock:
            while not self._lock.acquire(blocking=False):
                self._id += 1
                self._lock = fasteners.InterProcessLock(self.lock_path)
            print('[{}] Lock acquired: {}'.format(os.getpid(), self._id))

    def __del__(self):
        if not self.no_lock:
            try:
                if self._lock.acquired:
                    self._lock.release()
                    print('[{}] Lock released: {}'.format(os.getpid(),
                                                          self._id))
            except AttributeError:
                pass

    def test(self, coder: Encoder, stream: BitStream) -> Tuple[float, float]:
        # Encode, add sync and noise
        encoded = coder.encode(stream)
        sync = sync_padding(coder, self.sync_duration)
        padded = np.concatenate((sync, encoded))
        padded_noisy = add_noise(padded, self.noise_in)

        # AMR distortions
        scipy.io.wavfile.write(self.in_path, coder.r, padded_noisy)
        self.ffmpeg(self.in_path, self.temp_path, 8000)
        self.ffmpeg(self.temp_path, self.out_path, coder.r)
        converted_rate, raw = scipy.io.wavfile.read(self.out_path)
        converted_symbol_len = rint(converted_rate * coder.symbol_duration.c)
        converted = raw / 32768  # type: np.ndarray
        converted_noisy = add_noise(converted, self.noise_out)

        # Shift and sync
        shift = rint(0.005 * coder.rate.c)
        shifted = WavStream(converted_noisy[shift:len(padded) + shift],
                            converted_rate, converted_symbol_len)

        sync_padding_ = rint(self.sync_duration * converted_rate)
        sync_clip = sync_padding_ // 10
        sync_start = shift + sync_padding_
        if self.filter_resync:
            sync_start += min_error(padded, coder.filter(shifted), shift,
                                    sync_padding_, sync_clip)
        slice_ = slice(sync_start, sync_start + len(encoded))
        synced = WavStream(converted_noisy[slice_], converted_rate,
                           converted_symbol_len)

        # Decode distorted waveform
        output, certainties = coder.decode(synced, retcert=True)

        # Determine rate and quality
        check = stream == output.assymbolwidth(1)  # type: np.ndarray
        rate = len(check) / synced.duration
        quality = (check.sum() / len(check))  # type: np.ndarray

        return rate, quality.item()

    @property
    def lock_path(self):
        return os.path.join(self.working_dir, 'lock_{}'.format(self._id))

    @property
    def in_path(self):
        return os.path.join(self.working_dir, '{}_in.wav'.format(self._id))

    @property
    def temp_path(self):
        return os.path.join(self.working_dir, '{}_temp.amr'.format(self._id))

    @property
    def out_path(self):
        return os.path.join(self.working_dir, '{}_out.wav'.format(self._id))
