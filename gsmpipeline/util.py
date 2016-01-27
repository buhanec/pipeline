from typing import Callable, Union
from matplotlib import pyplot as plt
import subprocess


def ffmpeg(ffmpeg_bin: str):
    def ffmpeg_call(in_file: str, out_file: str, sample_rate: int,
                    bit_rate: Union[str, int, None]=None):
        cmd = [ffmpeg_bin, '-i', in_file, '-ar', str(sample_rate), '-y']
        if bit_rate:
            cmd += ['-b:a', str(sample_rate)]
        cmd.append(out_file)
        subprocess.run(cmd)

    return ffmpeg_call


def quickplot(xf, yf):
    fig, ax = plt.subplots()
    ax.plot(xf, yf)
    plt.show()
