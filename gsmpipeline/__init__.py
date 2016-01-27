import gsmpipeline.util
import os


ffmpeg = gsmpipeline.util.ffmpeg('ffmpeg')


class Pipeline:

    def __init__(self, temp_dir: str):
        self.temp_dir = temp_dir

    def apply_amr(self, in_file: str, out_file: str, sample_rate: int):
        temp = '{}{}{}.amr'.format(self.temp_dir, os.pathsep,
                                   os.path.basename(in_file))
        ffmpeg(in_file, temp, sample_rate)
        ffmpeg(temp, out_file, sample_rate)
