import torch
import os
from torch.utils.data import Dataset
import librosa
import fnmatch
import numpy as np


class VibDataset(Dataset):
    def __init__(self, data_dir, sr, dimension=4096):
        self.data_dir = data_dir
        self.sr = sr
        self.dim = dimension

        # 获取wav文件列表
        self.file_list = []
        for root, dirname, filenames in os.walk(data_dir):
            for filename in fnmatch.filter(filenames, "*.wav"):
                self.file_list.append(os.path.join(root, filename))

    def __getitem__(self, item):
        filename = self.file_list[item]
        wb_wav, _ = librosa.load(filename, sr=self.sr, mono=False)
        wb_wav = np.expand_dims(wb_wav, axis=0)
        path, file_name = os.path.split(filename)
        label = int(path[-1])

        # librosa.display.waveshow(wb_wav[0])
        # plt.show()
        # librosa.display.waveshow(wb_wav[1])
        # plt.show()

        return wb_wav, label, filename

    def __len__(self):

        return len(self.file_list)


