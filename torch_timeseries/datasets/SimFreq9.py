from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import Any, Callable, Generic, NewType, Optional, Sequence, TypeVar, Union
from torch import Tensor
import torch.utils.data
import os
from torchvision.datasets.utils import download_and_extract_archive, check_integrity
from abc import ABC, abstractmethod

from torch_timeseries.data.scaler import MaxAbsScaler, Scaler, StoreType

from enum import Enum, unique

from torch_timeseries.datasets.dataset import Freq, TimeSeriesDataset, TimeSeriesStaticGraphDataset


class SimFreq9(TimeSeriesDataset):
    """
    This dataset is designed to test models on frequency mutation (a type of distribution drift).
    The fundamental frequencies of the signal change abruptly at the midpoint of the series.
    """
    name: str = 'SimFreq9'
    num_features: int = 9
    sample_rate: int = 1
    length: int = 10000
    freq: str = 't'

    def download(self):
        pass

    def _load(self):
        # Generating date series
        dates = pd.date_range(start='2022-01-01', periods=self.length, freq='t')

        # --- START: FREQUENCY MUTATION LOGIC ---

        # Define two different sets of frequencies for the two halves of the series
        freqs1 = (12, 16, 24, 36, 48., 60, 72, 84, 96)  # Frequencies for the first half
        freqs2 = (10, 20, 30, 40, 50, 70, 80, 90, 100)  # Frequencies for the second half (distinctly different)

        mutation_point = self.length // 2

        # Keep the original time-varying amplitude logic
        freqs_mag = (
        [(0, 1, 2, 4), (1, 3, 5, 6), (3, 4, 6, 8), (1, 2, 4, 5), (1, 3, 5, 6), (1, 3, 5, 6), (1, 3, 5, 6), (1, 3, 5, 6),
         (1, 3, 5, 6)])
        seqs = []
        for i in range(self.num_features):
            seqs_ = []
            seqs_.append(np.linspace(freqs_mag[i][0], freqs_mag[i][1], num=int(self.length * 0.7)))
            seqs_.append(np.linspace(freqs_mag[i][1], freqs_mag[i][2], num=int(self.length * 0.2)))
            seqs_.append(np.linspace(freqs_mag[i][2], freqs_mag[i][3],
                                     num=int(self.length) - int(self.length * 0.7) - int(self.length * 0.2)))
            seqs.append(np.concatenate(seqs_))

        # --- END: FREQUENCY MUTATION LOGIC ---

        data = np.zeros((self.length, self.num_features))

        # Time vectors for the two halves
        t1 = np.arange(0, mutation_point)
        t2 = np.arange(mutation_point, self.length)

        for i in range(0, self.num_features):
            # Generate first half of the data with freqs1
            freq_signals1 = 0
            for j in range(0, i + 1):
                w = 2 * np.pi / freqs1[j]
                freq_signals1 += np.sin(w * t1)

            # Get the corresponding amplitude for the first half
            amplitude1 = seqs[i][:mutation_point]
            data[:mutation_point, i] = amplitude1 * freq_signals1

            # Generate second half of the data with freqs2
            freq_signals2 = 0
            for j in range(0, i + 1):
                w = 2 * np.pi / freqs2[j]
                freq_signals2 += np.sin(w * t2)

            # Get the corresponding amplitude for the second half
            amplitude2 = seqs[i][mutation_point:]
            data[mutation_point:, i] = amplitude2 * freq_signals2

        # Creating DataFrame with specified column names
        self.df = pd.DataFrame(data, columns=[f"data{i}" for i in range(self.num_features)])
        self.df['date'] = dates
        self.dates = pd.DataFrame({'date': dates})
        self.data = self.df.drop('date', axis=1).values
        return self.data