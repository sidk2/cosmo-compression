
from typing import Optional, Callable, Tuple, List
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from torchvision.datasets import VisionDataset
from torchio.transforms import Resize
from torch.utils.data import Dataset
import torch

logging.basicConfig(level=logging.WARNING)

# map_type: {resolution: , std: mean:, min: }

NORM_DICT = {
    'T': {256: {'mean': 4.63, 'std': 1.1, 'min': 224.6,}},
    'Mcdm': {256: {'mean': 11.00, 'std': 0.5, 'min': 2655544000.}},
    'HI': {256: {'mean': 4.5, 'std': 1.60, 'min': 0.,}},
    'Mstar': {256: {'mean': -5.7, 'std': 2.16, 'min': 0.}},
    'P': {256: {'mean': 3.8, 'std': 1.5, 'min': 0.3,}},
    'Z': {256: {'mean': -4.5, 'std': 1.3,'min': 0.,}},
    'Mtot': {256: {'mean': 11.12, 'std': 0.48, 'min': 4665044500.0}},
}


class CAMELS(VisionDataset):
    def __init__(
        self,
        root: str = "CAMELS",
        redshift: float = 0.0,
        transform: Optional[Callable] = None,
        parameters=['Omega_m', 'sigma_8',], 
        suite='Astrid', # 'IllustrisTNG', 'SIMBA', 'EAGLE', 'Magneticum'],
        resolution: int = 256,
        original_resolution: int = 256,
        idx_list: Optional[List[int]] = None,
        dataset: str = 'LH',
        map_type:str='Mcdm',
    ):
        super().__init__(
            root,
            transform=transform,
        )
        self.root = Path(self.root)
        self.redshift = redshift
        self.idx_list = idx_list
        if resolution != original_resolution:
            self.resize = Resize((resolution, resolution,))
        else:
            self.resize = None
        self.resolution = resolution
        self.suite = suite
        self.dataset = dataset
        self.map_type = map_type
        self._load_images(suite=suite, dataset=dataset,map_type=map_type,)
        self._load_parameters(suite=suite, dataset=dataset, parameters=parameters)
 
    def __len__(
        self,
    ):
        return len(self.x)

    def _load_images(self, suite, dataset,map_type='Mtot'):
        data = np.load(
            self.root / f'Maps_{map_type}_{suite}_{dataset}_z={self.redshift:.2f}.npy'
        )
        if self.idx_list is not None:
            if max(self.idx_list) >= len(data):
                idx_list = [idx for idx in self.idx_list if idx < len(data)]
                logging.warning(f'Index list contains indices out of bounds for dataset {suite}. Truncating idx_list to fit the data size ({len(idx_list)}).')
            else:
                idx_list = self.idx_list
            data = data[idx_list]
        if NORM_DICT[map_type][self.resolution]['min'] == 0.:
            data += 1.e-6 #- NORM_DICT[map_type][self.resolution]['min']
        elif NORM_DICT[map_type][self.resolution]['min'] < 0.:
            data -= 1.05*NORM_DICT[map_type][self.resolution]['min']

        self.y = np.log10(data)[:,None]



    def _load_parameters(
        self, 
        suite, 
        dataset,
        parameters= ['Omega_m', 'sigma_8', 'A_SN1', 'A_SN2', 'A_AGN1', 'A_AGN2',],
    ):
        column_names = ['Omega_m', 'sigma_8', 'A_SN1', 'A_SN2', 'A_AGN1', 'A_AGN2','Omega_b']
        params = pd.read_csv(
            self.root / f'params_{dataset}_{suite}.txt',
            sep=" ",
            names=column_names,
            header=None,
        )[parameters]
        params = params.loc[params.index.repeat(15)].reset_index(drop=True)
        if self.idx_list is not None:
            if max(self.idx_list) >= len(params):
                logging.warning(f'Index list contains indices out of bounds for dataset {suite}. Truncating idx_list to fit the data size.')
                idx_list = [idx for idx in self.idx_list if idx < len(params)]
            else:
                idx_list = self.idx_list
            params = params.iloc[idx_list].values
        else:
            params = params.values
 
        self.x =params.astype(np.float32)

    def __getitem__(
        self,
        index: int,
    ) -> Tuple[np.array, np.array, np.array]:
        x, y = self.x[index], self.y[index]
        if self.transform is not None:
            y =  self.transform(y)
        if self.resize is not None:
            y = self.resize(y)
        # Standarize
        y = (y - NORM_DICT[self.map_type][self.resolution]['mean']) / NORM_DICT[self.map_type][self.resolution]['std']
        return x, y