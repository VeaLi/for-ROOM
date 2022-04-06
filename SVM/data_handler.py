import os

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

import pandas as pd
import numpy as np

from functools import lru_cache

np.random.seed(7)

class LoadDBForSklearn(object):
    """
    Prepares data from BindingDB for Scikit-Learn

    Parameters
    ----------
    thresh : Float or Int
        threshold for data binarization
    resampler : imblearn.over_sampler or under_sampler
        funtion to resample the data if needed
    """
    def __init__(self, thresh=1, resampler=None):
        self.thresh = thresh
        self.filter = filter
        self.resampler = resampler

        self.bdconstants = ['Ki (nM)', 'IC50 (nM)', 'Kd (nM)', 'EC50 (nM)']

    def certain_to_float(self, S):
        """
        Converts BindingDB records to floats

        Parameters
        ----------
        S : Str
            Any of binding constants {Kd, Ki, IC50, EC50} as string
        Returns
        -------
        F : Float
            S as float
        Example
        ----------
        certain_to_float(">1000")
        >>> 1000.0
        """
        S = str(S)
        S = S.strip()
        S = str(S).replace(">", "").replace("<", "")
        if S.startswith('+') or S.startswith('-'):
            S = S.replace("+", '').replace("-", '')
        F = float(S.strip())
        return F

    def binarize(self, F):
        nM = F
        uM = nM/1000
        is_active = int(uM < self.thresh)
        return is_active

    @lru_cache(200000)
    def make_rdkit_canonical(self, s):
        """
        Makes SMILES RDkit canonincal removes isomeric features.

        Parameters
        ----------
        s : Str
            SMILES as string
        Returns
        -------
        s : Str
            SMILES as string

        Example:
        ----------
        make_rdkit_canonical("O=C(N[C@H]1CC[C@H](CCN2CCN(CC2)c2nsc3ccccc23)CC1)c1cc2ccccc2[nH]1")
        >>>'O=C(NC1CCC(CCN2CCN(c3nsc4ccccc34)CC2)CC1)c1cc2ccccc2[nH]1'
        """
        try:
            ms = Chem.MolFromSmiles(s)
            s = Chem.MolToSmiles(ms, canonical=True, isomericSmiles=False)
        except:
            return np.nan
        return s

    def get_mol(self, s):
        """
        Parameters
        ----------
        s : Str
            SMILES as string
        Returns
        -------
        m : rdkit.Chem.rdchem.Mol
            SMILES as mol structure
        """
        mol = Chem.MolFromSmiles(s)
        return mol

    def get_fp(self, m):
        """
        Parameters
        ----------
        m : rdkit.Chem.rdchem.Mol
            SMILES as mol structure
        Returns
        -------
        arr : array_like
            Morgan fingerprints array
        """
        fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)
        arr = np.zeros((0,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr

    def prepare(self, data):
        for c in self.bdconstants:
            data[c] = data[c].apply(
                lambda x: self.binarize(self.certain_to_float(x)))

        data['target'] = data[self.bdconstants].max(axis=1)
        data = data.dropna(subset=['target'])

        data = data[['Ligand SMILES', 'target']]
        data['smiles'] = data['Ligand SMILES'].apply(self.make_rdkit_canonical)
        data = data.groupby('smiles', as_index=False)['target'].median()
        data['target'] = data['target'].round()
        data = data.dropna()
        data = data.drop_duplicates()
        print("Num of active: ", np.round(data['target'].mean(), 3), "%")

        if self.resampler:
            X_resampled, y_resampled = self.resampler.fit_resample(
                data['smiles'].values.reshape(-1, 1), data['target'])

            data = pd.DataFrame()
            data['smiles'], data['target'] = X_resampled.ravel(), y_resampled

        data = data.sample(frac=1)

        print("Num of active after oversampling: ",
              np.round(data['target'].mean(), 3), "%")

        fpdata = np.vstack(data['smiles'].apply(
            lambda x: self.get_fp(self.get_mol(x))))

        X = fpdata
        y = data['target'].values

        return X, y