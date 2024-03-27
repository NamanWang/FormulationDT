# -*- coding: utf-8 -*-
# %%
from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd

df = pd.read_csv("des.csv")

def calc_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        descriptors = Descriptors.descList
        desc_dict = {}
        for name, func in descriptors:
            desc_dict[name] = func(mol)
        return list(desc_dict.values())
    else:
        return []

desc_list = [calc_descriptors(row[0]) for row in df.values]

desc_df = pd.DataFrame(desc_list, columns=list(Chem.Descriptors._descList))

desc_df.to_csv("output.csv", index=False)
# %%



