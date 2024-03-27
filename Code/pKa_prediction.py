# %%
import requests
import random
import os
import pandas as pd
from rdkit import Chem
import csv

# %%
def predict_pka(smi):
    upload_url=r'http://xundrug.cn:5001/modules/upload0/'
    param={"Smiles" : ("tmg", smi)}
    headers={'token':'O05DriqqQLlry9kmpCwms2IJLC0MuLQ7'}
    response=requests.post(url=upload_url, files=param, headers=headers)
    jsonbool=int(response.headers['ifjson'])
    if jsonbool==1:
        res_json=response.json()
        if res_json['status'] == 200:
            pka_datas = res_json['gen_datas']
            return pka_datas
        else:
            raise RuntimeError("Error for pKa prediction")
    else:
        raise RuntimeError("Error for pKa prediction")
        
if __name__=="__main__":
    smi = "COc(n1)ccc(c12)nccc2NC(=O)C3CCC(CC3)CCc(c4)ccc(c45)OCCO5"
    data_pka = predict_pka(smi)
    print(data_pka)

# %%
import csv

with open('input.csv', 'r') as file:
    reader = csv.reader(file)
    
    smiles = []
    
    for row in reader:
        smiles.append(row[0])
        
print(smiles)

# %%
output=[]
        
for smile in smiles:
    data_pka = predict_pka(smile)
    d = dict(list(data_pka.items())[:2])
    print(d)
    output.append(d)

# %%
header = ['Acid', 'Base']

with open('output.csv', 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=header)
    writer.writeheader()
    
    for d in output:
        row = {}
        for key in header:
            if key in d:
                row[key] = str(d[key])
            else:
                row[key] = ''
        writer.writerow(row)

# %%



