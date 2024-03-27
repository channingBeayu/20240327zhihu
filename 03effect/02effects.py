import pandas as pd

data_senti = []
with open('data/sentis.txt', 'r') as f:
    for line in f.readlines():
        data_senti.append(line.split())
data_hc = []
with open('data/hcs.txt', 'r') as f:
    for line in f.readlines():
        data_hc.append(line.split())
data_senti = pd.DataFrame(data_senti, dtype=float)
data_hc = pd.DataFrame(data_hc, dtype=float)
dataset = pd.concat([data_senti, data_hc], ignore_index=True, axis=1)

Y = dataset[0].values



# PSM倾向性得分匹配
from causalinference import CausalModel

for i in range(1, data_senti.shape[1]):
    X = dataset[i].values
    confounders = dataset.drop(columns=[0, i]).values
    model = CausalModel(Y, X, confounders)
    model.est_via_matching(bias_adj=True)
    print(f'---------topic{i}------------')
    print(model.estimates)

