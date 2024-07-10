import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model import GAT
from utils import *

import time

# device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def get_node_weight(k):
    if k == 6:
        return 3.84
    elif k == 5:
        return 2.2
    elif k == 3:
        return 1.74
    elif k == 1:
        return 1.92
    elif k == 7:
        return 1.4
    else:
        return 1


# hyper-parameters
num_epochs = 100
seed = 2022
dropout = 0.6
alpha = 0.2
nheads = 8
hidden = 8
lr = 0.005 
weight_decay = 5e-4
fix_seed(seed)

# data
adj, features, labels, idx_train, idx_val, idx_test = load_data()
node_weight = torch.ones(labels.shape[0], labels.shape[0]) ## N*N 只有对角线有数值
for i in range(labels.shape[0]):
    label = int(labels[i])
    try:
        # node_weight[i][i] = get_node_weight(label)
        node_weight[:, i] = get_node_weight(label)
    except:
        print()



model = GAT(nfeat=features.size()[1], nhid=hidden,
            nclass=labels.max().item() + 1, dropout=dropout,
            alpha=alpha, nheads=nheads).to(device)

# loss and optimizer
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
criterion = nn.NLLLoss()

def train(epoch):
    t = time.time()
    model.train()
    
    # Forward pass
    outputs = model(features, adj, node_weight)
    loss = criterion(outputs[idx_train], labels[idx_train])

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    acc_train = accuracy(outputs[idx_train], labels[idx_train])
    acc_val = accuracy(outputs[idx_val], labels[idx_val])

    print(
        'Epoch [{}/{}], Loss: {:.4f}, Acc: {:.2f}, Val-Acc: {:.2f}, time: {:.4f}s'.
        format(
            epoch + 1, num_epochs, loss.item(),
            100 * acc_train, 100 * acc_val, time.time()-t)
        )


def teest():
    with torch.no_grad():
        model.eval()
        outputs = model(features, adj, node_weight)
        test_acc = accuracy(outputs[idx_test], labels[idx_test])
        print('test acc: {:.2f}'.format(test_acc*100))
        visualize(outputs, color=labels, model_name=model.name)
        print('tSNE image is generated')
    
   
# Train model 
t_total = time.time()
for epoch in range(num_epochs):
    train(epoch)

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
teest()

# 保存模型
model_save_path = '../saved_model/model.pth'
torch.save(model.state_dict(), model_save_path)
print()

import pickle
f = open('../saved_model/node_features.pkl', "wb")
pickle.dump(features, f)
f.close()