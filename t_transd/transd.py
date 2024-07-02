import torch
import torch.nn as nn
import torch.nn.functional as F

from openke.data import TrainDataLoader, TestDataLoader
from openke.loss import MarginLoss
from openke.strategy import NegativeSampling
from openke.config import Trainer


class TransD(nn.Module):
    def __init__(self, ent_tot, rel_tot, dim_e=100, dim_r=100, p_norm=1, norm_flag=True, margin=None, epsilon=None):
        super(TransD, self).__init__()
        self.ent_tot = ent_tot
        self.rel_tot = rel_tot
        self.dim_e = dim_e
        self.dim_r = dim_r
        self.margin = margin
        self.epsilon = epsilon
        self.norm_flag = norm_flag
        self.p_norm = p_norm
        # embedding
        self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim_e)
        self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim_r)
        self.ent_transfer = nn.Embedding(self.ent_tot, self.dim_e)
        self.rel_transfer = nn.Embedding(self.rel_tot, self.dim_r)

        if margin == None or epsilon == None:
            nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
            nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
            nn.init.xavier_uniform_(self.ent_transfer.weight.data)
            nn.init.xavier_uniform_(self.rel_transfer.weight.data)
        else:
            self.ent_embedding_range = nn.Parameter(
                torch.Tensor([(self.margin + self.epsilon) / self.dim_e]), requires_grad=False
            )
            self.rel_embedding_range = nn.Parameter(
                torch.Tensor([(self.margin + self.epsilon) / self.dim_r]), requires_grad=False
            )
            nn.init.uniform_(
                tensor=self.ent_embeddings.weight.data,
                a=-self.ent_embedding_range.item(),
                b=self.ent_embedding_range.item()
            )
            nn.init.uniform_(
                tensor=self.rel_embeddings.weight.data,
                a=-self.rel_embedding_range.item(),
                b=self.rel_embedding_range.item()
            )
            nn.init.uniform_(
                tensor=self.ent_transfer.weight.data,
                a=-self.ent_embedding_range.item(),
                b=self.ent_embedding_range.item()
            )
            nn.init.uniform_(
                tensor=self.rel_transfer.weight.data,
                a=-self.rel_embedding_range.item(),
                b=self.rel_embedding_range.item()
            )
        if margin != None:
            self.margin = nn.Parameter(torch.Tensor([margin]))
            self.margin.requires_grad = False
            self.margin_flag = True
        else:
            self.margin_flag = False

    # 主要用于填充,
    # axis:需要填充的维度
    # size:需要填充后的大小
    def _resize(self, tensor, axis, size):
        shape = tensor.size()
        osize = shape[axis]
        if osize == size:
            return tensor
        if osize > size:
            return torch.narrow(tensor, axis, 0, size)
        paddings = []
        # 倒序填充
        for i in range(len(shape)):
            if i == axis:
                paddings = [0, size - osize] + paddings
            else:
                paddings = [0, 0] + paddings
        return F.pad(tensor, paddings=paddings, mode='constant', value=0)

    def _calc(self, h, t, r, mode):
        if self.norm_flag:
            h = F.normalize(h, 2, -1)
            r = F.normalize(r, 2, -1)
            t = F.normalize(t, 2, -1)
        if mode != 'normal':
            h = h.view(-1, r.shape[0], h.shape[-1])
            t = t.view(-1, r.shape[0], t.shape[-1])
            r = r.view(-1, r.shape[0], r.shape[-1])
        if mode == 'head_batch':
            score = h + (r - t)
        else:
            score = (h + r) - t
        score = torch.norm(score, self.p_norm, -1).flatten()
        return score

    def _transfer(self, e, e_transfer, r_transfer):
        if e.shape[0] != r_transfer.shape[0]:
            e = e.view(-1, r_transfer.shape[0], e.shape[-1])
            e_transfer = e_transfer.view(-1, r_transfer.shape[0], e_transfer.shape[-1])
            r_transfer = r_transfer.view(-1, r_transfer.shape[0], r_transfer.shape[-1])
            e = F.normalize(
                self._resize(e, -1, r_transfer.size()[-1]) + torch.sum(e * e_transfer, -1, True) * r_transfer,
                p=2,
                dim=-1
            )
            return e.view(-1, e.shape[-1])
        else:
            # torch.sum(e * e_transfer,-1,True): [num,1]
            # self._resize(e,-1,r_transfer.size()[-1]): [num,rel_dim]
            e = F.normalize(
                self._resize(e, -1, r_transfer.size()[-1]) + torch.sum(e * e_transfer, -1, True) * r_transfer,
                p=2,
                dim=-1
            )
            return e

    def forward(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        mode = data['mode']
        # embedding
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        # 得到投影矩阵需要的实体向量
        h_transfer = self.ent_transfer(batch_h)
        t_transfer = self.ent_transfer(batch_t)
        r_transfer = self.rel_transfer(batch_r)
        # 计算投影后的实体向量
        h = self._transfer(h, h_transfer, r_transfer)
        t = self._transfer(t, t_transfer, r_transfer)
        score = self._calc(h, t, r, mode)
        if self.margin_flag:
            return self.margin - score
        else:
            return score

    def regularization(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        # embedding
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        # 得到投影矩阵需要的实体向量
        h_transfer = self.ent_transfer(batch_h)
        t_transfer = self.ent_transfer(batch_t)
        r_transfer = self.rel_transfer(batch_r)
        # 计算正则项
        regul = (
                        torch.mean(h ** 2) +
                        torch.mean(t ** 2) +
                        torch.mean(r ** 2) +
                        torch.mean(h_transfer ** 2) +
                        torch.mean(t_transfer ** 2) +
                        torch.mean(r_transfer ** 2)
                ) / 6
        return regul

    def predict(self, data):
        score = self.forward(data)
        if self.margin_flag:
            return score.cpu().data.numpy()
        else:
            return score.cpu().data.numpy()


train_dataloader = TrainDataLoader(
    in_path="./benchmarks/FB15K237_tiny/",
    nbatches=100,
    threads=8,
    # 负采样
    sampling_mode='normal',
    # bern构建负样本方式
    bern_flag=1,
    # 负样本同replace entities
    neg_ent=25,
    neg_rel=0
)
transd = TransD(train_dataloader.get_ent_tot(), train_dataloader.get_rel_tot())

# dataloader for test
test_dataloader = TestDataLoader("./benchmarks/FB15K237_tiny/", "link")
loss = MarginLoss()
model = NegativeSampling(transd, loss, batch_size=train_dataloader.batch_size)
trainer = Trainer(model=model, data_loader=train_dataloader)
trainer.run()


