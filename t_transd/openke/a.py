import torch
import torch.nn as nn

# 前面部分照搬
# 1. 数据的预处理
sentences = ['It is a good day.','how are you?','I want to study the nn.embedding.','I want to elmate my pox.','the experience that I have done today is my favriate experience.']
sentences = [sentence.split() for sentence in sentences]
all_words = []
total_words = 0
for sentence in sentences:
    all_words += [ words for words in sentence ]
no_repeat_words = set(all_words)
total_words = len(no_repeat_words)
word_to_idx = {word: i+1 for i, word in enumerate(no_repeat_words)}
word_to_idx['<unk>'] = 0
idx_to_word = {i+1: word for i, word in enumerate(no_repeat_words)}

# 2. word to vector，将句子转化成向量
word_vector = []
sentences_pad = []
max_len = max([len(sentence) for sentence in sentences])
for sentence in sentences:
    if len(sentence) < max_len:
        sentences_pad += [sentence.extend("<unk>" for _ in range(max_len-len(sentence)))]
    else:
        sentences_pad += [sentence]
for sentence in sentences:
    word_vector += [[ word_to_idx[word] for word in sentence]]

# 3.传入向量化的句子，生成字向量
total_words = len(word_to_idx)
class myEmbed(nn.Module):
    def __init__(self,total_words,embedding_dim):
        super(myEmbed,self).__init__()
        self.embed = nn.Embedding(total_words,embedding_dim)
    def forward(self,sentences_idx):
        return self.embed(sentences_idx).clone().detach()
output_emb = myEmbed(total_words = total_words, embedding_dim = 8)
word_vector = torch.tensor(word_vector, dtype=torch.long).clone().detach()
output = output_emb(word_vector)
print(output)
