import torch
import torch.nn as nn
from gbertopic.bertopic.backend._sentencetransformers import SentenceTransformerBackend

# 定义文本嵌入层
class TextEmbedding(nn.Module):
    def __init__(self):
        super(TextEmbedding, self).__init__()
        # 初始化嵌入层，vocab_size 是词汇表大小，embed_dim 是嵌入向量的大小
        # self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)

        self.embedding_model = SentenceTransformerBackend("sentence-transformers/all-MiniLM-L6-v2")

    def forward(self, text_input):
        embeddings = self.embedding_model.embed_documents(document=text_input)
        return embeddings



# 定义双向LSTM层
class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(BiLSTM, self).__init__()
        # 初始化双向LSTM层
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, bidirectional=True)
    def forward(self, x):
        # x 应该是一个形状为 (batch_size, seq_len, input_dim) 的 FloatTensor
        x, _ = self.lstm(x)
        # 取出最后一层的输出
        x = x[:, -1, :]
        return x
# 定义多层感知器层
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        # 初始化MLP层
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        # x 应该是一个形状为 (batch_size, input_dim) 的 FloatTensor
        return self.fc(x)
# 定义整个模型
class ThreeLayerModel(nn.Module):
    def __init__(self, embed_dim, hidden_dim, mlp_hidden_dim, output_dim):
        super(ThreeLayerModel, self).__init__()
        self.embedding = TextEmbedding()
        self.bilstm = BiLSTM(input_dim=embed_dim, hidden_dim=hidden_dim)
        self.mlp = MLP(input_dim=hidden_dim, hidden_dim=mlp_hidden_dim, output_dim=output_dim)
    def forward(self, text_input):
        # 文本嵌入
        embed_output = self.embedding(text_input)
        # bi-LSTM操作
        bilstm_output = self.bilstm(embed_output)
        # MLP操作
        mlp_output = self.mlp(bilstm_output)
        return mlp_output
# 实例化模型
model = ThreeLayerModel(embed_dim=384, hidden_dim=64, mlp_hidden_dim=128, output_dim=1)
# 假设我们有一个批量大小为1，序列长度为20的文本数据
# 需要将文本数据转换为 LongTensor 类型
text_data = ['假设我们有一个批量大小为1，序列长度为20的文本数据', '需要将文本数据转换为 LongTensor 类型']
# 创建一个dummy input
dummy_input = torch.randint(0, 10000, (1, 20))
# 计算模型输出
output = model(text_data)
print(output)