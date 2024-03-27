import pickle

from gbertopic.bertopic.backend._sentencetransformers import SentenceTransformerBackend
import numpy as np

with open('utils/topic_model.pkl', 'rb') as f:
    topic_model = pickle.loads(f.read())


embedding_model = SentenceTransformerBackend("sentence-transformers/all-MiniLM-L6-v2")
embeddings = embedding_model.embed_documents(document=['哈尔滨', '哈哈'])
# topic_model.transform(['哈尔滨', '哈哈'], np.asarray([embeddings]))
topic_model.transform(['哈尔滨', '哈哈'], embeddings)

print(1)
