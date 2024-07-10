# 加载主题模型
import pickle

with open('save/topic_model.pkl', 'rb') as f:
   topic_model = pickle.load(f)


print(topic_model.get_topics())
print()