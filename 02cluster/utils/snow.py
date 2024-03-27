

# s = SnowNLP('这个东西真的很赞')
# print(s.words)  # 返回分词结果
# print(s.sentiments)
import re

doc = '111\nd d</n>'

doc = re.sub(r'[\s\r\n]|</n>', "", doc)
print(doc)