import matplotlib
from matplotlib import pyplot as plt

y = []
with open('tmp.txt', 'r', encoding='utf-8') as f:
    for line in f:
        if 'sse' in line:
            y.append(float(line.strip().split(' ')[1]))
        if len(y) > 50:
            break
x = range(3, 3+len(y))
plt.plot(x, y)
plt.xlabel('主题数目')  # 9
plt.ylabel('SSE')
plt.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.title('主题-SSE变化情况')
plt.show()