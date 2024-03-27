import pandas as pd
# 创建一个DataFrame作为追加的基础
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6]
})
# 创建一个新的DataFrame，包含要追加的数据行
new_row = pd.DataFrame({
    'A': [4],
    'B': [7]
}, index=[3])  # 指定新的行索引
# 使用append()方法将新的数据行追加到原DataFrame中
df = df.append(new_row, ignore_index=True)
print(df)



import pandas as pd

data = pd.DataFrame(columns=('url',))

answerData = pd.DataFrame(columns=('url', ))
row = {'url': [1],
       'dd': [2]}
answerData.append(pd.DataFrame(row), ignore_index=True)

data = data.append(pd.DataFrame(answerData), ignore_index=True)
print(data)
if data.empty:
    print(1)

