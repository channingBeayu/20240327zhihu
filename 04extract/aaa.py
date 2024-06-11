import re

a = re.findall(r'\[(.*?)\]', '[11]33')
print(a)


text = "这里是一些[文字]。这里还有一些[其他的文字]。"

# 使用正则表达式匹配中括号内的内容
matches = re.findall(r'\[(.*?)[\]]', text)[0]
print(matches)
# 打印匹配结果
# for match in matches:
#     print(match)