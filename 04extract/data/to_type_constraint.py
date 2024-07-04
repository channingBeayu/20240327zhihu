map = {}

file_path = 'test2id.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    line = file.readline()
    while True:
        line = file.readline()
        if not line:  # 如果读取到的行为空字符串，结束循环
            break
        line = line.split('\t')
        if line[0] not in map:
            map[line[0]] = [line[1]]
        else:
            map[line[0]].extend(line[1])
        print()

# 打开文件
file_path = 'type_constrain.txt'
with open(file_path, 'w', encoding='utf-8') as file:
    file.write(f'{len(map)}\n')
    for key in map:
        file.write(f'{key}')
        for ele in map[key]:
            file.write(f'\t{ele}')
        file.write('\n')


