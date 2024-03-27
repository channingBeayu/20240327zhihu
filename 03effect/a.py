hc_data = []


with open('data/hcs.txt', 'r') as file:
    for line in file:
        datas = line.split()
        if len(datas) != 3:
            print(1)



