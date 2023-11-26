import random
random.seed(300)
list = []
sub_list = []
for i in range(2):
    for j in range(2): 
        k = random.randint(0,13)
        sub_list.append(k)
    print(sub_list)
    list.append(sub_list)
    # print(list)
    #sub_list = []

# print(list)