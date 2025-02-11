import shutil
import os
import random

data_root = r'D:\DATABASE\ZXJ_GD\var_speed_sample\Continual_Learning\morlet_test'
class_list = os.listdir(data_root)
# class_list = class_list[0]
num_seq = list(range(900))
for i in class_list:
    random_train = random.sample(num_seq, 600)
    random_val = random.sample(num_seq, 300)
    file_list = os.listdir(os.path.join(data_root, i))
    os.makedirs(os.path.join(data_root, 'train', i))
    os.makedirs(os.path.join(data_root, 'val', i))
    file_select = list(map(lambda a: file_list[a], random_train))
    for j in range(len(file_select)):
        shutil.copy(os.path.join(data_root, i, file_select[j]), os.path.join(data_root, 'train', i, file_select[j]))
    file_select = list(map(lambda a: file_list[a], random_val))
    for j in range(len(file_select)):
        shutil.copy(os.path.join(data_root, i, file_select[j]), os.path.join(data_root, 'val', i, file_select[j]))