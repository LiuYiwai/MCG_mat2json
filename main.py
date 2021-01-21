import json
import os
import multiprocessing
import time
import scipy.io as scio
from utils import *

base_dir = '~/'
name_path = os.path.join(base_dir, 'trainval.txt')
thread_num = 8
name_list = []
with open(name_path, 'r') as f:
    for line in f:
        line = line.strip('\n')
        name_list.append(line)
data_path = os.path.join(base_dir, 'MCG-Pascal-Main_trainvaltest_2012-proposals')
json_path = os.path.join(base_dir, 'proposals')
global_limit = 100
# global_limit = 200
execution_interval = []


def work(start, end):
    for name in name_list[start:end]:
        print(name)
        mat_path = os.path.join(data_path, name + '.mat')
        data = scio.loadmat(mat_path)
        superpixels = data['superpixels']
        labels = data['labels']
        proposals = []
        limit = min(global_limit, len(labels))
        for item in labels[:limit]:
            label = item[0][0]
            img = np.zeros(shape=superpixels.shape)
            for index in label:
                mask = superpixels == index
                img[mask] = 1
            img = img.astype(bool)
            proposals.append(img)

        proposals_encode = list(map(rle_encode, proposals))
        json_file_name = os.path.join(json_path, name + '.json')
        with open(json_file_name, 'w') as f:
            json.dump(proposals_encode, f)


def preparation():
    global execution_interval
    execution_interval = []
    name_len = len(name_list)
    for i in range(thread_num - 1):
        execution_interval.append(int(i * name_len / thread_num))
    execution_interval.append(name_len)


def multiprocessing_function():
    processes = []
    for index in range(len(execution_interval) - 1):
        processes.append(multiprocessing.Process(target=work, args=(execution_interval[index],
                                                                    execution_interval[index + 1])))
    for p in processes:
        p.start()
    for p in processes:
        p.join()


def main():
    start = time.time()
    preparation()
    multiprocessing_function()
    print("over")
    end = time.time()
    print(str(round(end - start, 3)) + 's')


if __name__ == '__main__':
    main()
