import numpy as np
import logging
import os
import sys
import random
import torchtext
from utils.dataset_utils import check, separate_data, split_data, save_file
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


random.seed(1)
np.random.seed(1)
num_classes = 4
max_len = 200


# Allocate data to users
def generate_agnews(num_clients, num_classes, niid, balance, partition, dir_param):


    dir_path = f"AGNews_IID_Client{num_clients}/"
    if partition == 'pat':
        dir_path = f"AGNews_NonIID_Pat2_Client{num_clients}/"  # default pat=2 client 20
    elif partition == 'dir':
        dir_path = f"AGNews_NonIID_Dir{dir_param}_Client{num_clients}/"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

        
    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if check(config_path, train_path, test_path, num_clients, num_classes, niid, balance, partition, dir_param=dir_param):
        return

    # Get AG_News data
    print('downloading')
    trainset, testset = torchtext.datasets.AG_NEWS(root=dir_path+"rawdata")
    print('finish')
    trainlabel, traintext = list(zip(*trainset))
    testlabel, testtext = list(zip(*testset))

    dataset_text = []
    dataset_label = []

    dataset_text.extend(traintext)
    dataset_text.extend(testtext)
    dataset_label.extend(trainlabel)
    dataset_label.extend(testlabel)

    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(map(tokenizer, iter(dataset_text)), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    text_pipeline = lambda x: vocab(tokenizer(x))
    label_pipeline = lambda x: int(x) - 1

    def text_transform(text, label, max_len=0):
        label_list, text_list = [], []
        for _text, _label in zip(text, label):
            label_list.append(label_pipeline(_label))
            text_ = text_pipeline(_text)
            padding = [0 for i in range(max_len-len(text_))]
            text_.extend(padding)
            text_list.append(text_[:max_len])
        return label_list, text_list

    label_list, text_list = text_transform(dataset_text, dataset_label, max_len)

    text_lens = [len(text) for text in text_list]
    # max_len = max(text_lens)
    # label_list, text_list = text_transform(dataset_text, dataset_label, max_len)

    text_list = [(text, l) for text, l in zip(text_list, text_lens)]

    text_list = np.array(text_list, dtype=object)
    label_list = np.array(label_list)

    # dataset = []
    # for i in range(num_classes):
    #     idx = label_list == i
    #     dataset.append(text_list[idx])

    X, y, statistic, batch_size = separate_data((text_list, label_list), num_clients, num_classes, niid, balance, partition, class_per_client=2, dir_param=dir_param)
    print(batch_size)
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
            statistic, niid, balance, partition, dir_param=dir_param, batch_size=batch_size)

    print("The size of vocabulary:", len(vocab))


if __name__ == "__main__":
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None
    num_clients = int(sys.argv[4])
    logging.info(num_clients)
    alpha = float(sys.argv[5])
    logging.info(alpha)
    generate_agnews(num_clients, num_classes, niid, balance, partition, dir_param=alpha)