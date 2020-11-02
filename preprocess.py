import logging
import random
import torch
from collections import Counter

EOS = '<eos>'
BOS = '<bos>'
PAD = '<pad>'
OOV = '<oov>'
BOW = '<bow>'
EOW = '<eow>'

def divide(data, valid_size, gold_tree_list, include_valid_in_train=False, all_train_as_valid=False):
    logging.info('include valid in train is {}; all train as valid is {}'.format(include_valid_in_train, all_train_as_valid))
    assert len(data) == len(gold_tree_list)
    valid_size = min(valid_size, len(data) // 10)
    train_size = len(data) - valid_size
    # random.shuffle(data)
    if include_valid_in_train:
        return data, data[train_size:], gold_tree_list, gold_tree_list[train_size:]
    elif all_train_as_valid:
        return data, data, gold_tree_list, gold_tree_list
    return data[:train_size], data[train_size:], gold_tree_list[:train_size], gold_tree_list[train_size:]

def get_truncated_vocab(dataset, min_count, max_num):

    word_count = Counter()
    for sentence in dataset:
        word_count.update(sentence)

    word_count = list(word_count.items())
    word_count.sort(key=lambda x: x[1], reverse=True)

    i = 0
    for word, count in word_count:
        if count < min_count:
            break
        i += 1
    if i > max_num:
        i = max_num

    logging.info('Truncated word count: {}.'.format(sum([count for word, count in word_count[i:]])))
    logging.info('Original vocabulary size: {}. Truncated vocab size {}.'.format(len(word_count), i))
    return word_count[:i]

def read_corpus(path, portion='1', validset=False):

    dataset = []
    with open(path, 'r', encoding='utf-8') as fin:
        for line in fin:
            sent = []
            words = line.strip().split()
            if len(words) > 40 and not validset:
                words = words[:40]
            for token in words:
                if len(token) > 35:
                    token = token[:20] + token[-15:]
                sent.append(token.lower())
            dataset.append(sent)
    total_nums = len(dataset)
    portion = int(portion)
    num_keeping = total_nums // portion
    return dataset[:num_keeping]


def create_one_batch(x, word2id, oov=OOV, pad=PAD, sort=True, device='cpu', hapax=None):

    num_unk = 0
    batch_size = len(x)
    lst = list(range(batch_size))
    if sort:
        lst.sort(key=lambda l: -len(x[l]))

    x = [x[i] for i in lst]
    lens = [len(x[i]) for i in lst]
    max_len = max(lens)
    batch_hapax_indices = []

    if word2id is not None:
        oov_id, pad_id = word2id.get(oov, None), word2id.get(pad, None)
        assert oov_id is not None and pad_id is not None
        batch_w = torch.LongTensor(batch_size, max_len).fill_(pad_id).to(device)
        for i, x_i in enumerate(x):
            for j, x_ij in enumerate(x_i):
                token_index = word2id.get(x_ij, oov_id)
                if token_index == oov_id:
                    num_unk += 1
                batch_w[i][j] = token_index
                if hapax and token_index in hapax:
                    batch_hapax_indices.append((i,j))
    else:
        batch_w = None

    return batch_w, lens, batch_hapax_indices


# shuffle training examples and create mini-batches
def create_batches(x, image_x, batch_size, word2id, hapax=None, sort=True):

    lst = list(range(len(x)))
    # print(len(image_x), len(x))
    if len(image_x) * 5 == len(lst):
        image_list = [x // 5 for x in lst]
    else:
        image_list = [x for x in lst]

    if sort:
        lst.sort(key=lambda l: -len(x[l]))

    sorted_x = [x[i] for i in lst]
    sorted_image_list = [image_x[image_list[i]] for i in lst]

    sum_len = 0.0
    batches_w, batches_img, batches_lens, batch_indices, batch_hapax_indices = [], [], [], [], []
    size = batch_size
    cur_len = 0
    start_id = 0
    end_id = 0
    for sorted_index in range(len(sorted_x)):
        if cur_len == 0:
            cur_len = len(sorted_x[sorted_index])
            if len(sorted_x) > 1:
                continue
        if cur_len != len(sorted_x[sorted_index]) or sorted_index - start_id == batch_size or sorted_index == len(sorted_x)-1:
            if sorted_index != len(sorted_x) - 1:
                end_id = sorted_index
            else:
                end_id = None

            if (end_id is None and len(sorted_x[sorted_index]) == cur_len and len(sorted_x) - start_id <= batch_size) or end_id is not None:

                bw, blens, hapax_indices = create_one_batch(sorted_x[start_id: end_id], word2id, sort=sort, hapax=hapax)
                batch_indices.append(lst[start_id:end_id])
                sum_len += sum(blens)
                batches_w.append(bw)
                batches_lens.append(blens)
                batches_img.append(sorted_image_list[start_id:end_id])
                batch_hapax_indices.append(hapax_indices)
                start_id = end_id
                cur_len = len(sorted_x[sorted_index])
                # print('1', batches_w[-1].shape, start_id, end_id)

            else:
                end_id = sorted_index
                bw, blens, hapax_indices = create_one_batch(sorted_x[start_id: end_id], word2id, sort=sort, hapax=hapax)
                batch_indices.append(lst[start_id:end_id])
                batches_img.append(sorted_image_list[start_id:end_id])
                sum_len += sum(blens)
                batches_w.append(bw)
                batches_lens.append(blens)
                batch_hapax_indices.append(hapax_indices)
                # print('2', batches_w[-1].shape, start_id, end_id)

                bw, blens, hapax_indices = create_one_batch(sorted_x[-1:], word2id, sort=sort, hapax=hapax)
                batch_indices.append(lst[-1:])
                batches_img.append(sorted_image_list[-1:])
                sum_len += sum(blens)
                batches_w.append(bw)
                batches_lens.append(blens)
                batch_hapax_indices.append(hapax_indices)
                # print('3', batches_w[-1].shape, start_id, end_id)

    nbatch = len(batch_indices)

    logging.info("{} batches, avg len: {:.1f}, max len {}, min len {}.".format(nbatch, sum_len / len(x), len(sorted_x[0]),
                                                                               len(sorted_x[-1])))

    perm = list(range(nbatch))
    random.shuffle(perm)

    batches_w = [batches_w[i] for i in perm]
    batches_img = [batches_img[i] for i in perm]
    batches_lens = [batches_lens[i] for i in perm]
    batch_indices = [batch_indices[i] for i in perm]
    batch_hapax_indices = [batch_hapax_indices[i] for i in perm]

    for i in range(len(batches_w)):
        if not batch_hapax_indices[i]: continue
        max_len = max([x[1] for x in batch_hapax_indices[i]])
        if max_len >= batches_w[i].size(1):
            print(batches_w[i].shape, batch_hapax_indices[i])

    return batches_w, batches_img, batches_lens, batch_indices, batch_hapax_indices

def read_markers(fname):
    markers = [0]
    with open(fname) as fh:
        for l in fh:
            marker = int(l.strip().split(' ')[1])
            markers.append(marker)
    return markers