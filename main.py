import os
import sys, gzip, bz2
import argparse
import time
import random
import logging
import json
import bidict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from collections import Counter
from pcfg_models import PCFG
from top_models import *
import preprocess
import postprocess, model_use
import model_args
import numpy as np
from eval.eval_access import eval_access

def random_seed(seed_value, use_cuda):
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    random.seed(seed_value) # Python
    if use_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars

def train():

    opt = model_args.parse_args(sys.argv)

    # set seed before anything else.
    if opt.seed < 0: # random seed if seed is set to negative values
        opt.seed = int(int(time.time()) * random.random())
    random_seed(opt.seed, use_cuda=opt.device=='cuda')

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logfile_fh = gzip.open(os.path.join(opt.model_path, opt.logfile), 'wt')
    writer = SummaryWriter(os.path.join(opt.model_path, 'tensorboard'), flush_secs=10)
    filehandler = logging.StreamHandler(logfile_fh)
    streamhandler = logging.StreamHandler(sys.stdout)
    handler_list = [filehandler, streamhandler]
    logging.basicConfig(level='INFO', format='%(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', handlers=handler_list)

    # Dump configurations
    # print(opt)
    logging.info(opt)
    writer.add_text('args', str(opt))

    assert (opt.device == 'cuda' and torch.cuda.is_available()) or opt.device == 'cpu'

    train_data = preprocess.read_corpus(opt.train_path, portion=opt.train_portion)

    if opt.valid_path:
        valid_data = preprocess.read_corpus(opt.valid_path, validset=True)
    else:
        valid_data = None


    if opt.pretrained_imgemb:
        if opt.valid_image_path.endswith('bz2'):
            with bz2.open(opt.valid_image_path) as bzfh:
                valid_image_data = torch.from_numpy(np.load(bzfh).astype('float32'))
        else:
            valid_image_data = torch.from_numpy(np.load(opt.valid_image_path))
        if opt.train_image_path.endswith('bz2'):
            with bz2.open(opt.train_image_path) as bzfh:
                train_image_data = torch.from_numpy(np.load(bzfh).astype('float32'))
        else:
            train_image_data = torch.from_numpy(np.load(opt.train_image_path))
    else:
        if opt.train_image_path:
            valid_image_data = torch.load(opt.valid_image_path)
            if isinstance(valid_image_data[0], tuple):
                valid_image_data = [x[0] for x in valid_image_data]
            train_image_data = torch.load(opt.train_image_path)
            if isinstance(train_image_data[0], tuple):
                train_image_data = [x[0] for x in train_image_data]
        else:
            train_image_data = None

    logging.info('training instance: {}, training tokens: {}, max len: {}.'.format(len(train_data),
                                                                      sum([len(s) - 1 for s in train_data]), max([len(x) for x in train_data])))

    if opt.valid_gold_path:
        with open(opt.valid_gold_path) as tfh:
            valid_tree_list = [x.strip() for x in tfh]

    word_lexicon = bidict.bidict()

    # Maintain the vocabulary. vocabulary is used in either WordEmbeddingInput or softmax classification
    # logging.warning('enforcing minimun count of 1')
    # opt.min_count = 1
    vocab = preprocess.get_truncated_vocab(train_data, opt.min_count, opt.max_vocab_size)

    # Ensure index of '<oov>' is 0
    special_words = [preprocess.OOV, preprocess.BOS, preprocess.EOS, preprocess.PAD]
    special_chars =  [preprocess.BOS, preprocess.EOS, preprocess.OOV, preprocess.PAD,
                             preprocess.BOW, preprocess.EOW]

    for special_word in special_words:
        if special_word not in word_lexicon:
            word_lexicon[special_word] = len(word_lexicon)

    unk_index = word_lexicon[preprocess.OOV]
    hapax_words = set()
    for word, count in vocab:
        if word not in word_lexicon:
            word_lexicon[word] = len(word_lexicon)
        if count <= 2:
            hapax_words.add(word_lexicon[word])

    logging.info('Vocabulary size: {0}'.format(len(word_lexicon)) + '; Max length: {}'.format(max([len(x) for x in word_lexicon])))
    logging.info('Hapax set size: {}'.format(len(hapax_words)))

    # training batch size for the pre training is 8 times larger than in eval
    train = preprocess.create_batches(
            train_data, train_image_data, opt.batch_size, word_lexicon, hapax=hapax_words)

    logging.info('Evaluate every {0} epochs.'.format(opt.eval_steps))

    if valid_data is not None:
        valid = preprocess.create_batches(
            valid_data, valid_image_data, opt.batch_size, word_lexicon, char_lexicon)

    logging.info('vocab size: {0}'.format(len(word_lexicon)))

    pcfg_parser = PCFG(num_nonterminals=opt.num_nonterminals,
                       device=opt.device, num_words=len(word_lexicon), model_type=opt.model_type,
                       state_dim=opt.state_dim)

    if opt.image_loss_weight > 0:
        image_net = ImageNet(embedding_dim=opt.state_dim, img_dim=opt.img_dim,
                             word_embs=pcfg_parser.emit_prob_model.word_embs_module,
                             loss_type=opt.loss_type, pretrained_imgemb=opt.pretrained_imgemb,
                             projector_type=opt.projector_type, no_encoder=opt.no_encoder)
    else:
        image_net = None

    model = CharPCFG(pcfg_parser, image_net, writer=writer, no_structure=opt.no_structure)

    logging.info(str(model))
    num_grammar_params = 0
    for param in model.parameters():
        # print(param.sum().item())
        num_grammar_params += param.numel()
    logging.info("Top PCFG parser has {} parameters".format(num_grammar_params))

    model = model.to(opt.device)
    # if isinstance(model.pcfg, TopPCFGJin2NoPOSFlow) or isinstance(model.pcfg, TopPCFGJin2NoPOS):
    #     model.pcfg.setup_emission()

    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    with open(os.path.join(opt.model_path, 'char.dic'), 'w', encoding='utf-8') as fpo:
        for ch, i in char_lexicon.items():
            print('{0}\t{1}'.format(ch, i), file=fpo)

    with open(os.path.join(opt.model_path, 'word.dic'), 'w', encoding='utf-8') as fpo:
        for w, i in word_lexicon.items():
            print('{0}\t{1}'.format(w, i), file=fpo)

    opt_save_path = os.path.join(opt.model_path, 'opt.pth')
    torch.save(opt, opt_save_path)

    init_grammar_save_path = os.path.join(opt.model_path, 'init_grammar.pth')
    torch.save(model.init_grammar, init_grammar_save_path)

    best_eval_likelihood = float('inf')

    patient = 0

    for epoch in range(opt.max_epoch):


        optimizer, best_eval_likelihood, patient = model_use.train_model(epoch, opt, model, optimizer,
                                                                train, valid, valid_tree_list, valid_data, best_eval_likelihood, patient,
                                                                word_lexicon=word_lexicon, unk_index=unk_index)
        if patient < 0:
            break

    model.writer.close()


if __name__ == "__main__":
    train()

    logging.info('********** TRAINING IS OVER!! **********')
