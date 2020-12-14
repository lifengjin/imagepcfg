import os
import sys, gzip
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

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', required=True)
    parser.add_argument('--toks-path', required=True)
    parser.add_argument('--image-path', default=None, type=str)
    parser.add_argument('--gold-path')
    parser.add_argument('--first-k', default=None, type=int)

    parser.add_argument('--parseeval', default=False, action='store_true', help='parsing eval experiment')
    parser.add_argument('--unkeval', default=False, action='store_true', help='unk parsing eval experiment')

    parser.add_argument('--ppeval', default=False, action='store_true', help='the gavagai experiment')

    parser.add_argument('--device', default='cuda')

    args = parser.parse_args()

    logfile_fh = open(os.path.join(args.model_path, 'eval.results'), 'w')
    streamhandler = logging.StreamHandler(sys.stdout)
    filehandler = logging.StreamHandler(logfile_fh)
    handler_list = [filehandler, streamhandler]
    logging.basicConfig(level='INFO', format='%(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', handlers=handler_list)

    opt = torch.load(os.path.join(args.model_path, 'opt.pth')) # original configuration of the model
    logging.info('Model folder is: {}'.format(args.model_path))
    logging.info('Eval set is: {}'.format(args.toks_path))
    logging.info('Gold set is: {}'.format(args.gold_path))
    logging.info('Seed is: {}'.format(opt.seed))
    opt.model_path = args.model_path

    char_lexicon = bidict.bidict()
    word_lexicon = bidict.bidict()
    char_grams_lexicon = bidict.bidict()

    with open(os.path.join(args.model_path, 'char.dic'), encoding='utf-8') as fpo:
        for line in fpo:
            ch, i = line.strip().split('\t')
            char_lexicon[ch] = int(i)

    with open(os.path.join(args.model_path, 'word.dic'), encoding='utf-8') as fpo:

        for line in fpo:
            w, i = line.strip().split('\t')
            word_lexicon[w] = int(i)

    parse_toks = preprocess.read_corpus(args.toks_path, validset=True )
    if args.first_k is not None:
        parse_toks = parse_toks[:args.first_k]
    logging.info('Total number of sentences {}.'.format(len(parse_toks)))

    if args.unkeval:
        random.seed(1)
        logging.info('Replace words with UNKs.')
        replaces = 0
        unked_parse_toks = []
        for sent in parse_toks:
            new_sent = sent[:]
            num_sent_unk = 0
            for word in sent:
                if word not in word_lexicon:
                    num_sent_unk += 1
                    break
            else:
                if sent[-1] == '.' or sent[-1] == '。':
                    replace_index = random.randint(0, len(sent)-2)
                else:
                    replace_index = random.randint(0, len(sent) - 1)
                new_sent[replace_index] = 'UNK_SYMBOL'
                replaces += 1
            unked_parse_toks.append(new_sent)
        logging.info('Replaced {} sentences.'.format(replaces))

    if args.image_path is not None:
        valid_image_data = torch.load(args.image_path)
    else:
        valid_image_data = [None] * len(parse_toks)

    parse_patches = preprocess.create_batches(parse_toks, valid_image_data, 1, word_lexicon, char_lexicon)

    if args.unkeval:
        unk_parse_patches = preprocess.create_batches(unked_parse_toks, valid_image_data, 1, word_lexicon, char_lexicon)


    logging.info('Word vocab size: {}'.format(len(word_lexicon)))
    logging.info('Char vocab size: {}'.format(len(char_lexicon)))

    pcfg_parser = PCFG(num_nonterminals=opt.num_nonterminals, num_chars=len(char_lexicon),
                       device=opt.device, num_words=len(word_lexicon), model_type=opt.model_type,
                       state_dim=opt.state_dim)

    if opt.image_loss_weight > 0:
        if not hasattr(opt, 'no_encoder'):
            opt.no_encoder = False
        image_net = ImageNet(embedding_dim=opt.state_dim, img_dim=opt.img_dim,
                             word_embs=pcfg_parser.emit_prob_model.word_embs_module,
                             loss_type=opt.loss_type, pretrained_imgemb=opt.pretrained_imgemb,
                             projector_type=opt.projector_type, no_encoder=opt.no_encoder)
    else:
        image_net = None

    model = CharPCFG(pcfg_parser, image_net)

    logging.info('Model type is: {}'.format(opt.model_type))
    logging.info('Eval corpus size: {}'.format(len(parse_toks)))

    if args.device == 'cpu':
        model.load_state_dict(torch.load(os.path.join(args.model_path, 'best_model.pth'), map_location=args.device))
    else:
        model.load_state_dict(torch.load(os.path.join(args.model_path, 'best_model.pth')))
        model.to(args.device)

    if args.parseeval:
        logging.info('------------Begin normal parse eval---------------')

        with open(args.gold_path) as tfh:
            parse_tree_list = [x.strip() for x in tfh]

        total_eval_likelihoods, trees = model_use.parse_dataset(model, parse_patches, 'parseeval', opt)
        #
        logging.info('Total likelihood for valid: {}'.format(total_eval_likelihoods))

        tree_fn, valid_pred_trees = postprocess.print_trees(trees, parse_toks, '-parseeval', opt)
        eval_access(valid_pred_trees, parse_tree_list, None, '-parseeval')

    if args.unkeval:
        logging.info('------------Begin UNK parse eval---------------')
        with open(args.gold_path) as tfh:
            parse_tree_list = [x.strip() for x in tfh]

        total_eval_likelihoods, trees = model_use.parse_dataset(model, unk_parse_patches, 'unkeval', opt)
        #
        logging.info('Total likelihood for valid UNK dataset: {}'.format(total_eval_likelihoods))

        tree_fn, valid_pred_trees = postprocess.print_trees(trees, parse_toks, '-unkeval', opt)
        eval_access(valid_pred_trees, parse_tree_list, None, '-unkeval')


    if args.ppeval:
        logging.info('------------Begin PP parse eval---------------')

        with open(args.gold_path) as tfh:
            lines = tfh.readlines()

            gold_spans = [ ]

            attachment_indicators = []

            high_attachment = 0
            low_attachment = 0
            for line in lines:
                line = line.strip()
                start_index = line.index('(')
                end_index = line.index(')')
                if end_index == len(line)-1:
                    low_attachment += 1
                    attachment_indicators.append('L')
                else:
                    high_attachment += 1
                    attachment_indicators.append('H')

                span = line[start_index+2:end_index-1]
                spaced_span = span.replace('\t', ' ')
                gold_spans.append(spaced_span)
            assert len(gold_spans) == high_attachment + low_attachment

        total_eval_likelihoods, trees = model_use.parse_dataset(model, parse_patches, 'parseeval', opt)

        tree_fn, valid_pred_trees = postprocess.print_trees(trees, parse_toks, '-ppeval', opt)

        correct = 0
        total = len(gold_spans)

        correct_high = 0
        correct_low = 0

        for index, tree in enumerate(valid_pred_trees):
            for subtree in tree.subtrees(lambda x: x.height() > 2):
                span = ' '.join(subtree.leaves())
                if span == gold_spans[index]:
                    correct += 1
                    if attachment_indicators[index] == 'H':
                        correct_high += 1
                    else:
                        correct_low += 1
                    break
        logging.info('PP attachment High: {}; Low: {}'.format(high_attachment, low_attachment))

        logging.info('PP attachment accuracy: {}'.format(correct / total))
        logging.info('PP attachment High accuracy: {}'.format(correct_high / high_attachment))
        logging.info('PP attachment Low accuracy: {}'.format(correct_low / low_attachment))
