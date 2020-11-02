import torch
from torch import nn
import torch.nn.functional as F
from cky_parser_sgd import batch_CKY_parser

class ResidualLayer(nn.Module):
    def __init__(self, in_dim=100,
                 out_dim=100):
        super(ResidualLayer, self).__init__()
        self.lin1 = nn.Linear(in_dim, out_dim)
        self.lin2 = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        return F.relu(self.lin2(F.relu(self.lin1(x)))) + x

class WordProbFCFixVocab(nn.Module):
    def __init__(self, num_words, state_dim, dropout=0.0):
        super(WordProbFCFixVocab, self).__init__()
        self.fc = nn.Sequential(nn.Linear(state_dim*2, state_dim*2),
                                # nn.ReLU(),
                                ResidualLayer(state_dim*2, state_dim*2),
                                ResidualLayer(state_dim*2, state_dim*2),
                                nn.Linear(state_dim*2, 1))
        self.word_embs_module = nn.Embedding(num_words, state_dim)
        self.word_embs = self.word_embs_module.weight

    def forward(self, words, cat_embs, set_pcfg=True):
        if set_pcfg:
            dist = self.prep_input(cat_embs)
            self.dist = dist
        else:
            pass
        # word_indices = words[:, 1:-1]
        word_indices = words

        logprobs = self.dist[word_indices, :] # sent, word, cats; get rid of bos and eos
        return logprobs

    def prep_input(self, cat_embs):
        # cat_embs is num_cat, cat_dim
        expanded_embs = self.word_embs # words, dim
        cat_embs = cat_embs # cats, dim

        outs = []
        for cat_emb in cat_embs:
            cat_emb = cat_emb.unsqueeze(0).expand(self.word_embs.shape[0], -1) # words, dim

            inp = torch.cat([expanded_embs, cat_emb], dim=-1) # words, dim*2
            out = self.fc(inp) # vocab, 1
            outs.append(out)
        outs = torch.cat(outs, dim=-1) # vocab, cats
        logprobs = nn.functional.log_softmax(outs, dim=0) # vocab, cats

        return logprobs


class PCFG(nn.Module):
    def __init__(self,
                 state_dim=64,
                 num_nonterminals=60,
                 device='cpu',
                 model_type='char',
                 num_words=100):
        super(PCFG, self).__init__()
        self.state_dim = state_dim
        self.model_type = model_type
        self.all_states = num_nonterminals
        if self.model_type == 'word':
            self.emit_prob_model = WordProbFCFixVocab(num_words, state_dim)

        self.nont_emb = nn.Parameter(torch.randn(self.all_states, state_dim))

        # self.rule_mlp = nn.Sequential(TrilinearLayerForward(state_dim, state_dim),
        #                               TrilinearLayerCompose(state_dim, state_dim))
        self.rule_mlp = nn.Linear(state_dim, self.all_states ** 2)

        self.root_emb = nn.Parameter(torch.randn(1, state_dim))
        self.root_mlp = nn.Sequential(nn.Linear(state_dim, state_dim),
                                      ResidualLayer(state_dim, state_dim),
                                      ResidualLayer(state_dim, state_dim),
                                      nn.Linear(state_dim, self.all_states))

        self.split_mlp = nn.Sequential(nn.Linear(state_dim, state_dim),
                                       ResidualLayer(state_dim, state_dim),
                                       ResidualLayer(state_dim, state_dim),
                                       nn.Linear(state_dim, 2))

        self.pcfg_parser = batch_CKY_parser(nt=self.all_states, t=0, device=device)

        # for m in self.parameters():
        #     nn.init.normal_(m, mean=0., std=0.01)


    def forward(self, x, argmax=False, use_mean=False, indices=None, set_pcfg=True, return_ll=True, **kwargs):
        # x : batch x n
        if set_pcfg:
            self.emission = None

            nt_emb = self.nont_emb

            root_scores = F.log_softmax(self.root_mlp(self.root_emb).squeeze(), 0)
            full_p0 = root_scores

            # rule_score = F.log_softmax(self.rule_mlp([nt_emb, nt_emb, nt_emb]).squeeze().reshape([self.all_states, self.all_states**2]), dim=1)
            rule_score = F.log_softmax(self.rule_mlp(nt_emb), 1)  # nt x t**2

            full_G = rule_score
            split_scores = F.log_softmax(self.split_mlp(nt_emb), dim=1)
            full_G = full_G + split_scores[:, 0][..., None]

            self.pcfg_parser.set_models(full_p0, full_G, self.emission, pcfg_split=split_scores)

        if self.model_type != 'subgrams':
            x = self.emit_prob_model(x, self.nont_emb, set_pcfg=set_pcfg)
        else:
            x = self.emit_prob_model(x)

        if argmax:
            with torch.no_grad():
                logprob_list, vtree_list, vproduction_counter_dict_list, vlr_branches_list = \
                    self.pcfg_parser.marginal(x, viterbi_flag=True, only_viterbi=not return_ll, sent_indices=indices)
            return logprob_list, vtree_list, vproduction_counter_dict_list, vlr_branches_list
        else:
            logprob_list, _, _, _ = self.pcfg_parser.marginal(x)
            logprob_list = logprob_list * (-1)
            return logprob_list
