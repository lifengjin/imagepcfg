import torch.distributions
from treenode import convert_binary_matrix_to_strtree
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torch import optim

class CharPCFG(nn.Module):
    def __init__(self, pcfg, image_net, no_structure=False, writer:SummaryWriter=None):
        super(CharPCFG, self).__init__()
        self.pcfg = pcfg
        self.no_structure = no_structure
        self.writer = writer
        self.image_net = image_net
        self.do_parsing = True # type: bool
        self.accumulated_embeddings = None
        self.turn_off_training_for_bilm = False
        self.init_grammar = self.save_grammar()

    def save_grammar(self):
        self.eval()
        with torch.no_grad():
            self.pcfg.emission = None

            nt_emb = self.pcfg.nont_emb

            root_scores = torch.nn.functional.log_softmax(self.pcfg.root_mlp(self.pcfg.root_emb).squeeze(), 0)

            # rule_score = F.log_softmax(self.rule_mlp([nt_emb, nt_emb, nt_emb]).squeeze().reshape([self.all_states, self.all_states**2]), dim=1)
            rule_score = torch.nn.functional.log_softmax(self.pcfg.rule_mlp(nt_emb), 1)  # t x t**2

            full_G = rule_score
            split_scores = torch.nn.functional.log_softmax(self.pcfg.split_mlp(nt_emb), dim=1)
            full_G = full_G + split_scores[:, 0][..., None] # t x t**2 + t x 1

            log_emit = self.pcfg.emit_prob_model.prep_input(nt_emb) # v x t
            log_emit = log_emit.t() + split_scores[:, 1][..., None] # t x v + t x 1

            self.train()
            return full_G.exp().to('cpu'), log_emit.exp().to('cpu'), root_scores.exp().to('cpu')

    def forward(self, word_inp, img_y):

        if not self.no_structure:
            logprob_list = self.pcfg.forward(word_inp)

            structure_loss = torch.sum(logprob_list, dim=0)
        else:
            structure_loss  = torch.tensor([0.], device=self.pcfg.nont_emb.device)

        if self.image_net is not None:
            semvisual_distance, reconstruction_loss = self.image_net.loss(word_inp, img_y)
        else:
            semvisual_distance, reconstruction_loss = torch.tensor([0.], device=self.pcfg.nont_emb.device), torch.tensor([0.], device=self.pcfg.nont_emb.device)

        return structure_loss, semvisual_distance, reconstruction_loss

    def parse(self, word_inp, img_y, indices, set_pcfg=True):

        if not self.no_structure:

            structure_loss, vtree_list, _, _ = self.pcfg.forward(word_inp, argmax=True,
                                                                 indices=indices, set_pcfg=set_pcfg)

        else:
            structure_loss = torch.tensor([0.], device=self.pcfg.nont_emb.device)
            vtree_list = []

        if self.image_net is not None and img_y is not None:
            semvisual_distance, reconstruction_loss = self.image_net.loss(word_inp, img_y)
        else:
            semvisual_distance, reconstruction_loss = 0., 0.

        semvisual_distance = semvisual_distance if isinstance(semvisual_distance, float) else semvisual_distance.item()
        reconstruction_loss = reconstruction_loss if isinstance(reconstruction_loss, float) else reconstruction_loss.item()

        return structure_loss.sum().item(), vtree_list, semvisual_distance, reconstruction_loss

    def likelihood(self, word_inp, indices, set_pcfg=True):

        structure_loss = self.pcfg.forward(word_inp, argmax=False, indices=indices, set_pcfg=set_pcfg)

        return structure_loss.sum().item()


from image_models import WordToImageProjector, ImageEncoder, ImageGenerator, CNNWordToImageProjector

class ImageNet(nn.Module):

    def __init__(self, embedding_dim, img_dim, word_embs, loss_type='mse', pretrained_imgemb=False, projector_type='ff',
                 no_encoder=False):
        super(ImageNet, self).__init__()
        self.no_encoder = no_encoder

        self.syntactic_dim = img_dim # dimension of the syntactic embedding from the projector

        if projector_type != 'cnn':
            self.projector = WordToImageProjector(embedding_dim, self.syntactic_dim)
        else:
            self.projector = CNNWordToImageProjector(embedding_dim, self.syntactic_dim)

        self.pretrained_imgemb = pretrained_imgemb

        if not self.pretrained_imgemb:
            if not self.no_encoder:
                self.image_encoder = ImageEncoder(img_dim)
            else:
                self.image_encoder = None
            self.image_decoder = ImageGenerator(img_dim)

        if loss_type == 'mse':
            self.image_embedding_loss = nn.MSELoss(reduction='sum')
        elif loss_type == 'itemave':
            self.image_embedding_loss = self.composer_itemave_loss
        elif loss_type == 'itemsum':
            self.image_embedding_loss = self.composer_itemsum_loss
        self.word_embs = word_embs
        self.internal_mse_loss = nn.MSELoss(reduction='none')

    def forward(self, x): # x is a matrix of word indices
        embs = self.word_embs(x) # batch, words, wordemb_dims
        # summed_embs = torch.sum(embs, dim=-1)
        embs = embs.unsqueeze(1) # batch, channel, words, wordemb_dims
        return self.projector(embs)

    def loss(self, words, image):
        semvisual_embedding = self(words)

        if not self.pretrained_imgemb:
            if not self.no_encoder:
                visual_embedding = self.image_encoder(image)
                d4_shapes = (visual_embedding.shape[0], visual_embedding.shape[1], 1, 1)
                reconstructed_img = self.image_decoder(visual_embedding.reshape(d4_shapes))
                reconstruction_loss = self.image_embedding_loss(image, reconstructed_img)
                semvisual_distance = self.image_embedding_loss(semvisual_embedding, visual_embedding[:,:self.syntactic_dim])
            else:
                d4_shapes = (semvisual_embedding.shape[0], semvisual_embedding.shape[1], 1, 1)
                reconstructed_img = self.image_decoder(semvisual_embedding.reshape(d4_shapes))
                reconstruction_loss = self.image_embedding_loss(image, reconstructed_img)
                semvisual_distance = torch.tensor([0.], device=reconstruction_loss.device)
            # print(semvisual_embedding.size(), visual_embedding.size())
        else:
            semvisual_distance = self.image_embedding_loss(semvisual_embedding, image)
            reconstruction_loss = torch.tensor([0.], device=semvisual_distance.device)
            # print(semvisual_embedding.size(), image.size())
        return semvisual_distance, reconstruction_loss

    def composer_itemave_loss(self, y, y_prime):
        distance_matrix = self.internal_mse_loss(y, y_prime)
        return distance_matrix.mean(dim=-1).sum()

    def composer_itemsum_loss(self, y, y_prime):
        distance_matrix = self.internal_mse_loss(y, y_prime)
        return distance_matrix.sum(dim=-1).mean()

    def generate_unk_embedding(self, words, image, unk_index):
        # print(words, unk_index)
        for index, word in enumerate(words[0]):
            if word.item() == unk_index:
                unk_local_index = index
        no_unk_embs = self.word_embs(words)
        no_unk_embs[0, unk_local_index] *= 0
        no_unk_embs = no_unk_embs.unsqueeze(0)
        no_unk_embs = no_unk_embs.detach()
        # print(no_unk_embs.size())
        unk_embedding = torch.normal(mean=no_unk_embs.squeeze().mean(dim=0), std=no_unk_embs.squeeze().std(dim=0))
        unk_embedding= unk_embedding.to(no_unk_embs.device)
        unk_embedding.requires_grad_(True)

        local_optimizer = optim.Adam([unk_embedding])

        visual_embedding = self.image_encoder(image).detach()
        prev_loss = 0

        for i in range(500):
            local_optimizer.zero_grad()
            no_unk_embs_local = no_unk_embs.clone()
            no_unk_embs_local[0, 0, unk_local_index] = unk_embedding + no_unk_embs_local[0, 0, unk_local_index]
            semvisual_embedding = self.projector(no_unk_embs_local)
            semvisual_distance = self.image_embedding_loss(semvisual_embedding, visual_embedding)
            semvisual_distance.backward()
            # print(i, '\t', semvisual_distance.item())
            if prev_loss == 0:
                prev_loss = semvisual_distance.item()
            else:
                if prev_loss - semvisual_distance.item() < 1e-4:
                    break
                else:
                    prev_loss = semvisual_distance.item()

            local_optimizer.step()

        return unk_embedding.detach().requires_grad_(False)

    def make_picture_from_word(self, words):

        semvisual_embedding = self(words)

        d4_shapes = (semvisual_embedding.shape[0], semvisual_embedding.shape[1], 1, 1)
        reconstructed_img = self.image_decoder(semvisual_embedding.reshape(d4_shapes))

        return reconstructed_img