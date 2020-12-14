import random
import torch
import time
import logging
import postprocess
from eval.eval_access import eval_access
import os
from image_models import tensor_to_image

def train_model(epoch, opt, model, optimizer, train, valid, valid_tree_list, valid_data, best_eval_likelihood, patient, word_lexicon=None, unk_index=1):
    """
    Training model for one epoch
    """
    model.train()

    image_loss_weight = opt.image_loss_weight
    reconstruction_loss_weight = opt.reconstruction_loss_weight

    total_structure_loss, total_image_loss, total_tag, total_reconstruction_loss = 1e-7, 1e-7, 1e-7, 1e-7

    cnt = 0
    start_time = time.time()

    train_w, train_img, train_lens, train_indices, train_hapax_indices = train
    max_cnt = len(train_w)
    # tenths = list([int(max_cnt / 20) * i for i in range(1, 20)])

    # shuffling the training data each epoch
    lst = list(range(len(train_w)))
    random.shuffle(lst)

    train_w = [train_w[l] for l in lst]
    train_img = [train_img[l] for l in lst]
    train_lens = [train_lens[l] for l in lst]
    train_indices = [train_indices[l] for l in lst]
    train_hapax_indices = [train_hapax_indices[l] for l in lst]

    for w, img, lens, indices, hapax_indices in zip(train_w, train_img, train_lens, train_indices, train_hapax_indices):
        # if lens[0] != 3: continue

        cnt += 1
        gpu_w = w.to(opt.device)
        gpu_img = torch.stack([img_t.squeeze() for img_t in img]).to(opt.device)

        structure_loss, semvisual_distance, reconstruction_loss = model.forward(gpu_w, gpu_img)

        if opt.reconstruction_prior_epochs > 0 and epoch < opt.reconstruction_prior_epochs:
            loss = reconstruction_loss + structure_loss * 0 + semvisual_distance * 0
        else:
            loss = structure_loss + semvisual_distance * image_loss_weight + reconstruction_loss * reconstruction_loss_weight
        # logging.info('structure: {}; image: {}'.format(structure_loss.item(), img_loss.item()))
        if torch.isnan(loss).item():
            logging.warning(
                "Epoch={} iter={} Structure loss={:.4f} SemVisual loss={:.4f} Recon loss={:.4f}".format(
                    epoch, cnt,
                    structure_loss.item(), semvisual_distance.item(), reconstruction_loss.item()))
            raise ValueError('NaN found in loss!')

        loss.backward()
        total_structure_loss += structure_loss.item()

        if image_loss_weight > 0:
            total_image_loss += semvisual_distance.item()
        if reconstruction_loss_weight > 0:
            total_reconstruction_loss += reconstruction_loss.item()

        total_tag += sum(lens)

        global_cnt = cnt + epoch * max_cnt

        updated = False
        if global_cnt % opt.update_frequency == 0:
            if opt.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip_grad)
            optimizer.step()
            optimizer.zero_grad()
            updated = True

        # print(sloss/sum(lens))

        global_update_count = global_cnt // opt.update_frequency
        local_update_count = cnt // opt.update_frequency

        if local_update_count % 5e3 == 0 and updated:
            logging.info("Epoch={} iter={} Structure loss={:.4f} SemVisual loss={:.4f} Recon loss={:.4f} time={:.2f}s".format(
                epoch, local_update_count,
                total_structure_loss / total_tag, total_image_loss/total_tag, total_reconstruction_loss/total_tag, time.time() - start_time
            ))

            start_time = time.time()
            # model.writer.add_scalar('train_accumulative/average_total_loss', total_loss / total_tag, global_step)
            total_structure_loss = 0
            total_image_loss = 0
            total_reconstruction_loss = 0
            total_tag = 0

        if (global_update_count - opt.eval_start_iter) % opt.eval_steps == 0 and global_update_count >= opt.eval_start_iter and updated:

            logging.info('EVALING at cnt {}'.format(global_update_count))

            total_eval_likelihoods, trees = parse_dataset(model, valid, epoch, opt)

            if all(trees):
                tree_fn, valid_pred_trees = postprocess.print_trees(trees, valid_data, global_update_count, opt)
                eval_access(valid_pred_trees, valid_tree_list, model.writer, epoch)

            logging.info('Saving model for the final epoch')
            model_save_path = os.path.join(opt.model_path, 'model.pth')
            torch.save(model.state_dict(), model_save_path)

            if total_eval_likelihoods < best_eval_likelihood:
                logging.info('Better model found based on likelihood: {}! vs {}'.format(total_eval_likelihoods, best_eval_likelihood))
                best_eval_likelihood = total_eval_likelihoods
                patient = 0

                model_save_path = os.path.join(opt.model_path, 'best_model.pth')
                torch.save(model.state_dict(), model_save_path)

                best_grammar_save_path = os.path.join(opt.model_path, 'best_grammar.pth')
                best_grammar = model.save_grammar()
                torch.save(best_grammar, best_grammar_save_path)
            else:
                patient += 1
                if patient >= opt.eval_patient:
                    patient = -1
                    return optimizer, best_eval_likelihood, patient

    return optimizer, best_eval_likelihood, patient

def parse_dataset(model, dataset, epoch, opt):
    model.eval()
    with torch.no_grad():
        train_w, train_img, train_lens, train_indices, train_hapax_indices = dataset
        trees = [None] * sum([len(x) for x in train_indices])

        total_structure_loss = 0
        total_semivsual_loss = 0
        total_recon_loss = 0

        for batch_index, (w, img, lens, indices) in enumerate(zip(train_w, train_img, train_lens, train_indices)):
            gpu_w = w.to(opt.device)
            if all([x is not None for x in img]):
                gpu_img = torch.stack([img_t.squeeze() for img_t in img]).to(opt.device)
            else:
                gpu_img = None

            if batch_index == 0:
                structure_loss, v_treelist, semvisual_loss, recon_loss = model.parse(gpu_w, gpu_img, indices, set_pcfg=True)
            else:
                structure_loss, v_treelist, semvisual_loss, recon_loss = model.parse(gpu_w, gpu_img, indices, set_pcfg=False)

            if v_treelist:
                for t_id, t in zip(indices, v_treelist):
                    trees[t_id] = t

            total_structure_loss += structure_loss
            total_semivsual_loss += semvisual_loss
            total_recon_loss += recon_loss

        logging.info(
                'Epoch {} EVALUATION | Structure loss {:.4f} | SemVisual loss {:.4f} | Recon loss {:.4f} '.format(epoch, total_structure_loss, total_semivsual_loss, total_recon_loss))
        total_loss = (-1) * total_structure_loss + total_recon_loss + total_semivsual_loss
    return total_loss, trees

def likelihood_dataset(model, dataset, epoch, section='dev'):
    model.eval()
    with torch.no_grad():
        train_w, train_img, train_lens, train_indices = dataset
        total_structure_loss = 0
        total_num_tags = sum([sum(x) for x in train_lens])
        for batch_index, (w, img, lens, indices) in enumerate(zip(train_w, train_img, train_lens,
                                                                              train_indices)):
            if batch_index == 0:
                structure_loss = model.likelihood(w, img, indices, set_pcfg=True)
            else:
                structure_loss = model.likelihood(w, img, indices, set_pcfg=False)

            total_structure_loss += structure_loss

        model.writer.add_scalar(section+'_epochwise/average_structure_loss', total_structure_loss / total_num_tags, epoch)
        logging.info(
            'Epoch {} EVALUATION | Structure loss {:.4f} '.format(epoch, total_structure_loss))
    model.train()
    return total_structure_loss
