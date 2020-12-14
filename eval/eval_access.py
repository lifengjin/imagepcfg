from .compare_trees import main as sent_f1
from .evalb_unlabeled import eval_rvm_et_al, eval_postags
import tempfile
import nltk

def eval_access(pred_tree_list, gold_tree_list, writer, epoch, section='dev', mid_way_eval=False):

    if isinstance(gold_tree_list[0], str):
        gold_trees = []
        for t in gold_tree_list:
            gold_trees.append(nltk.tree.Tree.fromstring(t))
    else:
        gold_trees = gold_tree_list

    if mid_way_eval is False:
        p, r, f1, hom, rh = eval_rvm_et_al((gold_trees, pred_tree_list), mid_way_eval)

    else:
        matching_nums, gold_nums, pred_nums, gold_labels, pred_labels = eval_rvm_et_al((gold_trees, pred_tree_list), mid_way_eval)
        return list(matching_nums), list(gold_nums), list(pred_nums), list(gold_labels), list(pred_labels)
    # writer.add_scalar(section+'_epochwise/corpus_f1', corpus_f1_val, epoch)
    # writer.add_scalar(section+'_epochwise/corpus_recall', corpus_recall_val, epoch)
    # writer.add_scalar(section+'_epochwise/sent_recall', sent_recall_val, epoch)
    # writer.add_scalar(section+'_epochwise/sent_f1', sent_f1_val, epoch)
    if writer is not None:
        writer.add_scalar(section+'_epochwise/p', p, epoch)
        writer.add_scalar(section+'_epochwise/r', r, epoch)
        writer.add_scalar(section+'_epochwise/f1', f1, epoch)
        writer.add_scalar(section+'_epochwise/hom', hom, epoch)
        writer.add_scalar(section+'_epochwise/rh', rh, epoch)
    # eval_postags(gold_trees, pred_tree_list)
    return p, r, f1, hom, rh