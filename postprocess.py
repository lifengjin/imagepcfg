import gzip, os

def print_trees(trees, original_sents, epoch, opt):
    tree_filename = os.path.join(opt.model_path, 'e{}.vittrees.gz'.format(epoch))
    reconstituted_trees = []
    with gzip.open(tree_filename, 'wt', encoding='utf8') as ofh:
        for index, (tree, ori_sent) in enumerate(zip(trees, original_sents)):
            if tree is None: print(str(index)+'#!#!', file=ofh)
            # ori_sent = ori_sent[1:-1]
            assert len(tree.leaves()) == len(ori_sent), '\n'+str(index)+': '+str(tree) + '\n' + ' '.join(ori_sent)
            for index, position in enumerate(tree.treepositions('leaves')):
                # tree[position] = ori_sent[int(tree[position])]
                tree[position] = ori_sent[index]
            print(str(index)+'#!#!'+tree.pformat(margin=10000), file=ofh)
            reconstituted_trees.append(tree)
    return tree_filename, reconstituted_trees