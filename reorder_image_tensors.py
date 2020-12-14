import argparse
import torch
import bidict

parser = argparse.ArgumentParser()

parser.add_argument('--tensor-pt')
parser.add_argument('--mapping')

args = parser.parse_args()

tensors = torch.load(args.tensor_pt)

reordered_tensors = [None] * len(tensors)

mapping_dict = bidict.bidict()

with open(args.mapping) as mfh:
    for line in mfh:
        img_index, tree_index = line.strip().split('\t')
        img_index = int(img_index)
        tree_index = int(tree_index)
        if tree_index < 0:
            continue
        mapping_dict[img_index] = tree_index

for img_index, tensor in tensors.items():
    tree_index = mapping_dict[img_index]
    reordered_tensors[tree_index] = tensor

torch.save(reordered_tensors, args.tensor_pt.replace('.pt', '.treeordered.pt'))

