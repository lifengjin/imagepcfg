# ImagePCFG
This is the code repo for the AACL paper **Grounded PCFG Induction with Images**. The paper can be found [here](https://www.aclweb.org/anthology/2020.aacl-main.42/).


## Requirements

- PyTorch 1.0+
- Numpy 
- NLTK
- Bidict

## Use

You can take a look at `model_args.py` to check out some of the possible parameters. Here is a command-line recipe:
```bash
python3 main.py --batch_size 2 --device cuda --clip_grad 5 --eval_patient 10 
--eval_start_iter 20000 --eval_steps 20000 --train_path [TRAINING_TEXT]
--train_image_path [TRAINING_IMAGES] --train_portion 1 --valid_path [VALIDATION_TEXT]
--valid_image_path [VALIDATION_IMAGES] --num_nonterminals 90 --model [NAME] --update_frequency 1
--max_epoch 100 --min_count 1 --max_vocab_size 10000 --reconstruction_prior_epochs -1 --projector_type cnn
--lr 0.0005 --loss_type mse --state_dim 64 --img_dim 64 --image_loss_weight 1. --reconstruction_loss_weight 1.
```

The training text data and images can be found at MSCOCO and Flickr dataset websites. The translated and parsed datasets 
will be posted here soon. 