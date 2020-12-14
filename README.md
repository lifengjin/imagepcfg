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

You can manipulate the `image_loss_weight` and  `reconstruction_loss_weight` to turn on and off different parts of the model.
For example, if you set both to 0., then you are essentially running a neural PCFG induction model with text only. If you set
`reconstruction_loss_weight` to be 0. but `image_loss_weight` to be 1, then you are running a neural PCFG induction model with 
the image encoder fixed during training. In this case, you should also provide pretrained image embeddings.

## Data
The training text data and images can be found at MSCOCO and Flickr dataset websites. The parse trees, tokenized sentences and 
translated sentences are available for download here for your convenience. Please let me know if you are interested in getting the
encoded images too. However, it should also be straightforward to generate your own image encodings with the provided captions through
simple mapping.

[MSCOCO](https://drive.google.com/file/d/1oTVRnuDLky0-Orh0KkrY6ChQCNNtErJy/view?usp=sharing) : This package includes the captions for training, validation and test from the MSCOCO dataset, which is
also translated into Chinese, Korean and Polish, and parsed with the Kitaev parser. The original dataset is in English and preprocessed by [Shi et al (2019)](https://ttic.uchicago.edu/~freda/project/vgnsl/).

[Flickr](https://drive.google.com/file/d/1PzDKLFIGDSNBqjVDIh0J08kVdC86m5va/view?usp=sharing) : This package includes captions for training, validation, and test from the Flickr Multi30k dataset, which originally
includes English, German and French datasets and is translated to Chinese, Korean and Polish and parsed.