import argparse
import os, shutil, random, time
def parse_args(args):
    cmd = argparse.ArgumentParser(args[0], conflict_handler='resolve')
    cmd.add_argument('--seed', default=-1, type=int, help='The random seed.')
    cmd.add_argument('--device', default='cpu', type=str, help='Use id of gpu, -1 if cpu.')

    cmd.add_argument('--train_path', required=True, help='The path to the training file.')
    cmd.add_argument('--train_image_path', required=True, help='The path to the training image file.')
    cmd.add_argument('--train_portion', required=False, type=str, choices=['1', '2', '4', '10', '20'], default='1') # 1/train_portion, ie 2 means 0.5, 20 means 0.05 of the whole training set

    cmd.add_argument('--valid_path', help='The path to the development file.')
    cmd.add_argument('--valid_gold_path', help='The path to the development linetrees file.')
    cmd.add_argument('--valid_image_path', required=False, help='The path to the training image file.')

    cmd.add_argument('--test_path', help='The path to the testing file.')

    cmd.add_argument("--num_nonterminals", default=90, type=int, help="number of nonterminal categories")

    cmd.add_argument("--model", default=None, help="name of the model. used in actual path to save model")

    cmd.add_argument("--update_frequency", default=1, type=int, help="the frequency with which an update is made")

    cmd.add_argument("--batch_size", "--batch", type=int, default=16, help='the batch size.')
    cmd.add_argument("--max_epoch", type=int, default=100, help='the maximum number of iteration.')

    cmd.add_argument("--clip_grad", type=float, default=5, help='the tense of clipped grad. 0 for turn it off.')

    cmd.add_argument('--min_count', type=int, default=1, help='minimum word count.')

    cmd.add_argument('--max_vocab_size', type=int, default=10000, help='maximum vocabulary size.')

    cmd.add_argument('--eval_steps', required=False, type=int, default=20000, help='report every xx iterations.')
    cmd.add_argument('--eval_train', default=False, action='store_true', help='eval on the training set at the set intervals?')
    cmd.add_argument('--eval_start_iter', required=False, type=int, default=20000, help='the first iteration to start evaling.')
    cmd.add_argument('--reconstruction_prior_epochs', required=False, type=int, default=-1, help='the first epoch to start evaling.')

    cmd.add_argument('--logfile', default='log.txt.gz')

    cmd.add_argument('--model_type', type=str, default='word', help='word')
    cmd.add_argument('--projector_type', type=str, default='cnn', help='cnn or ff ')

    cmd.add_argument('--lr', type=float, default=5e-4, help='learning rate ')

    cmd.add_argument('--loss_type', type=str, default='mse', help='mse or batchave ')

    cmd.add_argument('--state_dim', type=int, default=64, help='state embedding size')

    cmd.add_argument('--img_dim', type=int, default=2048, help='image embedding size')
    cmd.add_argument('--image_loss_weight', type=float, default=1.0, help='image loss multiplier')
    cmd.add_argument('--reconstruction_loss_weight', type=float, default=1.0, help='reconstruction_loss_weight loss multiplier')

    cmd.add_argument('--no_encoder', default=False, action='store_true')
    cmd.add_argument('--no_structure', default=False, action='store_true')

    cmd.add_argument('--eval_patient', default=5, help='eval on the training set at the set intervals?', type=int)

    cmd.add_argument('--pretrained_imgemb', default=False, action='store_true',
                     help='the image path leads to the pretrained embs' )

    opt = cmd.parse_args(args[1:])
    # if opt.model_path is None:
    opt.model_path = os.path.join('outputs', opt.model)
    time.sleep(random.uniform(0, 5))
    for i in range(100):
        checking_path = opt.model_path+'_'+str(i)
        if not os.path.exists(checking_path):
            opt.model_path = checking_path
            break
    os.makedirs(opt.model_path)

    arg_file = os.path.join(opt.model_path, 'args.txt')
    with open(arg_file, 'w') as afh:
        print(vars(opt), file=afh)

    return opt