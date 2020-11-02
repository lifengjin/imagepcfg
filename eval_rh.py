import logging, sys, os
from eval.evalb_unlabeled import eval_rvm_et_al, eval_postags
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', required=True)
    parser.add_argument('--gold', required=True)

    args = parser.parse_args()

    logfile_fh = open(os.path.join(os.path.dirname(args.pred), 'eval.results'), 'w')
    streamhandler = logging.StreamHandler(sys.stdout)
    filehandler = logging.StreamHandler(logfile_fh)
    handler_list = [filehandler, streamhandler]
    logging.basicConfig(level='INFO', format='%(asctime)s %(message)s',
            datefmt='%m/%d/%Y %I:%M:%S %p', handlers=handler_list)

    logging.info('Model folder is: {}'.format(os.path.basename(args.pred)))
    logging.info('Eval set is: {}'.format(args.pred))
    logging.info('Gold set is: {}'.format(args.gold))

    args = parser.parse_args()
    args = ['--gold', args.gold, '--pred', args.pred]

    print(eval_rvm_et_al(args))
    logfile_fh.close()