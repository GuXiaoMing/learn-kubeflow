import argparse
import datetime
import torch
import os

def get_args():
    """
    获取参数
    :return: 模型参数
    """
    parser = argparse.ArgumentParser(description='CNN text classificer')
    # learning
    parser.add_argument('-train-file', type=str, default='SST/train.tsv',
                        help='the train file path')
    parser.add_argument('-test-file', type=str, default='SST/dev.tsv',
                        help='the test file path')
    parser.add_argument('-predict-file', type=str, default='data/test.csv',
                        help='the predict file path')
    parser.add_argument('-lr', type=float, default=0.0005,
                        help='initial learning rate [default: 0.001]')
    parser.add_argument('-l2', type=float, default=0.,
                        help='initial l2 regularization [default: 0.]')
    parser.add_argument('-epochs', type=int, default= 1,
                        help='number of epochs for train [default: 256]')
    parser.add_argument('-batch-size', type=int, default=128,
                        help='batch size for training [default: 64]')
    parser.add_argument('-max-len', type=int, default=4096,
                        help='max len of sentence for training [default: 1024]')
    parser.add_argument('-log-interval', type=int, default=10,
                        help='how many steps to wait before logging training status [default: 10]')
    parser.add_argument('-test-interval', type=int, default=100,
                        help='how many steps to wait before testing [default: 100]')
    parser.add_argument('-save-interval', type=int, default=500,
                        help='how many steps to wait before saving [default:500]')
    parser.add_argument('-save-dir', type=str, default='snapshot',
                        help='where to save the snapshot')
    parser.add_argument('-log-dir', type=str, default='logs',
                        help='where to save the log [tensorboard]')
    parser.add_argument('-early-stop', type=int, default=300,
                        help='iteration numbers to stop without performance increasing')
    parser.add_argument('-save-best', type=bool, default=True,
                        help='whether to save when get best performance')
    # data
    parser.add_argument('-test-ratio', type=float, default=0.3,
                        help='test data ratio')
    parser.add_argument('-shuffle', action='store_true', default=True,
                        help='shuffle the data every epoch')
    # model
    parser.add_argument('-dropout', type=float, default=0.5,
                        help='the probability for dropout [default: 0.5]')
    parser.add_argument('-embed-dim', type=int, default=300,
                        help='number of embedding dimension [default: 128]')

    parser.add_argument('-hidden-dim', type=int, default=128,
                        help='hidden dimension [default: 128]')
    parser.add_argument('-kernel-num', type=int, default=256,
                        help='number of each kind of kernel')
    parser.add_argument('-kernel-sizes', type=str, default='2,4,7',
                        help='comma-separated kernel size to use for convolution')

    parser.add_argument('-static', action='store_true', default=False,
                        help='fix the embedding')
    # device
    parser.add_argument('-device', type=int, default=0,
                        help='device to use for iterate data, -1 mean cpu [default: -1]')
    parser.add_argument('-cuda', action='store_true', default=True,
                        help='disable the gpu')
    # option
    parser.add_argument('-snapshot', type=str, default=None,#'snapshot/2019-06-17_10-33-35/best_steps_3500.pt'
                        help='filename of model snapshot [default: None]')
    parser.add_argument('-dict', type=bool, default=False,  # './snapshot/2018-06-25_15-51-07/best_steps_800.pt',
                        help='dict [default: False]')
    parser.add_argument('-vocab', type=str, default='./vocab/',
                        help='vocab for predict the sentence')
    parser.add_argument('-trained-model', type=str, default='./model/',
                        help='trained model')
    parser.add_argument('-label-column', type=str, default='label',
                        help='Select the column that contains the label or outcome column [default: 0]')
    parser.add_argument('-text-column', type=str, default='text',
                        help='Select the column that contains the Text or input column [default: 2]')
    parser.add_argument('-predict-result-file', type=str, default='predict',  # 'snapshot/2019-06-17_10-33-35/best_steps_3500.pt'
                        help='filename of predict result [default: None]')
    parser.add_argument('-logs', type=str, default='logs',
                        help='logs [default: None]')

    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    now_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # args.save_dir = os.path.join(args.save_dir, now_time)
    args.log_dir = os.path.join(args.log_dir, now_time)
    return args

def print_parameters(args):
    """
    print model parameters
    :param args: model parameters
    :return: None
    """
    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))

    # - --word2id
    # - outputPath: Word to id file
    # - --id2word
    # - outputPath: Id to word file
    # - --label2id
    # - outputPath: Label to id file
    # - --id2label
    # - outputPath: Id to label file
