import pyarrow.parquet as pq
import pickle
import sys
from TextCNN import TextCNN as Model
import torch.nn.functional as F
from torch.autograd import Variable
from args_util import *
from data_util import *
import logging

class Trainer():
    def __init__(self, args):
        self.args = args
        if not self.args.dict:
            logging.info('processing data...')
            self.word2id, self.id2word, self.label2id, self.id2label, self.max_len \
                = process_data(self.args,self.args.train_file)
            if not os.path.isdir(self.args.save_dir):
                os.makedirs(self.args.save_dir)
            with open(self.args.save_dir + '/' + 'word2id.pkl', 'wb') as f:
                pickle.dump(self.word2id, f)
            with open(self.args.save_dir + '/' + 'label2id.pkl', 'wb') as f:
                pickle.dump(self.label2id, f)
            with open(self.args.save_dir + '/' + 'id2label.pkl', 'wb') as f:
                pickle.dump(self.id2label, f)

        else:
            with open(self.args.save_dir + '/' + 'word2id.pkl', 'rb') as f:
                self.word2id = pickle.load(f)
            with open(self.args.save_dir + '/' + 'id2label.pkl', 'rb') as f:
                self.id2label = pickle.load(f)
            with open(self.args.save_dir + '/' + 'label2id.pkl', 'rb') as f:
                self.label2id = pickle.load(f)
        self.train_iter, self.test_iter = load_data(self.args.train_file, self.args.test_file, self.word2id, self.label2id, self.args)
        self.args.embed_num = len(self.word2id)
        self.args.class_num = len(self.id2label)
        self.model = Model(self.args)

        if self.args.snapshot is not None:
            logging.info('\nLoading model from %s...' % (self.args.snapshot))
            self.model.load_state_dict(torch.load(self.args.snapshot))

        if torch.cuda.is_available():
            print("GPU.............")
            torch.cuda.set_device(self.args.device)
            self.model = self.model.cuda()

    def train(self):
        if torch.cuda.is_available():
            self.model.cuda()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)

        step, best_acc, last_step = 0, 0, 0
        train_loss, train_acc = 0., 0.
        self.model.train()
        flag = 0
        for epoch in range(1, self.args.epochs+1):
            if flag == 1: break
            for feature, target in self.train_iter:
                feature = Variable(feature)
                target = Variable(target)
                if self.args.cuda:
                    feature, target = feature.cuda(), target.cuda()
                optimizer.zero_grad()
                logit = self.model(feature)

                loss = F.cross_entropy(logit, target)
                l2_reg = 0.
                for param in self.model.parameters():
                    l2_reg += torch.norm(param)
                loss += self.args.l2 * l2_reg
                loss.backward()
                optimizer.step()
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                accuracy = 100.0 * corrects / self.args.batch_size
                step += 1
                train_loss += loss.data
                train_acc += accuracy
                sys.stdout.write('\rEpoch[%d] Step[%d] - loss: %f  acc: %f%% (%d/%d)'
                                 % (epoch, step, loss.data, accuracy, corrects, self.args.batch_size))
                if step % self.args.log_interval == 0:
                    train_loss /= self.args.log_interval
                    train_acc /= self.args.log_interval
                    train_loss, train_acc = 0., 0.

                if step % self.args.test_interval == 0:
                    logging.info('Epoch[%d] Step[%d] - loss: %f  acc: %f%% (%d/%d)'
                                 % (epoch, step, loss.data, accuracy, corrects, self.args.batch_size))
                    test_loss, test_acc = self.eval()

                    self.model.train()
                    if test_acc > best_acc:
                        best_acc, last_step = test_acc, step
                        if self.args.save_best:
                            self.save('best', step)
                            continue
                    else:
                        if step - last_step >= self.args.early_stop:
                            logging.info('\nEarly stop by %d steps.'% (self.args.early_stop))
                            flag = 1
                            break
                if step % self.args.save_interval == 0:
                    self.save('last', step)

    def eval(self):
        self.model.eval()
        corrects, avg_loss = 0, 0
        for feature, target in self.test_iter:
            feature = Variable(feature)
            target = Variable(target)
            if self.args.cuda:
                feature, target = feature.cuda(), target.cuda()

            logit = self.model(feature)
            loss = F.cross_entropy(logit, target, size_average=False)

            avg_loss += loss.data
            corrects += (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()

        size = len(self.test_iter.dataset)
        avg_loss /= size
        accuracy = 100.0 * corrects / size
        logging.info('Evaluation - loss: %f  acc: %f (%d/%d)\n' % (avg_loss, accuracy, corrects, size))
        sys.stdout.write('\rEvaluation - loss: %f  acc: %f (%d/%d)\n' % (avg_loss, accuracy, corrects, size))
        return avg_loss, accuracy

    def save(self, save_prefix, steps):
        if not os.path.isdir(self.args.save_dir):
            os.makedirs(self.args.save_dir)

        save_prefix = os.path.join(self.args.save_dir, save_prefix)
        save_path = '%s_steps_%d.pt' % (save_prefix, 100)
        torch.save(self.model.state_dict(), save_path)

if __name__ == '__main__':
    args = get_args()
    if not os.path.isdir(args.logs):
        os.makedirs(args.logs)
    logname = args.logs + "/train.log"
    logging.basicConfig(filename=logname, filemode='w', level=logging.DEBUG)

    trainer = Trainer(args)
    with open(args.save_dir + '/' + 'config.pkl', 'wb') as f:
        pickle.dump(args, f)
    try:
        trainer.train()
    except KeyboardInterrupt:
        logging.warning('\nExiting from training early')

