import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time, pickle
from torch.autograd import Variable
from torch import optim

import random
import argparse
from statistics import mean

import Model_crf_s
import DataLoader
import optim_custorm

#from logger import Logger
import util.cal_f1
from config import *

parser = argparse.ArgumentParser(description='multi_tagger')
parser.add_argument('--gpu', type=str, default='20', help='# of machine')
parser.add_argument('--mode', type=str, default='train', help='mode')
parser.add_argument('--optim', type=str, default='sgd', help='optim')
parser.add_argument('--decay', type=str, default='normal', help='decay')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout')
parser.add_argument('--weight', type=float, default=1, help='weight')
parser.add_argument('--momentum', type=float, default=.9, help='momentum')

args = parser.parse_args()

pkl_path = IO["pkl_path"]
model_path = IO["model_path"] + args.gpu

def extract_data(data_holder, name):
    data = [iter(item[name]) for item in data_holder]
    lens = [len(i) for i in data]
    return data, lens
def sample_idx(idx_list, count_list, lens_list):
    idx = random.choice(idx_list)
    count_list[idx] += 1
    if count_list[idx] == lens_list[idx]:
        idx_list.remove(idx)
    return idx, idx_list, count_list
def show_result(list1, list2, list3, id2task, logger=None, step=None):
    indicator = ["prec", "rec", "F1"]
    for i, t in enumerate(zip(list1, list2, list3)):
        print("%s prec: %f, rec: %f, F1: %f" %(id2task[i], t[0]*100, t[1]*100, t[2]*100))
        #for idx, idc in enumerate(indicator):
        #    logger.scalar_summary(id2task[i]+"_"+idc, t[idx]*100, step+1)



def main():
    data_holder, task2id, id2task, num_feat, num_voc, num_char, tgt_dict, embeddings = DataLoader.multitask_dataloader(pkl_path, num_task=num_task, batch_size=BATCH_SIZE)
    para = model_para
    task2label = {"conll2000": "chunk", "unidep": "POS", "conll2003": "NER"}
    #task2label = {"conll2000": "chunk", "wsjpos": "POS", "conll2003": "NER"}
    #logger = Logger('./logs/'+str(args.gpu))
    para["id2task"] = id2task
    para["n_feats"] = num_feat
    para["n_vocs"] = num_voc
    para["n_tasks"] = num_task
    para["out_size"] = [len(tgt_dict[task2label[id2task[ids]]]) for ids in range(num_task)]
    para["n_chars"] = num_char
    model = Model_crf_s.build_model_cnn(para)
    model.Word_embeddings.apply_weights(embeddings)


    params = list(filter(lambda p: p.requires_grad, model.parameters()))
    num_params = sum(p.numel() for p in model.parameters())
    print(model)
    print("Num of paras:", num_params)
    print(model.concat_flag)
    def lr_decay(optimizer, epoch, decay_rate=0.05, init_lr=0.15):
        lr = init_lr/(1+decay_rate*epoch)
        print(" Learning rate is set as:", lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return optimizer
    def exp_lr_decay(optimizer, epoch, decay_rate=0.05, init_lr=0.015):
        lr = init_lr * decay_rate ** epoch
        print(" Learning rate is set as:", lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return optimizer

    if args.optim == "noam":
        model_optim = optim_custorm.NoamOpt(para["d_hid"], 1, 1000, torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.98), eps=1e-9, weight_decay=L2))
        args.decay = None
    elif args.optim == "sgd":
        model_optim = optim.SGD(params, lr=0.015, momentum=args.momentum, weight_decay=1e-8)
    elif args.optim == "adam":
        model_optim = optim.Adam(params, lr=0.01, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-8)
    if args.mode == "train":
        best_F1 = 0
        if not para["crf"]:
            calculate_loss = nn.NLLLoss()
        else: 
            calculate_loss = None
            #calculate_loss = [CRFLoss_vb(len(tgt_dict[task2label[id2task[idx]]])+2, len(tgt_dict[task2label[id2task[idx]]]), len(tgt_dict[task2label[id2task[idx]]])+1) for idx in range(num_task)]
            #if USE_CUDA:
            #    for x in calculate_loss:
            #        x = x.cuda()
        print("Start training...")
        print('-' * 60)
        KLLoss = None#nn.KLDivLoss()
        start_point = time.time()
        for epoch_idx in range(NUM_EPOCH):
            if args.optim == "sgd":
                if args.decay == "exp":
                    model_optim = exp_lr_decay(model_optim, epoch_idx)
                elif args.decay == "normal":
                    model_optim = lr_decay(model_optim, epoch_idx)
            Pre, Rec, F1, loss_list = run_epoch(model, data_holder, model_optim, calculate_loss, KLLoss, para, epoch_idx, id2task)

            use_time = time.time() - start_point
            print("Time using: %f mins" %(use_time/60))
            if not best_F1 or best_F1 < F1:
                best_F1 = F1
                Model_crf_s.save_model(model_path, model, para)
                print('*' * 60)
                print("Save model with average Pre: %f, Rec: %f, F1: %f on dev set." %(Pre, Rec, F1))
                save_idx = epoch_idx
                print('*' * 60)
        print("save model at epoch:", save_idx)

    else:
        para_path = os.path.join(path, 'para.pkl')
        with open(para_path, "wb") as f:
            para_save = pickle.load(f)
        model = Model_crf_s.build_model(para_save)
        model = Model_crf_s.read_model(model_path, model)
        prec_list, rec_list, f1_list = infer(model, data_holder, "test")

def wrap_variable(flag, *args):
    return [Variable(tensor, volatile=flag).cuda() if USE_CUDA else Variable(tensor) for tensor in args]

def update_log(model, logger, loss, step):
    # 1. Log scalar values (scalar summary)
    info = { 'loss': loss.data[0]}

    for tag, value in info.items():
        logger.scalar_summary(tag, value, step+1)

    # 2. Log values and gradients of the parameters (histogram summary)
    for tag, value in model.named_parameters():
        tag = tag.replace('.', '/')
        logger.histo_summary(tag, value.data.cpu().numpy(), step+1)
        logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), step+1)


def run_epoch(model, data_holder, model_optim, calculate_loss, KLLoss, para, epoch_idx, id2task):####

    model.train()
    train_data, train_lens = extract_data(data_holder, "train")
    idx_list = [idx for idx in range(len(train_data))]
    count_list = [0 for i in range(len(train_data))]
    total_loss = 0
    loss_list = [0] * len(id2task)
    for i in range(sum(train_lens)):
        start_time = time.time()
        idx, idx_list, count_list = sample_idx(idx_list, count_list, train_lens)
        model.zero_grad()
        src_seqs, src_masks, src_feats, tgt_seqs, tgt_masks, src_chars, _ = next(train_data[idx])
        src_seqs, src_masks, src_feats, tgt_seqs, tgt_masks, src_chars = wrap_variable(False, src_seqs, src_masks, src_feats, tgt_seqs, tgt_masks, src_chars)
        batch_size, seq_len = src_seqs.size()
        neglog, h_p, h_s = model(src_seqs, src_masks, src_feats, src_chars,
               tgt_seqs, tgt_masks, idx)

        if para["crf"]:
            loss = - neglog
        else:
            loss = calculate_loss(neglog.view(batch_size*seq_len, -1), tgt_seqs.view(batch_size*seq_len))*(batch_size*seq_len)/torch.sum(tgt_masks) 
            l2_reg = None
            
        if id2task[idx] == "conll2003":
            loss = loss * args.weight
        loss_list[idx] += loss.detach().cpu().data[0]
        try:
            loss.backward()
        except:
            print("fk ")
        total_loss += loss.detach()
        torch.nn.utils.clip_grad_norm(model.parameters(), CLIP)
        lr_now = model_optim.step()

        if i % PRINT_EVERY == 0 and i:
            using_time = time.time() - start_time
            print('| ep %2d | %4d/%5d btcs | ms/btc %4.4f | loss %5.7f |' %(epoch_idx+1, i, sum(train_lens), using_time * 1000 / (PRINT_EVERY*batch_size), total_loss/PRINT_EVERY))
            #update_log(model, logger, total_loss, i)
            total_loss = 0

    prec_list_dev, rec_list_dev, f1_list_dev = infer(model, data_holder, "dev")
    prec_list_test, rec_list_test, f1_list_test = infer(model, data_holder, "test")

    print('-' * 60)
    print("On dev set:")
    show_result(prec_list_dev, rec_list_dev, f1_list_dev, para["id2task"])
    print("On test set:")
    show_result(prec_list_test, rec_list_test, f1_list_test, para["id2task"])
    if args.optim == "noam":
        print("learning rate is ", lr_now)
    for i, x in enumerate(loss_list):
        print(id2task[i] + " loss: ", x)
    print("total loss on train set: ", sum(loss_list) / len(train_data))

    return mean(prec_list_dev), mean(rec_list_dev), mean(f1_list_dev), loss_list

#def infer(model, packer_list, evaluator_list, data_holder, name):

def infer(model, data_holder, name):
    model.eval()
    dev_data, _ = extract_data(data_holder, name)
    prf_list = []
    for i, task in enumerate(dev_data):
        confusion_list = []
        for idx in range(len(task)):
            src_seqs, src_masks, src_feats, tgt_seqs, tgt_masks, src_chars, tgt_list = next(task)
            src_seqs, src_masks, src_feats, tgt_seqs, tgt_masks, src_chars = wrap_variable(True, src_seqs, src_masks, src_feats, tgt_seqs, tgt_masks, src_chars)
            preds = model.predict(src_seqs, src_masks, src_feats, src_chars,tgt_seqs, tgt_masks, i)
            if CRF_FLAG:
                prec_num, prec_den, rec_num, rec_den = util.cal_f1.evaluate_acc(tgt_list, preds)
            else:
                prec_num, prec_den, rec_num, rec_den = util.cal_f1.evaluate_acc_(tgt_seqs, preds)
            confusion_list += [prec_num, prec_den, rec_num, rec_den]

        prec, rec, f1 = util.cal_f1.eval_f1(sum(confusion_list[0::4]), sum(confusion_list[1::4]), sum(confusion_list[2::4]), sum(confusion_list[3::4]))
        prf_list += [prec, rec, f1]
    return prf_list[0::3], prf_list[1::3], prf_list[2::3]


if __name__ == '__main__':
    main()
