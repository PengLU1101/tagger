import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time, pickle
from torch.autograd import Variable

import random
import argparse
from statistics import mean

import Model
import DataLoader
import optim_custorm
#from logger import Logger
import util.cal_f1
from config import *

parser = argparse.ArgumentParser(description='multi_tagger')
parser.add_argument('--gpu', type=str, default='20', help='# of machine')
parser.add_argument('--mode', type=str, default='train', help='mode')
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
    data_holder, task2id, id2task, num_feat, num_voc, num_voc, tgt_dict, embeddings = DataLoader.multitask_dataloader(pkl_path, num_task=num_task, batch_size=BATCH_SIZE)
    para = model_para
    task2label = {"conll2000": "chunk", "unidep": "POS", "conll2003": "NER"}
    #task2label = {"conll2000": "chunk", "wsjpos": "POS", "conll2003": "NER"}
    #logger = Logger('./logs/'+str(args.gpu))
    para["id2task"] = id2task
    para["n_feats"] = num_feat
    para["n_vocs"] = num_voc
    para["n_tasks"] = num_task
    para["out_size"] = [len(tgt_dict[task2label[id2task[ids]]]) for ids in range(num_task)]
    model = Model.build_model(para)
    model.Word_embeddings.apply_weights(embeddings)

    params = model.parameters()
    num_params = sum(p.numel() for p in model.parameters())
    print(model)
    print("Num of paras:", num_params)
    print(model.concat_flag)
    model_optim = optim_custorm.NoamOpt(para["d_hid"], 1, 1000, torch.optim.Adam(params, lr=0.0, betas=(0.9, 0.98), eps=1e-9, weight_decay=L2))

    #model_optim = optim_custorm.NoamOpt(para["d_hid"], 1, 1000, torch.optim.SGD(params, lr=0.001, momentum=0.9))
    if args.mode == "train":
        best_F1 = 0
        if not para["crf"]:
            calculate_loss = nn.NLLLoss()
        else: 
            calculate_loss = None
        print("Start training...")
        print('-' * 60)
        KLLoss = nn.KLDivLoss()
        start_point = time.time()
        for epoch_idx in range(NUM_EPOCH):
            Pre, Rec, F1 = run_epoch(model, data_holder, model_optim, calculate_loss, KLLoss, para, epoch_idx, id2task)
            use_time = time.time() - start_point
            print("Time using: %f mins" %(use_time/60))
            if not best_F1 or best_F1 < F1:
                best_F1 = F1
                Model.save_model(model_path, model, para)
                print('*' * 60)
                print("Save model with average Pre: %f, Rec: %f, F1: %f on dev set." %(Pre, Rec, F1))
                save_idx = epoch_idx
                print('*' * 60)
        print("save model at epoch:", save_idx)

    else:
        para_path = os.path.join(path, 'para.pkl')
        with open(para_path, "wb") as f:
            para_save = pickle.load(f)
        model = Model.build_model(para_save)
        model = Model.read_model(model_path, model)
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
    for i in range(sum(train_lens)):
        start_time = time.time()
        idx, idx_list, count_list = sample_idx(idx_list, count_list, train_lens)
        model.zero_grad()
        src_seqs, src_masks, src_feats, tgt_seqs, tgt_masks, _,  _ = next(train_data[idx])
        src_seqs, src_masks, src_feats, tgt_seqs, tgt_masks = wrap_variable(False, src_seqs, src_masks, src_feats, tgt_seqs, tgt_masks)
        batch_size, seq_len = src_seqs.size()
        neglog, h_p, h_s = model(src_seqs, src_masks, src_feats,
               tgt_seqs, tgt_masks, idx)
        if para["crf"]:
            loss = neglog
        else:
            loss_pre = calculate_loss(neglog.view(batch_size*seq_len, -1), tgt_seqs.view(batch_size*seq_len))*(batch_size*seq_len)/torch.sum(tgt_masks) 
            l2_reg = None

            if id2task[idx] == "conll2003":
                loss_pre = loss_pre * 10
            ###
            #reg_s_p = KLLoss(torch.log(h_p), h_s)
            #print("reg_s_p", reg_s_p)
            #print("loss", loss_pre)
            loss = loss_pre #- 0.000001*reg_s_p
            ###
            '''
            for W in model.parameters():
                if l2_reg is None:
                    l2_reg = W.norm(2)
                else:
                    l2_reg = l2_reg + W.norm(2)
            '''
            #if id2task[idx] == "unidep":

                #loss = loss * 2 #+ .000002 * l2_reg
            ####loss has problem
        loss.backward()
        total_loss += loss.detach()

        torch.nn.utils.clip_grad_norm(model.parameters(), CLIP)
        model_optim.step()
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

    return mean(prec_list_dev), mean(rec_list_dev), mean(f1_list_dev)

def infer(model, data_holder, name):
    model.eval()
    dev_data, _ = extract_data(data_holder, name)
    prf_list = []
    for i, task in enumerate(dev_data):
        confusion_list = []
        for idx in range(len(task)):
            src_seqs, src_masks, src_feats, tgt_seqs, tgt_masks, _, _ = next(task)
            src_seqs, src_masks, src_feats, tgt_seqs, tgt_masks = wrap_variable(True, src_seqs, src_masks, src_feats, tgt_seqs, tgt_masks)
            score, preds = model.predict(src_seqs, src_masks, src_feats, tgt_seqs, tgt_masks, i)
            prec_num, prec_den, rec_num, rec_den = util.cal_f1.evaluate_acc_(tgt_seqs, preds)
            confusion_list += [prec_num, prec_den, rec_num, rec_den]

        prec, rec, f1 = util.cal_f1.eval_f1(sum(confusion_list[0::4]), sum(confusion_list[1::4]), sum(confusion_list[2::4]), sum(confusion_list[3::4]))
        prf_list += [prec, rec, f1]
    return prf_list[0::3], prf_list[1::3], prf_list[2::3]


if __name__ == '__main__':
    main()
