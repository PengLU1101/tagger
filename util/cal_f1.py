import numpy as np

def evaluate_acc_(gold_y, pred_y):
    if isinstance(pred_y, list):
        pred_y = np.array(pred_y)
    else:
        pred_y = pred_y.data.long().cpu().numpy()
    gold_y = gold_y.data.long().cpu().numpy()
    rec_num = ((gold_y == pred_y) & (pred_y != 0)).sum()
    rec_den = (gold_y != 0).sum()
    prec_num = ((gold_y == pred_y) & (pred_y != 0)).sum()
    prec_den = (pred_y != 0).sum()
    
    return prec_num, prec_den, rec_num, rec_den

def evaluate_acc(gold_y, pred_y):
    
    rec_num = 0
    rec_den = 0
    prec_num = 0
    prec_den = 0
    for x, y in zip(gold_y, pred_y):
        x, y = np.array(x), np.array(y)
        rec_n = ((x == y) & (y != 0)).sum()
        rec_d = (x != 0).sum()
        prec_n = ((x == y) & (y != 0)).sum()
        prec_d = (y != 0).sum()
        rec_num += rec_n
        rec_den += rec_d
        prec_num += prec_n
        prec_den += prec_d
    #rec_num = ((gold_y == pred_y) & (pred_y != 0)).sum()
    #rec_den = (gold_y != 0).sum()
    #prec_num = ((gold_y == pred_y) & (pred_y != 0)).sum()
    #prec_den = (pred_y != 0).sum()
    
    return prec_num, prec_den, rec_num, rec_den

def eval_f1(prec_nums, prec_dens, rec_nums, rec_dens):
    prec = 0.0 if prec_dens == 0 else prec_nums/prec_dens
    rec = 0.0 if rec_dens == 0 else rec_nums/rec_dens
    f1 = 0.0 if (prec+rec) == 0 else (2*prec*rec)/(prec+rec)
    return prec, rec, f1


def eval_file(path):
    prec_nums, prec_dens, rec_nums, rec_dens = 0, 0, 0, 0
    with codecs.open(path, "r") as f:
        for line in tqdm(f.readlines()):
            if len(line) > 5:
                tmp = line[:-1].split("\t")
                if tmp[1] == tmp[2] and tmp[2] != "O":
                    rec_nums += 1
                    prec_nums += 1
                if tmp[1] != "O":
                    rec_dens += 1
                if tmp[2] != "O":
                    prec_dens += 1
    prec = 0.0 if prec_dens == 0 else prec_nums/prec_dens
    rec = 0.0 if rec_dens == 0 else rec_nums/rec_dens
    f1 = 0.0 if (prec+rec) == 0 else (2*prec*rec)/(prec+rec)

    print(prec*100, rec*100, f1*100)