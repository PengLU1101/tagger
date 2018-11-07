#train.config
import torch
PRINT_EVERY = 300
NUM_EPOCH = 30
PATIENCE = 5
BATCH_SIZE = 10
L2 = 0
num_task = 3
USE_CUDA = torch.cuda.is_available()
CLIP = 5
CRF_FLAG = False

model_para = {
	"d_emb": 100,
	"d_hid": 256*2*2, # 19: 256*2*2 # other 256*3*2
	"d_feat": 10,
	"n_layers": 2,
	"dropout": 0.50,
	"crf": CRF_FLAG,
    "concat_flag": True,
    #"n_chars": 
    "d_char_emb": 16,
    "d_char": 64*2,
    "kernel_size": 3,
    "padding": 1,
    "use_elmo": True
}


IO = {
	"model_path": "./models/models",
	"pkl_path": '/u/lupeng/Project/emnlp2017-bilstm-cnn-crf/pkl/conll2000_ontonotes_wsjpos_glove.6B.100d.pkl',
    #"pkl_path": '/u/lupeng/Project/emnlp2017-bilstm-cnn-crf/pkl/conll2003_glove.6B.100d.pkl',
	"raw_file_dir": "./data/"
}
"""
model_para = {
    "d_emb": 100,
    "d_hid": 200,
    "d_feat": 10,
    "n_layers": 2,
    "dropout": 0.50,
    "crf": True,
    "concat_flag": True,
    #"n_chars": 
    "d_char_emb": 50,
    "d_char": 50,
    "kernel_size": 3,
    "padding": 1
}

IO = {"model_path": "./models/models", "pkl_path": '/u/lupeng/Project/emnlp2017-bilstm-cnn-crf/pkl/conll2000_conll2003_unidep_glove.6B.100d.pkl', "raw_file_dir": "./data/"
}

"""

datasets = {
    'unidep':
        {'columns': {1:'tokens', 3:'unidep_POS'},
         'label': 'POS',
         'evaluate': True,
         'commentSymbol': None},
    'conll2000':
        {'columns': {0:'tokens', 2:'conll2000_chunk_BIO'},
         'label': 'chunk_BIO',
         'evaluate': True,
         'commentSymbol': None},
    'conll2003':                                  
        {'columns': {0:'tokens', 1:'conll2003_NER_BIO'},    
         'label': 'NER_BIO',                      
         'evaluate': True,                        
         'commentSymbol': None}                  
                   
}
