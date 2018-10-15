import argparse
parser = argparse.ArgumentParser(description='H_summary')
parser.add_argument('--gpu', type=str, default='20', help='# of machine')
parser.add_argument('--mode', type=str, default='train', help='mode')
parser.add_argument('--model_path', type=str, default='./models/', help='model path')
parser.add_argument('--pkl_path', type=str, default='./pkl/data.pkl', help='pkl file path')


parser.add_argument('--L2', type=float, default=0, help='weight decay')
parser.add_argument('--max_epoch', type=int, default=20, help='max_epoch')
parser.add_argument('--clip', type=float, default=1, help='clip')
parser.add_argument('--print_every', type=int, default=100, help='print_every')


parser.add_argument('--batch_size', type=int, default=8, help='batch_size')
parser.add_argument('--d_in', type=int, default=20, help='input_size')
parser.add_argument('--d_hid', type=int, default=20, help='hidden_size')
parser.add_argument('--dropout', type=float, default=20, help='dropout')
parser.add_argument('--d_emb', type=int, default=300, help='embedding size')
parser.add_argument('--n_layers', type=int, default=1, help='num of layers')



args = parser.parse_args()