import argparse

parser = argparse.ArgumentParser(description='Hyper-parameters management')

# Hardware options
parser.add_argument('--n_threads', type=int, default=6,help='number of threads for data loading')
parser.add_argument('--cuda', action='store_true',help='use cpu only')
parser.add_argument('--gpu_id', type=list,default=[0,1], help='use cpu only')
parser.add_argument('--seed', type=int, default=2021, help='random seed')
parser.add_argument('--num_workers', type=int, default=3, help='num worker')
parser.add_argument('--mini_data',type=int,default=200,help=" mini data for train and val, if value set to -1, it will load full dataset")
# Preprocess parameter
#f segmenting only the liver, set the label to 2 (binary segmentation). If segmenting both the liver and tumors, 
#set the label to 3 (multiclass segmentation)
parser.add_argument('--n_labels', type=int, default=2,help='number of classes') 
parser.add_argument('--upper', type=int, default=400, help='')
parser.add_argument('--lower', type=int, default=0, help='')
parser.add_argument('--norm_factor', type=float, default=200.0, help='')
parser.add_argument('--expand_slice', type=int, default=20, help='')
parser.add_argument('--min_slices', type=int, default=48, help='')
parser.add_argument('--xy_down_scale', type=float, default=0.5, help='')
parser.add_argument('--slice_down_scale', type=float, default=1.0, help='')
parser.add_argument('--valid_rate', type=float, default=0.2, help='')

# data in/out and dataset
# _dataset-path seem to be tge fixed part
parser.add_argument('--ori_train_path',default = '/root/data/liver/train',help='original trainset root path')
parser.add_argument('--fix_train_path',default = '/root/data/liver/old_fix_train',help='fix trainset root path')
parser.add_argument('--ori_test_path',default = '/root/data/liver/train',help='ori testset root path')
parser.add_argument('--test_data_path',default = '/ssd/lzq/dataset/LiTS/test',help='Testset path')
parser.add_argument('--save',default='ResUNet',help='save path of trained model')
parser.add_argument('--batch_size', type=list, default=1,help='batch size of trainset')
parser.add_argument('--split',type=float,default=0.1,help="train val split")
# train
parser.add_argument('--epochs', type=int, default=5, metavar='N',help='number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',help='learning rate (default: 0.0001)')
parser.add_argument('--early-stop', default=30, type=int, help='early stopping (default: 30)')
parser.add_argument('--crop_size', type=int, default=48)
parser.add_argument('--val_crop_max_size', type=int, default=96)
parser.add_argument('--model',default="unet", help="training model")
# test
parser.add_argument('--test_cut_size', type=int, default=48, help='size of sliding window')
parser.add_argument('--test_cut_stride', type=int, default=24, help='stride of sliding window')
parser.add_argument('--postprocess', type=bool, default=False, help='post process')


args = parser.parse_args()

