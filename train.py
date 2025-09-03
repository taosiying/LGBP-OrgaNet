import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from models.HiFormer import HiFormer
import configs.HiFormer_configs as configs 
from trainer2 import trainer
from models.LGBP_OrgaNet import LGBPOrga
from configs.SROrga_configs import get_SROrga_configs
from configs.LGBPOrga_configs import get_LGBPOrga_configs
from models.yolo import YOLO11n


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='./data/Synapse/train_npz', help='root dir for data')
parser.add_argument('--test_path', type=str,
                    default='./data/Synapse/test_vol_h5', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='organic', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=101, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=10, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--num_workers', type=int,  default=0,
                    help='number of workers')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--output_dir', type=str,
                    default='./results', help='root dir for output log')
parser.add_argument('--model_name', type=str,
                    default='SROrga')
parser.add_argument('--eval_interval', type=int,
                    default=20, help='evaluation epoch')
parser.add_argument('--z_spacing', type=int,
                    default=1, help='z_spacing')
parser.add_argument('--base-size', type=int, default=224,
                        help='base image size')
parser.add_argument('--crop-size', type=int, default=224,
                        help='crop image size')

args = parser.parse_args()

args.output_dir = args.output_dir + f'/{args.model_name}'
os.makedirs(args.output_dir, exist_ok=True)


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


    CONFIGS = {
        'hiformer-s': configs.get_hiformer_s_configs(),
        'hiformer-b': configs.get_hiformer_b_configs(),
        'hiformer-l': configs.get_hiformer_l_configs(),
    }

    if args.batch_size != 24 and args.batch_size % 6 == 0:
        args.base_lr *= args.batch_size / 24
    configs = get_LGBPOrga_configs()

    # model = HiFormer(config=CONFIGS[args.model_name], img_size=args.img_size, n_classes=args.num_classes).cuda()
    model = LGBPOrga(config=configs, img_size=args.img_size, n_classes=args.num_classes).cuda()
    # model = YOLO11n(nc=2).cuda()
    # state_dict = torch.load('results/SROrga/SROrga_organic_best_0.928.pth')
    # model.load_state_dict(state_dict)
    # torch.autograd.set_detect_anomaly(True)
    trainer(args, model, args.output_dir)
