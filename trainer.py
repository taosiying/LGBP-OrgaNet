import logging
import os
import random
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss, test_single_volume
from torchvision import transforms
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import torch.nn.functional as F


from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
from datasets.organic import Organic_dataset
from datasets.ISIC2017 import ISIC_dataset
from datasets.Brainorganoids import Brainorganoid_dataset
from datasets.vessel import Vessel_dataset
from datasets.grayorganoid import Grayorganoid_dataset
from datasets.organoidgray import Organoidgray_dataset
from datasets.realimage import Realimage_dataset
from datasets.organoid_frames import Organoidframe_dataset

#Isoperimetric Quotient Loss
class IsoperimetricLoss(nn.Module):
    def __init__(self, alpha=0.001, gamma=0.001, reduction='mean'):
        super(IsoperimetricLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs):
        precict = torch.softmax(inputs, dim=1).squeeze(0)
        padded_x = F.pad(inputs, (0, 1, 0, 0), mode='constant', value=0)
        padded_y = F.pad(inputs, (0, 0, 0, 1), mode='constant', value=0)
        # print('inputs.shape:',inputs.shape,'padded_x.shape',padded_x.shape,'padded_y.shape',padded_y.shape)
        gradient_x = torch.abs(padded_x[:, :, :, :-1] - padded_x[:, :, :, 1:])

        # print('gradient_x.shape',gradient_x.shape,'precict.shape',precict.shape)
        gradient_y = torch.abs(padded_y[:, :, :-1, :] - padded_y[:, :, 1:, :])*precict
        boundary = torch.sum(gradient_x**2 + gradient_y**2) + self.alpha
        area = torch.sum(precict)
        # print('boundary:',boundary,'area:',area)
        isoperimetric_loss = boundary/(4 * torch.pi * area) + self.gamma
        if self.reduction == 'mean':
            return isoperimetric_loss.mean()
        elif self.reduction == 'sum':
            return isoperimetric_loss.sum()
        else:
            return isoperimetric_loss


#适用于二分类的focal loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, labels):
        preds = torch.softmax(preds,dim=1)
        labels_one_hot = torch.zeros_like(preds)
        labels_one_hot.scatter_(1, labels.unsqueeze(1), 1)
        cross_entropy_loss = F.binary_cross_entropy(preds, labels_one_hot, reduction='none')
        alpha_factor = torch.where(labels == 1, self.alpha, 1 - self.alpha)
        alpha_factor = alpha_factor.unsqueeze(1)
        focal_weight = alpha_factor * (1 - preds) ** self.gamma
        focal_loss = focal_weight * cross_entropy_loss
        focal_loss = torch.mean(focal_loss)
        return focal_loss


def inference(model, testloader, args, test_save_path=None):
    model.eval()
    metric_list = 0.0

    for i_batch, sampled_batch in enumerate(testloader):
        h, w = sampled_batch["image"].size()[2:]
        # image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        image, label = sampled_batch["image"], sampled_batch["label"]
        # metric_i = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
        #                               test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)
        metric_i = test_single_volume(image, label, model, classes=args.num_classes,
                                      patch_size=[args.img_size, args.img_size])
        metric_list += np.array(metric_i)
        logging.info(' idx %d mean_dice %f mean_hd95 %f' % (i_batch, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
    
    metric_list = metric_list / len(testloader.dataset)
    
    for i in range(1, args.num_classes):
        logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
    
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    
    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    
    return performance, mean_hd95


def plot_result(dice, h, snapshot_path,args):
    dict = {'mean_dice': dice, 'mean_hd95': h} 
    df = pd.DataFrame(dict)
    plt.figure(0)
    df['mean_dice'].plot()
    resolution_value = 1200
    plt.title('Mean Dice')
    date_and_time = datetime.datetime.now().date()
    filename = f'{args.model_name}_' + str(date_and_time)+'dice'+'.png'
    save_mode_path = os.path.join(snapshot_path, filename)
    plt.savefig(save_mode_path, format="png", dpi=resolution_value)
    plt.figure(1)
    df['mean_hd95'].plot()
    plt.title('Mean hd95')
    filename = f'{args.model_name}_' + str(date_and_time)+'hd95'+'.png'
    save_mode_path = os.path.join(snapshot_path, filename)
    #save csv 
    filename = f'{args.model_name}_' + str(date_and_time)+'results'+'.csv'
    save_mode_path = os.path.join(snapshot_path, filename)
    df.to_csv(save_mode_path, sep=',')


class totle_loss(nn.Module):
    def __init__(self,num_classes = 2):
        super(totle_loss,self).__init__()
        self.ce_loss = CrossEntropyLoss()
        self.focal_loss = FocalLoss()
        self.dice_loss = DiceLoss(num_classes)
        self.iq_loss = IsoperimetricLoss()
    def forward(self,outputs,label):
        loss_focal = self.focal_loss(outputs, label[:].long())
        loss_ce = self.ce_loss(outputs, label[:].long())
        loss_dice = self.dice_loss(outputs, label, softmax=True)
        loss_iq = self.iq_loss(outputs)
        # loss = 0.3 * loss_focal + 0.5 * loss_dice + 0.2 * loss_iq
        loss = 0.5 * loss_ce + 0.5 * loss_dice
        return loss
def setup_seed(seed=42, deterministic=True):
    """
    固定随机种子 + 确保确定性
    """
    # Python & Numpy
    random.seed(seed)
    np.random.seed(seed)

    # PyTorch CPU / GPU
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # CuDNN
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False  # 确保每次卷积选择相同算法

    if deterministic:
        torch.use_deterministic_algorithms(True)

        # 设置 cuBLAS 环境变量 (必须在 import torch 前设置也行，这里保险写一次)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        # 或者 os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    print(f"[Init] Random seed set to {seed}, deterministic={deterministic}")
def trainer(args, model, snapshot_path):
    date_and_time = datetime.datetime.now()
    date_and_time = date_and_time.strftime("%Y-%m-%d-%H-%M-%S").replace(" ", "")

    os.makedirs(os.path.join(snapshot_path, 'test'), exist_ok=True)
    test_save_path = os.path.join(snapshot_path, 'test')
    
    # Save logs
    logging.basicConfig(filename=snapshot_path + args.model_name + str(date_and_time) + "_log.txt", level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu


    db_train = Organic_dataset(args,split="train")
    db_test = Organic_dataset(args,split="val")
    # db_train = ISIC_dataset(args,split="train")
    # db_test = ISIC_dataset(args,split="val")
    Loss = totle_loss()
    # db_train = Brainorganoid_dataset(args, split="train")
    # db_test = Brainorganoid_dataset(args, split="val")


    # db_train = Grayorganoid_dataset(args, split="train")
    # db_test = Grayorganoid_dataset(args, split="val")
    # db_train = Organoidgray_dataset(args, split="train")
    # db_test = Organoidgray_dataset(args, split="val")
    # db_train = Vessel_dataset(args,split="train")
    # db_test = Vessel_dataset(args,split="val")
    # db_train = Realimage_dataset(args, split="train")
    # db_test = Realimage_dataset(args,split="val")
    # db_train = Organoidframe_dataset(args, split="train")
    # db_test = Organoidframe_dataset(args,split="val")
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=0)
    print('testloader:',len(testloader))

    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    # optimizer = optim.SGD(model.parameters(), lr=base_lr,momentum=0.9, weight_decay=0.0001)
    optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=0.0001)

    writer = SummaryWriter('results/hiformer-s/log')

    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader) 
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))


    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)    
    dice_=[]
    hd95_= []
    # setup_seed(42,False)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']

            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            B, C, H, W = image_batch.shape
            image_batch = image_batch.expand(B, 3, H, W)

            outputs = model(image_batch)
            # print("outputs:",outputs.shape,"label:",label_batch.shape)
            loss = Loss(outputs,label_batch)
            # loss_focal = focal_loss(outputs,label_batch[:].long())
            # # loss_ce = ce_loss(outputs, label_batch[:].long())
            # loss_dice = dice_loss(outputs, label_batch, softmax=True)
            # loss_iq = iq_loss(outputs)
            # loss = 0.4 * loss_focal + 0.6 * loss_dice
            # loss = 0.3 * loss_focal + 0.5 * loss_dice + 0.2 * loss_iq
            # logging.info("loss_focal:",loss_focal,"loss_dice:",loss_dice,'loss_iq:',loss_iq,'loss:',loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)

            # logging.info('iteration %d : loss : %f, loss_ce: %f loss_dice: %f' % (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))

            try:
                if iter_num % 10 == 0:
                    image = image_batch[1, 0:1, :, :]
                    image = (image - image.min()) / (image.max() - image.min())
                    writer.add_image('train/Image', image, iter_num)
                    outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                    writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                    labs = label_batch[1, ...].unsqueeze(0) * 50
                    writer.add_image('train/GroundTruth', labs, iter_num)
            except: pass
        
        # Test
        if (epoch_num + 1) % args.eval_interval == 0:
            logging.info("*" * 20)
            logging.info(f"Running Inference after epoch {epoch_num}")
            print(f"Epoch {epoch_num}")
            mean_dice, mean_hd95 = inference(model, testloader, args, test_save_path=test_save_path)
            dice_.append(mean_dice)
            hd95_.append(mean_hd95)
            model.train()
        if (epoch_num + 1) % (args.eval_interval * 10) == 0:
            filename = f'{args.model_name}_epoch_{epoch_num}.pth'
            save_mode_path = os.path.join(snapshot_path, filename)
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            filename = f'{args.model_name}_epoch_{epoch_num}.pth'
            save_mode_path = os.path.join(snapshot_path, filename)
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

            if not (epoch_num + 1) % args.eval_interval == 0:
                logging.info("*" * 20)
                logging.info(f"Running Inference after epoch {epoch_num} (Last Epoch)")
                print(f"Epoch {epoch_num}, Last Epcoh")
                mean_dice, mean_hd95 = inference(model, testloader, args, test_save_path=test_save_path)
                dice_.append(mean_dice)
                hd95_.append(mean_hd95)
                model.train()

            iterator.close()
            break
            
    plot_result(dice_,hd95_,snapshot_path,args)
    writer.close()
    return "Training Finished!"