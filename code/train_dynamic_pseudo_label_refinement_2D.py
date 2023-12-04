import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from dataloaders import utils
from dataloaders.dataset import (BaseDataSets, RandomGenerator,
                                 TwoStreamBatchSampler)
from networks.net_factory import net_factory
from utils import losses, metrics, ramps
from val_2D import test_single_volume

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/Dynamic_Pseudo_Label_Refinement', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=60000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[256, 256],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=12,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=7,
                    help='labeled data')
# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
parser.add_argument('--loss_weights', nargs='+', type=float, 
                    default=[0.14, 0.33, 0.35, 0.18], help='Weights for the loss function')
parser.add_argument('--thr', nargs=2, type=float, default=[0.48, 0.53], help='Threshold')
args = parser.parse_args()

def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model

def xavier_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model

def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "140": 1312}
    elif "Prostate":
        ref_dict = {"2": 27, "4": 53, "8": 120,
                    "12": 179, "16": 256, "21": 312, "42": 623}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def get_current_threshold1_weight(epoch):
    return 0.12 * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def get_current_threshold2_weight(epoch):
    return 0.07 * ramps.sigmoid_rampup(epoch, args.consistency_rampup)
    
    
def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    thr1 = args.thr[0]
    thr2 = args.thr[1]
    ignore_value = 6

    def create_model(ema=False):
        # Network definition
        model = net_factory(net_type=args.model, in_chns=1,
                            class_num=num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model1 = create_model()
    model2 = create_model()
    
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path, split="train", num=None, transform=transforms.Compose([
        RandomGenerator(args.patch_size)]), snapshot_path=snapshot_path)
    db_val = BaseDataSets(base_dir=args.root_path, split="val")

    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labeled_num)
    print("Total slices is: {}, labeled slices is: {}".format(
        total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    model1.train()
    model2.train()

    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)

    optimizer1 = optim.SGD(model1.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    optimizer2 = optim.SGD(model2.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    mse_loss_v = nn.MSELoss(reduction='none')
    weights = torch.tensor(args.loss_weights).cuda()
    ce_loss_i = CrossEntropyLoss(ignore_index=ignore_value, weight = weights)
    ce_loss = CrossEntropyLoss(weight = weights)
    dice_loss = losses.DiceLoss(num_classes)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance1 = 0.0
    best_performance2 = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            pseudo_label1_batch, pseudo_label2_batch = sampled_batch["pseudo_label1"][:][args.labeled_bs:].cuda(), sampled_batch["pseudo_label2"][:][args.labeled_bs:].cuda()

            outputs1  = model1(volume_batch)
            outputs_soft1 = torch.softmax(outputs1, dim=1)

            outputs2 = model2(volume_batch)
            outputs_soft2 = torch.softmax(outputs2, dim=1)
            consistency_weight = get_current_consistency_weight(iter_num // 150)
            
            #supervision loss
            sup_loss1 = 0.5 * (ce_loss(outputs1[:args.labeled_bs], label_batch[:][:args.labeled_bs].long()) + dice_loss(
                outputs_soft1[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1)))
            sup_loss2 = 0.5 * (ce_loss(outputs2[:args.labeled_bs], label_batch[:][:args.labeled_bs].long()) + dice_loss(
                outputs_soft2[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1)))
            
            # unsupervision        
            pseudo_outputs1 = torch.argmax(outputs_soft1[args.labeled_bs:].detach(), dim=1, keepdim=False)
            pseudo_outputs2 = torch.argmax(outputs_soft2[args.labeled_bs:].detach(), dim=1, keepdim=False)
            
            threshold1_weight = get_current_threshold1_weight(iter_num // 150)
            threshold2_weight = get_current_threshold2_weight(iter_num // 150)
            threshold1 = torch.tensor(thr1 + threshold1_weight)
            threshold2 = torch.tensor(thr2 + threshold2_weight)

            pseudo_label_mask1 = (pseudo_label1_batch != ignore_value).bool()
            pseudo_label_mask2 = (pseudo_label2_batch != ignore_value).bool()
            equal_mask = (pseudo_label1_batch == pseudo_label2_batch) & pseudo_label_mask1 & pseudo_label_mask2
            inequal_mask = (pseudo_label1_batch != pseudo_label2_batch) & pseudo_label_mask1 & pseudo_label_mask2
            only_label1_mask = (pseudo_label1_batch != ignore_value) & (pseudo_label2_batch == ignore_value)
            only_label2_mask = (pseudo_label1_batch == ignore_value) & (pseudo_label2_batch != ignore_value)
            
            # Correct the areas with pseudo-labels in the pseudo-labeled regions
            if (torch.any(pseudo_label_mask1) or torch.any(pseudo_label_mask2)) and iter_num % 5 == 0:
                
                if torch.any(equal_mask):
                    pseudo_label_values = pseudo_label1_batch * equal_mask

                    outputs_soft1_values = torch.zeros_like(pseudo_label_values, dtype=torch.float32)
                    outputs_soft2_values = torch.zeros_like(pseudo_label_values, dtype=torch.float32)
                    for i in range(0, num_classes):
                        outputs_soft1_i = outputs_soft1[args.labeled_bs:].detach()[:, i, :, :]
                        outputs_soft2_i = outputs_soft2[args.labeled_bs:].detach()[:, i, :, :]
                        temp_condition = (pseudo_label_values == i)
                        outputs_soft1_values = torch.where(temp_condition, outputs_soft1_i, outputs_soft1_values)
                        outputs_soft2_values = torch.where(temp_condition, outputs_soft2_i, outputs_soft2_values)

                    remove_mask = ((outputs_soft1_values <= threshold1) | (outputs_soft2_values <= threshold2)) & equal_mask
                    pseudo_label1_batch[remove_mask] = ignore_value
                    pseudo_label2_batch[remove_mask] = ignore_value

                if torch.any(inequal_mask):
                    pseudo_label1_values = pseudo_label1_batch * inequal_mask
                    pseudo_label2_values = pseudo_label2_batch * inequal_mask

                    outputs_soft1_values1 = torch.zeros_like(pseudo_label1_values, dtype=torch.float32)
                    outputs_soft2_values1 = torch.zeros_like(pseudo_label2_values, dtype=torch.float32)

                    outputs_soft1_values2 = torch.zeros_like(pseudo_label1_values, dtype=torch.float32)
                    outputs_soft2_values2 = torch.zeros_like(pseudo_label2_values, dtype=torch.float32)


                    for i in range(0, num_classes):
                        outputs_soft1_i = outputs_soft1[args.labeled_bs:].detach()[:, i, :, :]
                        outputs_soft2_i = outputs_soft2[args.labeled_bs:].detach()[:, i, :, :]

                        temp_condition = (pseudo_label1_values == i)
                        outputs_soft1_values1 = torch.where(temp_condition, outputs_soft1_i, outputs_soft1_values1)
                        outputs_soft2_values1 = torch.where(temp_condition, outputs_soft2_i, outputs_soft2_values1)

                        temp_condition = (pseudo_label2_values == i)
                        outputs_soft1_values2 = torch.where(temp_condition, outputs_soft1_i, outputs_soft1_values2)
                        outputs_soft2_values2 = torch.where(temp_condition, outputs_soft2_i, outputs_soft2_values2)


                    trust1_mask = (outputs_soft1_values1 > threshold1) & (outputs_soft2_values1 > threshold2)
                    trust2_mask = (outputs_soft1_values2 > threshold1) & (outputs_soft2_values2 > threshold2)
                    
                    update0_mask1 = trust1_mask & trust2_mask
                    if torch.any(update0_mask1):
                        outputs_soft1_values = 0.5 * outputs_soft1_values1 + (1 - 0.5) * outputs_soft2_values1
                        outputs_soft2_values = 0.5 * outputs_soft1_values2 + (1 - 0.5) * outputs_soft2_values2
                        update1_mask1 = torch.zeros_like(outputs_soft1_values)
                        update1_mask1[outputs_soft1_values > outputs_soft2_values] = 1
                        update1_mask1[outputs_soft1_values <= outputs_soft2_values] = 2
                        
                        update2_mask1= update1_mask1 * update0_mask1 * inequal_mask
    
                        pseudo_label1_batch = torch.where(update2_mask1==2, pseudo_label2_batch, pseudo_label1_batch)
                        pseudo_label2_batch = torch.where(update2_mask1==1, pseudo_label1_batch, pseudo_label2_batch)
                    
                        update3_mask1 = trust1_mask & torch.logical_not(trust2_mask) & inequal_mask
                        update4_mask1 = torch.logical_not(trust1_mask) & trust2_mask & inequal_mask   

                        pseudo_label2_batch = torch.where(update3_mask1, pseudo_label1_batch, pseudo_label2_batch)

                        pseudo_label1_batch = torch.where(update4_mask1, pseudo_label2_batch, pseudo_label1_batch)

                        remove_mask = (torch.logical_not(trust1_mask) | torch.logical_not(trust2_mask)) & inequal_mask
                        pseudo_label1_batch[remove_mask] = ignore_value
                        pseudo_label2_batch[remove_mask] = ignore_value
                    
                if torch.any(only_label1_mask): 
                    pseudo_label1_values = pseudo_label1_batch * only_label1_mask

                    for i in range(0, num_classes):
                        outputs_soft1_i = outputs_soft1[args.labeled_bs:].detach()[:, i, :, :]
                        outputs_soft2_i = outputs_soft2[args.labeled_bs:].detach()[:, i, :, :]
                        temp_condition = (pseudo_label1_values == i)
                        outputs_soft1_values = torch.where(temp_condition, outputs_soft1_i, outputs_soft1_values)
                        outputs_soft2_values = torch.where(temp_condition, outputs_soft2_i, outputs_soft1_values)
                    
                    update1_mask2 = (outputs_soft1_values > threshold1) & (outputs_soft2_values > threshold2) & only_label1_mask
                    pseudo_label2_batch = torch.where(update1_mask2, pseudo_label1_batch, pseudo_label2_batch)
                    
                    
                    remove_mask = ((outputs_soft1_values <= threshold1) | (outputs_soft2_values <= threshold2)) & only_label1_mask
                    pseudo_label1_batch[remove_mask] = ignore_value 
                    
                if torch.any(only_label2_mask): 
                    pseudo_label2_values = pseudo_label2_batch * only_label2_mask

                    for i in range(0, num_classes):
                        outputs_soft1_i = outputs_soft1[args.labeled_bs:].detach()[:, i, :, :]
                        outputs_soft2_i = outputs_soft2[args.labeled_bs:].detach()[:, i, :, :]
                        temp_condition = (pseudo_label2_values == i)
                        outputs_soft1_values = torch.where(temp_condition, outputs_soft1_i, outputs_soft1_values)
                        outputs_soft2_values = torch.where(temp_condition, outputs_soft2_i, outputs_soft1_values)

                    update1_mask3 = (outputs_soft1_values > threshold1) & (outputs_soft2_values > threshold2) & only_label2_mask
                    pseudo_label1_batch = torch.where(update1_mask3, pseudo_label2_batch, pseudo_label1_batch)    

                    remove_mask = ((outputs_soft1_values <= threshold1) | (outputs_soft2_values <= threshold2)) & only_label2_mask
                    pseudo_label2_batch[remove_mask] = ignore_value 

            pseudo_label1 = pseudo_label1_batch.to(torch.int64)
            pseudo_label2 = pseudo_label2_batch.to(torch.int64)
            
            # Generate pseudo-labels in the regions without pseudo-labels
            no_pseudo_label_mask1 = (pseudo_label1 == ignore_value)
            no_pseudo_label_mask2 = (pseudo_label2 == ignore_value)

            max_values_soft1 = outputs_soft1[args.labeled_bs:].max(dim=1)[0]  
            max_values_soft2 = outputs_soft2[args.labeled_bs:].max(dim=1)[0]
            
            equal_pred_mask = (pseudo_outputs1 == pseudo_outputs2)
            masked_mse_loss = 0
            if torch.any(equal_pred_mask): 
                clean_mask1 = (max_values_soft1 > threshold1) & (max_values_soft2 > threshold2) & equal_pred_mask & no_pseudo_label_mask1
                clean_mask2 = (max_values_soft1 > threshold1) & (max_values_soft2 > threshold2) & equal_pred_mask & no_pseudo_label_mask2
                
                pseudo_label1 = torch.where(clean_mask1, pseudo_outputs1.to(torch.int64), pseudo_label1)
                pseudo_label2 = torch.where(clean_mask2, pseudo_outputs2.to(torch.int64), pseudo_label2)
                
                mse_mask = (((max_values_soft1 <= threshold1) & (max_values_soft2 > threshold2)) | ((max_values_soft1 > threshold1) & (max_values_soft2 <= threshold2))) & equal_pred_mask
                mse_mask = torch.cat([mse_mask.unsqueeze(1)] * num_classes, dim=1)
                total_mse_loss = mse_loss_v(outputs_soft1[args.labeled_bs:], outputs_soft2[args.labeled_bs:])
                masked_mse_loss = (total_mse_loss * mse_mask).sum() / torch.clamp(mse_mask.sum(), min=1.0)
            
            inequal_pred_mask = (pseudo_outputs1 != pseudo_outputs2)
            if torch.any(inequal_pred_mask): 
                both_mask = (max_values_soft1 < threshold1) & (max_values_soft2 < threshold2)
                both_mask = torch.logical_not(both_mask)
                ambiguous_mask1 = (max_values_soft2 > max_values_soft1) & inequal_pred_mask & no_pseudo_label_mask1 & both_mask
                ambiguous_mask2 = (max_values_soft1 > max_values_soft2) & inequal_pred_mask & no_pseudo_label_mask2 & both_mask
                pseudo_label1 = torch.where(ambiguous_mask1, pseudo_outputs2.to(torch.int64), pseudo_label1)
                pseudo_label2 = torch.where(ambiguous_mask2, pseudo_outputs1.to(torch.int64), pseudo_label2)
            
            #calculate loss
            pseudo_supervision1 = ce_loss_i(outputs1[args.labeled_bs:], pseudo_label1.long())
            pseudo_supervision2 = ce_loss_i(outputs2[args.labeled_bs:], pseudo_label2.long())
            
            model1_loss = sup_loss1 + consistency_weight * pseudo_supervision1
            model2_loss = sup_loss2 + consistency_weight * pseudo_supervision2
            mse_loss = consistency_weight * masked_mse_loss 
            loss = model1_loss + model2_loss + mse_loss
            
            optimizer1.zero_grad()
            optimizer2.zero_grad()

            loss.backward()

            optimizer1.step()
            optimizer2.step()
            
            if iter_num % 5 == 0:
                db_train.update_labels(sampled_batch["idx"][args.labeled_bs:], sampled_batch['aug'][args.labeled_bs:].cpu().numpy(), pseudo_label1, pseudo_label2, snapshot_path)

            iter_num = iter_num + 1
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.90
            for param_group in optimizer1.param_groups:
                param_group['lr'] = lr_
            for param_group in optimizer2.param_groups:
                param_group['lr'] = lr_

            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar(
                'consistency_weight/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('loss/model1_loss',
                              model1_loss, iter_num)
            writer.add_scalar('loss/model2_loss',
                              model2_loss, iter_num)
            logging.info('iteration %d : model1 loss : %f model2 loss : %f' % (iter_num, model1_loss.item(), model2_loss.item()))
            if iter_num % 50 == 0:
                image = volume_batch[1, 0:1, :, :]
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(
                    outputs1, dim=1), dim=1, keepdim=True)
                writer.add_image('train/model1_Prediction',
                                 outputs[1, ...] * 50, iter_num)
                outputs = torch.argmax(torch.softmax(
                    outputs2, dim=1), dim=1, keepdim=True)
                writer.add_image('train/model2_Prediction',
                                 outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model1.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], model1, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/model1_val_{}_dice'.format(class_i+1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/model1_val_{}_hd95'.format(class_i+1),
                                      metric_list[class_i, 1], iter_num)

                performance1 = np.mean(metric_list, axis=0)[0]

                mean_hd951 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/model1_val_mean_dice', performance1, iter_num)
                writer.add_scalar('info/model1_val_mean_hd95', mean_hd951, iter_num)

                if performance1 > best_performance1:
                    best_performance1 = performance1
                    save_mode_path = os.path.join(snapshot_path,
                                                  'model1_iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance1, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model1.pth'.format(args.model))
                    torch.save(model1.state_dict(), save_mode_path)
                    torch.save(model1.state_dict(), save_best)

                logging.info(
                    'iteration %d : model1_mean_dice : %f model1_mean_hd95 : %f' % (iter_num, performance1, mean_hd951))
                model1.train()

                model2.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], model2, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/model2_val_{}_dice'.format(class_i+1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/model2_val_{}_hd95'.format(class_i+1),
                                      metric_list[class_i, 1], iter_num)

                performance2 = np.mean(metric_list, axis=0)[0]

                mean_hd952 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/model2_val_mean_dice', performance2, iter_num)
                writer.add_scalar('info/model2_val_mean_hd95', mean_hd952, iter_num)

                if performance2 > best_performance2:
                    best_performance2 = performance2
                    save_mode_path = os.path.join(snapshot_path,
                                                  'model2_iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance2, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model2.pth'.format(args.model))
                    torch.save(model2.state_dict(), save_mode_path)
                    torch.save(model2.state_dict(), save_best)

                logging.info(
                    'iteration %d : model2_mean_dice : %f model2_mean_hd95 : %f' % (iter_num, performance2, mean_hd952))
                model2.train()

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'model1_iter_' + str(iter_num) + '.pth')
                torch.save(model1.state_dict(), save_mode_path)
                logging.info("save model1 to {}".format(save_mode_path))

                save_mode_path = os.path.join(
                    snapshot_path, 'model2_iter_' + str(iter_num) + '.pth')
                torch.save(model2.state_dict(), save_mode_path)
                logging.info("save model2 to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()


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

    snapshot_path = "../model/{}_{}/{}".format(
        args.exp, args.labeled_num, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
