from __future__ import print_function
import argparse
import sys
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision.transforms as transforms
from data_loader import SYSUData, RegDBData, LLCMData, TestData
from data_manager import *
from eval_metrics import eval_sysu, eval_regdb, eval_llcm
from model import embed_net
from utils import *
from loss import OriTripletLoss, DualModalityCenterLoss
from tensorboardX import SummaryWriter

import build_optimizer
import build_lrscheduler
import build_transforms

from torch.cuda import amp

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='sysu',help='dataset name: regdb or sysu or llcm]')
parser.add_argument('--arch', default='resnet50', type=str,help='network baseline:resnet18 or resnet50')
parser.add_argument('--resume', '-r', default='', type=str, help='resume from checkpoint')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--save_epoch', default=20, type=int,metavar='s', help='save model every 10 epochs')
parser.add_argument('--optim', default='SGD', type=str, help='SGD,ADM')
parser.add_argument('--model_path', default='result/saved_model/', type=str, help='model save path')
parser.add_argument('--log_path', default='result/log/', type=str, help='log save path')
parser.add_argument('--vis_log_path', default='result/log/vis_log/', type=str, help='log save path')
parser.add_argument('--loss_tri', default='DMC', type=str, help='')
parser.add_argument('--lr_scheduler', default='step',type=str, help='step or consine')
parser.add_argument('--backbone', default='AGW',type=str, help='AGW')
parser.add_argument('--lr', default=0.1, type=float,help='learning rate, 0.00035/0.0001 for adam,sgd 0.1')
# parser.add_argument('--lr', default=1e-2, type=float,help='learning rate, 0.00035/0.0001 for adam,sgd 0.1')
parser.add_argument('--workers', default=4, type=int, metavar='N',help='number of data loading workers (default: 4)')
parser.add_argument('--img_w', default=192, type=int,metavar='imgw', help='img width')
parser.add_argument('--img_h', default=384, type=int,metavar='imgh', help='img height')
# parser.add_argument('--batch-size', default=6, type=int,metavar='B', help='training batch size')
parser.add_argument('--batch-size', default=4, type=int,metavar='B', help='training batch size')
parser.add_argument('--test-batch', default=64, type=int,metavar='tb', help='testing batch size')
parser.add_argument('--margin', default=0.7, type=float, metavar='margin', help='triplet loss margin')
parser.add_argument('--num_pos', default=4, type=int,help='num of pos per identity in each modality')
parser.add_argument('--trial', default=1, type=int,metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--seed', default=0, type=int,metavar='t', help='random seed')
parser.add_argument('--gpu', default='0', type=int,help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='all', type=str, help='all or indoor')

args = parser.parse_args()

torch.cuda.set_device(args.gpu)

set_seed(args.seed)

dataset = args.dataset

# 将print输出重定向到文件
if(dataset=='sysu'):
    trainingLogFilePath = 'trainLog_QFEMFACAMPAM_sysu.txt'
elif(dataset=='regdb'):
    trainingLogFilePath = 'trainLog_QFEMFACAMPAM_regdb.txt'
trainingLogFile = open(trainingLogFilePath, 'w')
sys.stdout = trainingLogFile

if dataset == 'sysu':

    # data_path = './datasets/SYSU-MM01/'
    data_path ='../datasets/SYSU-MM01/'
    log_path = args.log_path + 'sysu_log/'
    test_mode = [1, 2]  # thermal to visible
elif dataset == 'regdb':
    # data_path = './datasets/RegDB/'
    data_path = '../datasets/RegDB/'
    log_path = args.log_path + 'regdb_log/'
    test_mode = [2, 1]  # visible to thermal
elif dataset == 'llcm':
    data_path = '../datasets/LLCM/'
    log_path = args.log_path + 'llcm_log/'
    test_mode = [1, 2]  # [1, 2]: IR to VIS; [2, 1]: VIS to IR;

checkpoint_path = args.model_path

if not os.path.isdir(log_path):
    os.makedirs(log_path)
if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)
if not os.path.isdir(args.vis_log_path):
    os.makedirs(args.vis_log_path)

suffix = dataset
suffix = suffix + '_hsfl_p{}_n{}_lr_{}_seed_{}'.format(
    args.num_pos, args.batch_size, args.lr, args.seed)

if not args.optim == 'SGD':
    suffix = suffix + '_' + args.optim

if dataset == 'regdb':
    suffix = suffix + '_trial_{}'.format(args.trial)

sys.stdout = Logger(log_path + suffix + '_os.txt')
vis_log_dir = args.vis_log_path + suffix + '/'

if not os.path.isdir(vis_log_dir):
    os.makedirs(vis_log_dir)
writer = SummaryWriter(vis_log_dir)
print("==========\nArgs:{}\n==========".format(args))



device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0

print('==> Loading data..')
# Data loading code
# normalize
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

transform_test = build_transforms.test_transforms(
    args.img_h, args.img_w, normalize)
transform_color1 = build_transforms.train_transforms_color1(
    args.img_h, args.img_w, normalize)
transform_color2 = build_transforms.train_transforms_color2(
    args.img_h, args.img_w, normalize)
transform_thermal1 = build_transforms.train_transforms_thermal1(
    args.img_h, args.img_w, normalize)
transform_thermal2 = build_transforms.train_transforms_thermal2(
    args.img_h, args.img_w, normalize)
transform_train = transform_color1, transform_color2, transform_thermal1, transform_thermal2

end = time.time()

if dataset == 'sysu':
    # training set
    trainset = SYSUData(data_path, transform=transform_train)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(
        trainset.train_color_label, trainset.train_thermal_label)

    # testing set
    query_img, query_label, query_cam = process_query_sysu(
        data_path, mode=args.mode)
    gall_img, gall_label, gall_cam = process_gallery_sysu(
        data_path, mode=args.mode, trial=0)

elif dataset == 'regdb':
    # training set
    trainset = RegDBData(data_path, args.trial, args.img_h,
                         args.img_w, transform=transform_train)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(
        trainset.train_color_label, trainset.train_thermal_label)

    # testing set
    query_img, query_label = process_test_regdb(
        data_path, trial=args.trial, modal='visible')
    gall_img, gall_label = process_test_regdb(
        data_path, trial=args.trial, modal='thermal')
    
elif dataset == 'llcm':
    # training set
    trainset = LLCMData(data_path, args.trial, args.img_h,
                        args.img_w, transform=transform_train)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(
        trainset.train_color_label, trainset.train_thermal_label)

    # testing set
    query_img, query_label, query_cam = process_query_llcm(
        data_path, mode=test_mode[1])
    gall_img, gall_label, gall_cam = process_gallery_llcm(
        data_path, mode=test_mode[0], trial=0)

gallset = TestData(gall_img, gall_label, transform=transform_test,
                   img_size=(args.img_w, args.img_h))
queryset = TestData(query_img, query_label,
                    transform=transform_test, img_size=(args.img_w, args.img_h))

# testing data loader
gall_loader = data.DataLoader(
    gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
query_loader = data.DataLoader(
    queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

n_class = len(np.unique(trainset.train_color_label))
nquery = len(query_label)
ngall = len(gall_label)

gall_loader
print('Dataset {} statistics:'.format(dataset))
print('  ------------------------------')
print('  subset   | # ids | # images')
print('  ------------------------------')
print('  visible  | {:5d} | {:8d}'.format(
    n_class, len(trainset.train_color_label)))
print('  thermal  | {:5d} | {:8d}'.format(
    n_class, len(trainset.train_thermal_label)))
print('  ------------------------------')
print('  query    | {:5d} | {:8d}'.format(len(np.unique(query_label)), nquery))
print('  gallery  | {:5d} | {:8d}'.format(len(np.unique(gall_label)), ngall))
print('  ------------------------------')
print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

print('==> Building model..')


##############################################

if args.backbone == 'AGW':
    net = embed_net(n_class, arch=args.arch)
    embeding_dim = net.pool_dim

net.to(device)
# cudnn.benchmark = True

if len(args.resume) > 0:
    model_path = checkpoint_path + args.resume

    if os.path.isfile(model_path):
        print('==> loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['net'])
        print('==> loaded checkpoint {} (epoch {})'
              .format(args.resume, checkpoint['epoch']))
    else:
        print('==> no checkpoint found at {}'.format(args.resume))

# define loss function
criterion_id = nn.CrossEntropyLoss()
# criterion_id_s = nn.CrossEntropyLoss(label_smoothing=0.5)
if args.loss_tri == 'DMC':
    loader_batch = args.batch_size * args.num_pos
    criterion_tri = DualModalityCenterLoss(k_size=4, margin=0.7)

else:
    loader_batch = args.batch_size * args.num_pos
    criterion_tri = OriTripletLoss(batch_size=loader_batch, margin=args.margin)


criterion_id.to(device)
criterion_tri.to(device)


optimizer, shallow_optimizer = build_optimizer.build_optim(
    net, args.optim, args.lr)
scdule = build_lrscheduler.adjust_learning_rate(optimizer, args.lr_scheduler)
scdule_shallow = build_lrscheduler.adjust_learning_rate(
    shallow_optimizer, args.lr_scheduler)

scaler = amp.GradScaler()

# cam_mask = None

def train(epoch):
    # global cam_mask
    current_lr = scdule.get_lr()[0]
    lr_shallow = scdule_shallow.get_lr()[0]
    train_loss = AverageMeter()
    id_loss = AverageMeter()
    tri_loss = AverageMeter()
    sep_loss = AverageMeter()

    data_time = AverageMeter()
    batch_time = AverageMeter()

    correct_id = 0

    total = 0

    # switch to train mode
    net.train()
    end = time.time()

    for batch_idx, (input10, input11, input20, input21, label1, label2) in enumerate(trainloader):

        with amp.autocast(enabled=True):

            labels = torch.cat((label1, label1, label2, label2), 0)

            input10 = Variable(input10.cuda())
            input11 = Variable(input11.cuda())
            input20 = Variable(input20.cuda())
            input21 = Variable(input21.cuda())


            labels = Variable(labels.cuda())
            data_time.update(time.time() - end)

            # feat, out0, loss_sep = net(input10, input11, input20, input21)
            feat, out0, loss_sep, feat_logit_styles = net(input10, input11, input20, input21)
            
            loss_id = criterion_id(out0, labels)
            for i in range(net.part_num):
                logit = feat_logit_styles[i]
                loss_id += criterion_id(logit, labels)
            
            loss_tri = criterion_tri(feat, labels)

            _, predicted = out0.max(1)
            correct_id += (predicted.eq(labels).sum().item())

            loss = loss_id+loss_tri+loss_sep  



        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update P
        train_loss.update(loss.item(), 2 * input10.size(0))
        id_loss.update(loss_id.item(), 2 * input10.size(0))
        tri_loss.update(loss_tri.item(), 2 * input10.size(0))
        sep_loss.update(loss_sep.item(), 2 * input10.size(0))

        total += labels.size(0)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % 50 == 0:
            print('Epoch: [{}][{}/{}] '
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  'lr:{:.6f} '
                  'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f}) '
                  'IDLoss: {id_loss.val:.4f} ({id_loss.avg:.4f}) '
                  'QCTLoss: {tri_loss.val:.4f} ({tri_loss.avg:.4f}) '
                  'SepLoss: {sep_loss.val:.4f} ({sep_loss.avg:.4f}) '
                  'Accu: {:.2f} '.format(
                      epoch, batch_idx, len(trainloader), current_lr,
                      100. * correct_id / total, batch_time=batch_time,
                      train_loss=train_loss, id_loss=id_loss, tri_loss=tri_loss, sep_loss=sep_loss))
        sys.stdout.flush()
    scdule.step()
    scdule_shallow.step()

    writer.add_scalar('total_loss', train_loss.avg, epoch)
    writer.add_scalar('id_loss', id_loss.avg, epoch)
    writer.add_scalar('tri_loss', tri_loss.avg, epoch)
    writer.add_scalar('sep_loss', sep_loss.avg, epoch)
    writer.add_scalar('lr', current_lr, epoch)

    print(lr_shallow)


def test(epoch):
    # switch to evaluation mode
    net.eval()
    print('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    nout = embeding_dim
    gall_feat = np.zeros((ngall, nout))
    gall_feat_att = np.zeros((ngall, nout))

    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(gall_loader):
            batch_num = input.size(0)

            # input = torch.cat((input, input,), 0)
            input = Variable(input.cuda())
            featA, feat_attA = net(input, input, input, input, test_mode[0])

            gall_feat[ptr:ptr + batch_num, :] = featA.detach().cpu().numpy()
            gall_feat_att[ptr:ptr + batch_num,
                          :] = feat_attA.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    # switch to evaluation
    net.eval()
    print('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    nout = embeding_dim
    query_feat = np.zeros((nquery, nout))
    query_feat_att = np.zeros((nquery, nout))

    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(query_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            featA, feat_attA = net(input, input, input, input, test_mode[1])

            query_feat[ptr:ptr + batch_num, :] = featA.detach().cpu().numpy()
            query_feat_att[ptr:ptr + batch_num,
                           :] = feat_attA.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    start = time.time()

    # compute the similarity
    distmat = np.matmul(query_feat, np.transpose(gall_feat))
    distmat_att = np.matmul(query_feat_att, np.transpose(gall_feat_att))

    # evaluation
    if dataset == 'regdb':
        cmc, mAP, mINP = eval_regdb(-distmat, query_label, gall_label)
        cmc_att, mAP_att, mINP_att = eval_regdb(
            -distmat_att, query_label, gall_label)
    elif dataset == 'sysu':
        cmc, mAP, mINP = eval_sysu(-distmat, query_label,
                                   gall_label, query_cam, gall_cam)
        cmc_att, mAP_att, mINP_att = eval_sysu(
            -distmat_att, query_label, gall_label, query_cam, gall_cam)
    elif dataset == 'llcm':
        cmc, mAP, mINP = eval_llcm(-distmat, query_label,
                                   gall_label, query_cam, gall_cam)
        cmc_att, mAP_att, mINP_att = eval_llcm(
            -distmat_att, query_label, gall_label, query_cam, gall_cam)

    print('Evaluation Time:\t {:.3f}'.format(time.time() - start))

    writer.add_scalar('rank1', cmc[0], epoch)
    writer.add_scalar('mAP', mAP, epoch)
    writer.add_scalar('mINP', mINP, epoch)
    writer.add_scalar('rank1_att', cmc_att[0], epoch)
    writer.add_scalar('mAP_att', mAP_att, epoch)
    writer.add_scalar('mINP_att', mINP_att, epoch)
    return cmc, mAP, mINP, cmc_att, mAP_att, mINP_att


# training
print('==> Start Training...')
for epoch in range(start_epoch, 150 - start_epoch):

    print('==> Preparing Data Loader...')
    # identity sampler
    sampler = IdentitySampler(trainset.train_color_label,
                              trainset.train_thermal_label, color_pos, thermal_pos, args.num_pos, args.batch_size,
                              epoch)

    trainset.cIndex = sampler.index1  # color index
    trainset.tIndex = sampler.index2  # thermal index
    print(epoch)
    print(trainset.cIndex)
    print(trainset.tIndex)

    loader_batch = args.batch_size * args.num_pos

    trainloader = data.DataLoader(trainset, batch_size=loader_batch,
                                  sampler=sampler, num_workers=args.workers, drop_last=True,)

    # training
    train(epoch)

    if epoch > 0 and epoch % 1 == 0:
        print('Test Epoch: {}'.format(epoch))

        # testing
        cmc, mAP, mINP, cmc_att, mAP_att, mINP_att = test(epoch)
        # save model
        if cmc_att[0] > best_acc:  # not the real best for sysu-mm01
            best_acc = cmc_att[0]
            best_epoch = epoch
            state = {
                'net': net.state_dict(),
                'cmc': cmc_att,
                'mAP': mAP_att,
                'mINP': mINP_att,
                'epoch': epoch,
            }
            torch.save(state, checkpoint_path + suffix + '_best.t')

        # save model
        # if epoch > 0 and epoch % args.save_epoch == 0:
        #     state = {
        #         'net': net.state_dict(),
        #         'cmc': cmc,
        #         'mAP': mAP,
        #         'epoch': epoch,
        #     }
        #     torch.save(state, checkpoint_path + suffix +
        #                '_epoch_{}.t'.format(epoch))

        print('POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
        print('FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc_att[0], cmc_att[4], cmc_att[9], cmc_att[19], mAP_att, mINP_att))
        print('Best Epoch [{}]'.format(best_epoch))

trainingLogFile.close()
# 恢复标准输出
sys.stdout = sys.__stdout__