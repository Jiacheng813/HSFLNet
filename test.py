from __future__ import print_function
import argparse
import time
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision.transforms as transforms
from data_loader import SYSUData, RegDBData, TestData
from data_manager import *
from eval_metrics import eval_sysu, eval_regdb, eval_llcm
from model import embed_net
from utils import *
import sys

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='sysu',
                    help='dataset name: regdb or sysu or llcm]')
parser.add_argument('--lr', default=0.1, type=float,
                    help='learning rate, 0.00035 for adam')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--arch', default='resnet50', type=str,
                    help='network baseline: resnet50')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--save_epoch', default=20, type=int,
                    metavar='s', help='save model every 10 epochs')
parser.add_argument('--resume', '-r', default='sysu_hsfl_p4_n4_lr_0.1_seed_0_best.t', type=str,
                    help='resume from checkpoint')
parser.add_argument('--model_path', default='result/saved_model/',
                    type=str, help='model save path')
parser.add_argument('--log_path', default='result/log/',
                    type=str, help='log save path')
parser.add_argument(
    '--vis_log_path', default='result/log/vis_log/', type=str, help='log save path')
parser.add_argument('--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--img_w', default=192, type=int,
                    metavar='imgw', help='img width')
parser.add_argument('--img_h', default=384, type=int,
                    metavar='imgh', help='img height')
parser.add_argument('--batch-size', default=6, type=int,
                    metavar='B', help='training batch size')
parser.add_argument('--test-batch', default=4, type=int,
                    metavar='tb', help='testing batch size')
parser.add_argument('--method', default='AGW', type=str,
                    metavar='m', help='method type: base or AGW')
parser.add_argument('--margin', default=0.3, type=float,
                    metavar='margin', help='triplet loss margin')
parser.add_argument('--num_pos', default=4, type=int,
                    help='num of pos per identity in each modality')
parser.add_argument('--trial', default=0, type=int,
                    metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--seed', default=0, type=int,
                    metavar='t', help='random seed')
parser.add_argument('--gpu', default='0', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='all', type=str,
                    help='all or indoor for sysu')
parser.add_argument('--tvsearch', action='store_true',
                    help='retrive thermal to visible search on RegDB')
args = parser.parse_args()

dataset = args.dataset

# 将print输出重定向到文件
if(dataset=='sysu'):
    trainingLogFilePath = 'testLog_QFEMFACAMPAM_sysu_mode={}.txt'.format(args.mode)
elif(dataset=='regdb'):
    trainingLogFilePath = 'testLog_QFEMFACAMPAM_regdb_tvsearch={}.txt'.format(args.tvsearch)
    
trainingLogFile = open(trainingLogFilePath, 'w')
sys.stdout = trainingLogFile

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
need_evaluation = 1


if dataset == 'sysu':
    data_path = '../datasets/SYSU-MM01/'
    n_class = 395
    test_mode = [1, 2]
elif dataset == 'regdb':
    data_path = '../datasets/RegDB/'
    n_class = 206
    test_mode = [2, 1]
elif dataset == 'llcm':
    data_path = '../datasets/LLCM/'
    n_class = 713
    test_mode = [1, 2]  # [1, 2]: IR to VIS; [2, 1]: VIS to IR;

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0

print('==> Building model..')
if args.method == 'AGW':
    net = embed_net(n_class, arch=args.arch)

pool_dim = net.pool_dim
    
net.to(device)
cudnn.benchmark = True

checkpoint_path = args.model_path


print('==> Loading data..')
# Data loading code

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop((args.img_h, args.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_h, args.img_w)),
    transforms.ToTensor(),
    normalize,
])

end = time.time()

def extract_gall_feat(gall_loader):
    net.eval()
    print('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat_pool = np.zeros((ngall, pool_dim))
    gall_feat_fc = np.zeros((ngall, pool_dim))
    gall_feat_fc_label = []
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(gall_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat_poolA, feat_fcA = net(
                input, input, input, input, test_mode[0])
            # feat_pool=feat_poolA[feat_poolA.size(0)//2:feat_poolA.size(0),:]
            # feat_fc=feat_fcA[feat_fcA.size(0)//2:feat_fcA.size(0),:]
            gall_feat_pool[ptr:ptr+batch_num,
                           :] = feat_poolA.detach().cpu().numpy()
            gall_feat_fc[ptr:ptr+batch_num,
                         :] = feat_fcA.detach().cpu().numpy()
            ptr = ptr + batch_num

            # ########
            if need_evaluation:
                gall_feat_fc_label.extend(label.detach().cpu().numpy())

    print('Extracting Time:\t {:.3f}'.format(time.time()-start))
    return gall_feat_pool, gall_feat_fc, np.array(gall_feat_fc_label)

def extract_query_feat(query_loader):
    net.eval()
    print('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat_pool = np.zeros((nquery, pool_dim))
    query_feat_fc = np.zeros((nquery, pool_dim))
    query_feat_fc_label = []
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(query_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat_poolA, feat_fcA = net(
                input, input, input, input, test_mode[1])
            query_feat_pool[ptr:ptr+batch_num,
                            :] = feat_poolA.detach().cpu().numpy()
            query_feat_fc[ptr:ptr+batch_num,
                          :] = feat_fcA.detach().cpu().numpy()
            ptr = ptr + batch_num
            # ######
            if need_evaluation:
                query_feat_fc_label.extend(label.detach().cpu().numpy())

    print('Extracting Time:\t {:.3f}'.format(time.time()-start))
    return query_feat_pool, query_feat_fc, np.array(query_feat_fc_label)

if dataset == 'sysu':

    print('==> Resuming from checkpoint..')
    if len(args.resume) > 0:
        model_path = checkpoint_path + args.resume
        # model_path = checkpoint_path + 'sysu_hsfl_p4_n4_lr_0.1_seed_0_best.t'
        print(model_path)
        if os.path.isfile(model_path):
            print('==> loading checkpoint {}'.format(model_path))
            checkpoint = torch.load(model_path, map_location={
                                    'cuda:2': 'cuda:0', 'cuda:1': 'cuda:0'})
            net.load_state_dict(checkpoint['net'])
            print('==> loaded checkpoint {} (epoch {})'
                  .format(model_path, checkpoint['epoch']))
        else:
            print('==> no checkpoint found at {}'.format(args.resume))

    # testing set
    query_img, query_label, query_cam = process_query_sysu(
        data_path, mode=args.mode)
    gall_img, gall_label, gall_cam = process_gallery_sysu(
        data_path, mode=args.mode, trial=0)

    nquery = len(query_label)
    ngall = len(gall_label)
    print("Dataset statistics:")
    print("  ------------------------------")
    print("  subset   | # ids | # images")
    print("  ------------------------------")
    print("  query    | {:5d} | {:8d}".format(
        len(np.unique(query_label)), nquery))
    print("  gallery  | {:5d} | {:8d}".format(
        len(np.unique(gall_label)), ngall))
    print("  ------------------------------")

    queryset = TestData(query_img, query_label, transform=transform_test, img_size=(
        args.img_w, args.img_h))
    query_loader = data.DataLoader(
        queryset, batch_size=args.test_batch, shuffle=False, num_workers=8)
    print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

    query_feat_pool, query_feat_fc, query_feat_fc_label = extract_query_feat(
        query_loader)
    for trial in range(10):

        gall_img, gall_label, gall_cam = process_gallery_sysu(
            data_path, mode=args.mode, trial=trial)

        trial_gallset = TestData(
            gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
        trial_gall_loader = data.DataLoader(
            trial_gallset, batch_size=args.test_batch, shuffle=False, num_workers=4)

        gall_feat_pool, gall_feat_fc, gall_feat_fc_label = extract_gall_feat(
            trial_gall_loader)

        # pool5 feature
        distmat_pool = np.matmul(query_feat_pool, np.transpose(gall_feat_pool))
        cmc_pool, mAP_pool, mINP_pool = eval_sysu(
            -distmat_pool, query_label, gall_label, query_cam, gall_cam)

        # fc feature
        distmat = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))
        cmc, mAP, mINP = eval_sysu(-distmat, query_label,
                                   gall_label, query_cam, gall_cam)

        if trial == 0:
            all_cmc = cmc
            all_mAP = mAP
            all_mINP = mINP
            all_cmc_pool = cmc_pool
            all_mAP_pool = mAP_pool
            all_mINP_pool = mINP_pool
        else:
            all_cmc = all_cmc + cmc
            all_mAP = all_mAP + mAP
            all_mINP = all_mINP + mINP
            all_cmc_pool = all_cmc_pool + cmc_pool
            all_mAP_pool = all_mAP_pool + mAP_pool
            all_mINP_pool = all_mINP_pool + mINP_pool

        print('Test Trial: {}'.format(trial))
        print(
            'FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
        print(
            'POOL: Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc_pool[0], cmc_pool[4], cmc_pool[9], cmc_pool[19], mAP_pool, mINP_pool))

elif dataset == 'regdb':

    for trial in range(10):
        test_trial = trial + 1
        model_path = checkpoint_path + args.resume
        # model_path = checkpoint_path + 'regdb_hsfl_p4_n4_lr_0.1_seed_0_trial_1_best.t'
        print('model_path={}'.format(model_path))
        if os.path.isfile(model_path):
            print('==> loading checkpoint {}'.format(model_path))
            checkpoint = torch.load(model_path)
            net.load_state_dict(checkpoint['net'])

        # training set
        trainset = RegDBData(data_path, test_trial, transform=transform_train)
        # generate the idx of each person identity
        color_pos, thermal_pos = GenIdx(
            trainset.train_color_label, trainset.train_thermal_label)

        # testing set
        query_img, query_label = process_test_regdb(
            data_path, trial=test_trial, modal='visible')
        gall_img, gall_label = process_test_regdb(
            data_path, trial=test_trial, modal='thermal')

        gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(
            args.img_w, args.img_h))
        gall_loader = data.DataLoader(
            gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

        nquery = len(query_label)
        ngall = len(gall_label)

        queryset = TestData(query_img, query_label, transform=transform_test, img_size=(
            args.img_w, args.img_h))
        query_loader = data.DataLoader(
            queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
        print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

        query_feat_pool, query_feat_fc, label1 = extract_query_feat(
            query_loader)
        gall_feat_pool,  gall_feat_fc, label2 = extract_gall_feat(gall_loader)

        if args.tvsearch:
            print('tvsearch')
            # pool5 feature
            distmat_pool = np.matmul(
                gall_feat_pool, np.transpose(query_feat_pool))
            cmc_pool, mAP_pool, mINP_pool = eval_regdb(
                -distmat_pool, gall_label, query_label)

            # fc feature
            distmat = np.matmul(gall_feat_fc, np.transpose(query_feat_fc))
            cmc, mAP, mINP = eval_regdb(-distmat, gall_label,  query_label)
        else:
            print('vtsearch')
            # pool5 feature
            distmat_pool = np.matmul(
                query_feat_pool, np.transpose(gall_feat_pool))
            cmc_pool, mAP_pool, mINP_pool = eval_regdb(
                -distmat_pool, query_label, gall_label)

            # fc feature
            distmat = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))
            cmc, mAP, mINP = eval_regdb(-distmat, query_label, gall_label)

        if trial == 0:
            all_cmc = cmc
            all_mAP = mAP
            all_mINP = mINP
            all_cmc_pool = cmc_pool
            all_mAP_pool = mAP_pool
            all_mINP_pool = mINP_pool
        else:
            all_cmc = all_cmc + cmc
            all_mAP = all_mAP + mAP
            all_mINP = all_mINP + mINP
            all_cmc_pool = all_cmc_pool + cmc_pool
            all_mAP_pool = all_mAP_pool + mAP_pool
            all_mINP_pool = all_mINP_pool + mINP_pool

        print('Test Trial: {}'.format(trial))
        print(
            'FC:     Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
        print(
            'POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc_pool[0], cmc_pool[4], cmc_pool[9], cmc_pool[19], mAP_pool, mINP_pool))

elif dataset == 'llcm':
    print('==> Resuming from checkpoint..')
    if len(args.resume) > 0:
        model_path = checkpoint_path + args.resume
        if os.path.isfile(model_path):
            print('==> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(model_path)
            net.load_state_dict(checkpoint['net'])
            print('==> loaded checkpoint {} (epoch {})'
                  .format(args.resume, checkpoint['epoch']))
        else:
            print('==> no checkpoint found at {}'.format(args.resume))

    # testing set
    query_img, query_label, query_cam = process_query_llcm(data_path, mode=test_mode[1])
    gall_img, gall_label, gall_cam = process_gallery_llcm(data_path, mode=test_mode[0], trial=0)

    nquery = len(query_label)
    ngall = len(gall_label)
    print("Dataset statistics:")
    print("  ------------------------------")
    print("  subset   | # ids | # images")
    print("  ------------------------------")
    print("  query    | {:5d} | {:8d}".format(len(np.unique(query_label)), nquery))
    print("  gallery  | {:5d} | {:8d}".format(len(np.unique(gall_label)), ngall))
    print("  ------------------------------")

    queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
    query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
    print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

    query_feat1, query_feat2,query_label= extract_query_feat(query_loader)
    for trial in range(10):
        gall_img, gall_label, gall_cam = process_gallery_llcm(data_path, mode=test_mode[0], trial=trial)

        trial_gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
        trial_gall_loader = data.DataLoader(trial_gallset, batch_size=args.test_batch, shuffle=False, num_workers=4)

        gall_feat1, gall_feat2,gall_label = extract_gall_feat(trial_gall_loader)

        # fc feature
        #if test_mode == [1, 2]:
        if 1:
            distmat1 = np.matmul(query_feat1, np.transpose(gall_feat1))
            distmat2 = np.matmul(query_feat2, np.transpose(gall_feat2))

            a = 0.1
            distmat = distmat1 + distmat2 
            distmat_A = a * (distmat1) + (1 - a) * (distmat2 )
            
            cmc, mAP, mINP = eval_llcm(-distmat, query_label, gall_label, query_cam, gall_cam)
            cmc_pool, mAP_pool, mINP_pool = eval_llcm(-distmat_A, query_label, gall_label, query_cam, gall_cam)


        if trial == 0:
            all_cmc = cmc
            all_mAP = mAP
            all_mINP = mINP

            all_cmc_pool = cmc_pool
            all_mAP_pool = mAP_pool
            all_mINP_pool = mINP_pool

        else:
            all_cmc = all_cmc + cmc
            all_mAP = all_mAP + mAP
            all_mINP = all_mINP + mINP

            all_cmc_pool = all_cmc_pool + cmc_pool
            all_mAP_pool = all_mAP_pool + mAP_pool
            all_mINP_pool = all_mINP_pool + mINP_pool


        print('Test Trial: {}'.format(trial))
        print('POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
        print('POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc_pool[0], cmc_pool[4], cmc_pool[9], cmc_pool[19], mAP_pool, mINP_pool))


cmc = all_cmc / 10
mAP = all_mAP / 10
mINP = all_mINP / 10

cmc_pool = all_cmc_pool / 10
mAP_pool = all_mAP_pool / 10
mINP_pool = all_mINP_pool / 10
print('All Average:')
print('FC:     Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
    cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
print('POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
    cmc_pool[0], cmc_pool[4], cmc_pool[9], cmc_pool[19], mAP_pool, mINP_pool))
