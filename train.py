import os
from functools import partial
import numpy as np
import torch
import torch.backends.cudnn as cudnn

# Benchmark模式会提升计算速度，但是由于计算中有随机性，每次网络前馈结果略有差异。如果想要避免这种结果波动
torch.backends.cudnn.deterministic = True
# 设置为True，说明设置为使用使用非确定性算法：
torch.backends.cudnn.enabled = True
cudnn.benchmark = True
# torch.distributed 包提供分布式支持，包括 GPU 和 CPU 的分布式训练支持。
import torch.distributed as dist
# torch自带的一个优化器，里面自带了求导，更新等操作
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.utils import get_num_classes
from nets.facenet import Facenet
from utils.callback import LossHistory


def triple_loss(alpha=.2):
    def _triplet_loss(y_pred, Batch_size):
        anchor, positive, negative = (y_pred[:int(batch_size)],
                                      y_pred[int(Batch_size):int(2 * Batch_size)],
                                      y_pred[int(2 * Batch_size):])
        pos_dist = torch.sqrt(torch.sum(torch.pow(anchor - positive, 2), axis=-1))
        neg_dist = torch.sqrt(torch.sum(torch.pow(anchor - negative, 2), axis=-1))

        keep_all = (neg_dist - pos_dist < alpha).cpu().numpy().flatten()
        hard_triples = np.where(keep_all == 1)

        pos_dist = pos_dist[hard_triples]
        neg_dist = neg_dist[hard_triples]

        basic_loss = pos_dist - neg_dist + alpha
        loss = torch.sum(basic_loss) / torch.max(torch.tensor(1), torch.tensor(len(hard_triples[0])))
        return loss

    return _triplet_loss


if __name__ == '__main__':
    # -------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    # -------------------------------#
    Cuda = True
    # ----------------------------------------------#
    #   Seed    用于固定随机种子
    #           使得每次独立训练都可以获得一样的结果
    # ----------------------------------------------#
    seed = 11
    # ---------------------------------------------------------------------#
    #   distributed     用于指定是否使用单机多卡分布式运行
    #                   终端指令仅支持Ubuntu。CUDA_VISIBLE_DEVICES用于在Ubuntu下指定显卡。
    #                   Windows系统下默认使用DP模式调用所有显卡，不支持DDP。
    #   DP模式：
    #       设置            distributed = False
    #       在终端中输入    CUDA_VISIBLE_DEVICES=0,1 python train.py
    #   DDP模式：
    #       设置            distributed = True
    #       在终端中输入    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py
    # nproc_per_node：一个节点中显卡的数量
    distributed = False
    # ---------------------------------------------------------------------#
    #   sync_bn     是否使用sync_bn，DDP模式多卡可用
    # ---------------------------------------------------------------------#
    sync_bn = False
    # ---------------------------------------------------------------------#
    #   fp16        是否使用混合精度训练
    #               可减少约一半的显存、需要pytorch1.7.1以上
    # ---------------------------------------------------------------------#
    fp16 = False
    # --------------------------------------------------------#
    #   指向根目录下的cls_train.txt，读取人脸路径与标签
    # --------------------------------------------------------#
    annotation_path = "cls_train.txt"
    # --------------------------------------------------------#
    #   输入图像大小，常用设置如[112, 112, 3]
    # --------------------------------------------------------#
    input_shape = [160, 160, 3]
    # --------------------------------------------------------#
    #   主干特征提取网络的选择
    #   mobilenet
    #   inception_resnetv1
    # --------------------------------------------------------#
    backbone = "mobilenet"
    #   权值文件的下载请看README，可以通过网盘下载。
    #   模型的 预训练权重 比较重要的部分是 主干特征提取网络的权值部分，用于进行特征提取。
    #   如果训练过程中存在中断训练的操作，可以将model_path设置成logs文件夹下的权值文件，将已经训练了一部分的权值再次载入。
    #   同时修改下方的训练的参数，来保证模型epoch的连续性。
    #   当model_path = ''的时候不加载整个模型的权值。
    #
    #   此处使用的是整个模型的权重，因此是在train.py进行加载的，pretrain不影响此处的权值加载。
    #   如果想要让模型从主干的预训练权值开始训练，则设置model_path = ''，pretrain = True，此时仅加载主干。
    #   如果想要让模型从0开始训练，则设置model_path = ''，pretrain = Fasle，此时从0开始训练。
    model_path = "model_data/facenet_mobilenet.pth"
    # ----------------------------------------------------------------------------------------------------------------------------#
    #   是否使用主干网络的预训练权重，此处使用的是主干的权重，因此是在模型构建的时候进行加载的。
    #   如果设置了model_path，则主干的权值无需加载，pretrained的值无意义。
    #   如果不设置model_path，pretrained = True，此时仅加载主干开始训练。
    #   如果不设置model_path，pretrained = False，此时从0开始训练。
    # ----------------------------------------------------------------------------------------------------------------------------#
    pretrained = False
    # ----------------------------------------------------------------------------------------------------------------------------#
    #   显存不足与数据集大小无关，提示显存不足请调小batch_size。
    #   受到BatchNorm层影响，不能为1。
    #   #   在此提供若干参数设置建议，各位训练者根据自己的需求进行灵活调整：
    #   （一）从预训练权重开始训练：
    #       Adam：
    #           Init_Epoch = 0，Epoch = 100，optimizer_type = 'adam'，Init_lr = 1e-3，weight_decay = 0。
    #       SGD：
    #           Init_Epoch = 0，Epoch = 100，optimizer_type = 'sgd'，Init_lr = 1e-2，weight_decay = 5e-4。
    #       其中：UnFreeze_Epoch可以在100-300之间调整。
    #   （二）batch_size的设置：
    #       在显卡能够接受的范围内，以大为好。显存不足与数据集大小无关，提示显存不足（OOM或者CUDA out of memory）请调小batch_size。
    #       受到BatchNorm层影响，batch_size最小为2，不能为1。
    #       正常情况下Freeze_batch_size建议为Unfreeze_batch_size的1-2倍。不建议设置的差距过大，因为关系到学习率的自动调整。
    # ----------------------------------------------------------------------------------------------------------------------------#
    # ------------------------------------------------------#
    #   训练参数
    #   Init_Epoch      模型当前开始的训练世代
    #   batch_size      每次输入的图片数量
    #                   受到数据加载方式与triplet loss的影响
    #                   batch_size需要为3的倍数
    #   Epoch           模型总共训练的epoch
    # ------------------------------------------------------#
    batch_size = 96
    Init_Epoch = 0
    Epoch = 100
    # ------------------------------------------------------------------#
    #   其它训练参数：学习率、优化器、学习率下降有关
    # ------------------------------------------------------------------#
    # ------------------------------------------------------------------#
    #   Init_lr         模型的最大学习率
    #   Min_lr          模型的最小学习率，默认为最大学习率的0.01
    # ------------------------------------------------------------------#
    Init_lr = 1e-3
    Min_lr = Init_lr * 0.01
    # ------------------------------------------------------------------#
    #   optimizer_type  使用到的优化器种类，可选的有adam、sgd
    #                   当使用Adam优化器时建议设置  Init_lr=1e-3
    #                   当使用SGD优化器时建议设置   Init_lr=1e-2
    #   momentum        优化器内部使用到的momentum参数
    #   weight_decay    权值衰减，可防止过拟合
    #                   adam会导致weight_decay错误，使用adam时建议设置为0。
    # ------------------------------------------------------------------#
    optimizer_type = "adam"
    momentum = 0.9
    weight_decay = 0
    # ------------------------------------------------------------------#
    #   lr_decay_type   使用到的学习率下降方式，可选的有step、cos
    # ------------------------------------------------------------------#
    lr_decay_type = 'cos'
    # ------------------------------------------------------------------#
    #   save_period     多少个epoch保存一次权值，默认每个世代都保存
    # ------------------------------------------------------------------#
    save_period = 1
    # ------------------------------------------------------------------#
    #   save_dir        权值与日志文件保存的文件夹
    # ------------------------------------------------------------------#
    save_dir = 'logs'
    # ------------------------------------------------------------------#
    #   用于设置是否使用多线程读取数据
    #   开启后会加快数据读取速度，但是会占用更多内存
    #   内存较小的电脑可以设置为2或者0
    # ------------------------------------------------------------------#
    num_workers = 4
    # ------------------------------------------------------------------#
    #   是否开启LFW评估
    # ------------------------------------------------------------------#
    lfw_eval_flag = True
    # ------------------------------------------------------------------#
    #   LFW评估数据集的文件路径和对应的txt文件
    # ------------------------------------------------------------------#
    lfw_dir_path = "lfw"
    lfw_pairs_path = "model_data/lfw_pair.txt"
    ngpus_per_node = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ['LOCAL_RANK'])
        rank = int(os.environ['RNAK'])
        device = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f'[{os.getpid()}] (rank={rank} , local_rank={local_rank} training....')
            print('GPU Device Count :', ngpus_per_node)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank = 0
        rank = 0
    num_classes = get_num_classes(annotation_path)
    # ---------------------------------#
    #   载入模型并加载预训练权重
    # ---------------------------------#

    model = Facenet(backbone=backbone, num_classes=num_classes, pretrained=pretrained)
    if model_path != '':
        # ------------------------------------------------------#
        #   权值文件请看README，百度网盘下载
        # ------------------------------------------------------#
        if local_rank == 0:
            print('load weights {}.'.format(model_path))
        # ------------------------------------------------------#
        #   根据预训练权重的Key和模型的Key进行加载
        # ------------------------------------------------------#
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        # ------------------------------------------------------#
        #   显示没有匹配上的Key
        # ------------------------------------------------------#
        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
            print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")
    loss = triple_loss()
    # ----------------------#
    #   记录Loss
    # ----------------------#
    if local_rank == 0:
        loss_history = LossHistory(save_dir, model, input_shape=input_shape)
    else:
        loss_history = None
    # ------------------------------------------------------------------#
    #   torch 1.2不支持amp，建议使用torch 1.7.1及以上正确使用fp16
    #   因此torch1.2这里显示"could not be resolve"
    # ------------------------------------------------------------------#
    if fp16:
        from torch.cuda.amp import GradScalar as GradScaler
        scaler = GradScaler()
