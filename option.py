import argparse

def get_args():
    parser = argparse.ArgumentParser()

    # model architecture & checkpoint
    parser.add_argument('--checkpoint_dir', type=str, default='data_hdd/hoseong/checkpoint_onlytrain')  #保存模型路径
    parser.add_argument('--checkpoint_name', type=str, default='')    #模型名
    # data loading
    parser.add_argument('--num_workers', type=int, default=0)    #多线程
    parser.add_argument('--seed', type=int, default=2021, help='random seed')  #随机数种子
    # training hyper parameters
    parser.add_argument('--batch_size', type=int, default=10)     
    parser.add_argument('--epochs', type=int, default=1000)        #训练次数
    # optimzier & learning rate scheduler
    parser.add_argument('--learning_rate', type=float, default=0.001)  #学习率
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--optimizer', default='ADAM', choices=('SGD', 'ADAM'),   #优化器
                        help='optimizer to use (SGD | ADAM )')
    parser.add_argument('--decay_type', default='cosine_warmup', choices=('step', 'step_warmup', 'cosine_warmup'),
                        help='optimizer to use (step | step_warmup | cosine_warmup)')
    parser.add_argument('--lr_change', default='YES', choices=('YES', 'NO'))#优化器
    args = parser.parse_args()
    return args
if __name__ == "__main__":
    get_args()
