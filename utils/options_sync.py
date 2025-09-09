import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    
    # 数据参数
    parser.add_argument("-d", "--dir", type=str, default="../data", help="dir of the data")
    parser.add_argument("--valdir", type=str, default="../data", help="dir of the val data")
    
    # 训练参数
    parser.add_argument("-b", "--batchsize", type=int, default=50, help="batchsize")
    parser.add_argument("-e", "--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("-v", "--interval", type=int, default=10, help="log interval (epochs)")
    parser.add_argument("-n", "--nsplit", type=int, default=40, help="number of split")
    parser.add_argument("-l", "--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--momentum", type=float, default=0, help="momentum")
    
    # FedSync特有参数
    parser.add_argument("--frac", type=float, default=1.0, help="fraction of clients selected each round")
    parser.add_argument("--local_epochs", type=int, default=1, help="number of local epochs")
    parser.add_argument("--lr_decay", type=float, default=0.1, help="learning rate decay rate")
    parser.add_argument("--lr_decay_epoch", type=str, default='50,80', help="learning rate decay epochs")
    parser.add_argument("--min_clients", type=int, default=1, help="minimum number of clients per round")
    parser.add_argument("--aggregation", type=str, default='fedavg', choices=['fedavg', 'fedprox'], 
                       help="aggregation method")
    
    # 模型参数
    parser.add_argument("-c", "--classes", type=int, default=10, help="number of classes")
    parser.add_argument("--model", type=str, default='mobilenetv2', help="model architecture")
    parser.add_argument("--seed", type=int, default=733, help="random seed")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID, -1 for CPU")
    
    return parser.parse_args()