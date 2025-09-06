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
    
    # FedAsync特有参数
    parser.add_argument("--rho", type=float, default=0.00001, help="regularization × lr")
    parser.add_argument("--alpha", type=float, default=0.9, help="mixing hyperparameter")
    parser.add_argument("-i", "--iterations", type=int, default=50, help="number of local epochs")
    parser.add_argument("--alpha-decay", type=float, default=0.5, help="alpha decay rate")
    parser.add_argument("--alpha-decay-epoch", type=str, default='400', help="alpha decay epoch")
    parser.add_argument("--alpha-adaptive", type=float, default=0, help="adaptive mixing hyperparameter")
    parser.add_argument("--alpha-adaptive2", type=float, default=1, help="adaptive mixing hyperparameter2")
    parser.add_argument("--alpha-type", type=str, default='none', help="type of adaptive alpha")
    parser.add_argument("--max-delay", type=int, default=10, help="maximum of global delay")
    
    # 模型参数
    parser.add_argument("-c", "--classes", type=int, default=10, help="number of classes")
    parser.add_argument("--model", type=str, default='mobilenetv2', help="model architecture")
    parser.add_argument("--seed", type=int, default=733, help="random seed")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID, -1 for CPU")
    
    return parser.parse_args()