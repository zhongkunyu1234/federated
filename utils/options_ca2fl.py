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
    
    # CA2FL特有参数
    parser.add_argument("--M", type=int, default=10, help="buffer size M for CA2FL")
    parser.add_argument("--eta", type=float, default=0.01, help="global learning rate")
    parser.add_argument("-i", "--iterations", type=int, default=50, help="number of local epochs")
    
    # 模型参数
    parser.add_argument("-c", "--classes", type=int, default=10, help="number of classes")
    parser.add_argument("--model", type=str, default='mobilenetv2', help="model architecture")
    parser.add_argument("--seed", type=int, default=733, help="random seed")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID, -1 for CPU")
    
    return parser.parse_args()
