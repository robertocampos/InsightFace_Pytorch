from config import get_config
from Learner import face_learner
import argparse

# python train.py -net ir_se -b 64 -e 50 -w ir_se50.pth

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("-e", "--epochs", help="training epochs", default=20, type=int)
    parser.add_argument("-net", "--net_mode", help="which network, [ir, ir_se, mobilefacenet]",default='ir_se', type=str)
    parser.add_argument("-depth", "--net_depth", help="how many layers [50,100,152]", default=50, type=int)
    parser.add_argument('-lr','--lr',help='learning rate',default=1e-3, type=float)
    parser.add_argument("-b", "--batch_size", help="batch_size", default=128, type=int)
    #parser.add_argument("-w", "--num_workers", help="workers number", default=3, type=int)
    parser.add_argument("-d", "--data_mode", help="use which database, [vgg, ms1m, emore, concat]",default='emore', type=str)
    parser.add_argument("-w", "--weight_file", help="weight file name",default='', type=str)
    args = parser.parse_args()

    conf = get_config()
    
    if args.net_mode == 'mobilefacenet':
        conf.use_mobilfacenet = True
    else:
        conf.net_mode = args.net_mode
        conf.net_depth = args.net_depth    
    
    conf.lr = args.lr
    conf.batch_size = args.batch_size
    conf.data_mode = args.data_mode
    learner = face_learner(conf)
    
    if args.weight_file:
        learner.load_state(conf, args.weight_file, model_only=True, from_save_folder=False)

    learner.train(conf, args.epochs)
    
    