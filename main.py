import argparse
import os
import torch
from easydict import EasyDict

from utils.sse import sse_input_path_validated, sse_output_path_validated, sse_error
from utils.yaml_rw import save_yaml
from models.model_factory import get_model
from train.dataset import get_dataloaders
from train.trainer import train_model

def parse_args():
    parser = argparse.ArgumentParser(description='Decision AI Training Framework')
    parser.add_argument('--input_dir', type=str, default='/input', help='input data and model path')
    parser.add_argument('--output_dir', type=str, default='/output', help='output path')
    
    # Model configuration
    parser.add_argument('--model', type=str, default='resnet50', choices=['vgg16', 'resnet50', 'inception_v3'], help='model architecture')
    parser.add_argument('--pretrained', action='store_true', help='use pretrained weights')
    
    # Dataset configuration
    parser.add_argument('--dataset', type=str, default='CIFAR-10', choices=['CIFAR-10', 'imagenet', 'MNIST'], help='dataset name')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    
    # Process type (for different training scenarios)
    parser.add_argument('--process', type=str, default='train', help='process type [train, optimize, develop]')
    
    # Misc
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='device to use')
    parser.add_argument('--cfg_path', type=str, default='./cfgs', help='config save path')

    args = parser.parse_args()
    
    # Support for lowercase environment variables except INPUT_DIR and OUTPUT_DIR
    args.input_dir = os.getenv('INPUT_DIR', args.input_dir)
    args.output_dir = os.getenv('OUTPUT_DIR', args.output_dir)
    
    # These must be lowercase as per requirement
    args.model = os.getenv('model', args.model)
    args.dataset = os.getenv('dataset', args.dataset)
    args.epochs = int(os.getenv('epochs', args.epochs))
    args.batch_size = int(os.getenv('batch_size', args.batch_size))
    args.lr = float(os.getenv('lr', args.lr))
    args.process = os.getenv('process', args.process)
    
    # For backward compatibility with the validator in sse.py which uses .input_path and .output_path
    args.input_path = args.input_dir
    args.output_path = args.output_dir
    
    return args

def main():
    args = parse_args()
    
    # 1. Validate paths
    sse_input_path_validated(args)
    sse_output_path_validated(args)
    
    # 2. Setup configuration
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cfg_path, exist_ok=True)
    
    cfg = EasyDict(vars(args))
    save_yaml(dict(cfg), os.path.join(args.cfg_path, 'config.yaml'))
    
    # 3. Load Model
    pretrained_path = None
    if args.pretrained:
        model_filename = f"{args.model}.pth"
        pretrained_path = os.path.join(args.input_dir, 'model', model_filename)
    
    # Determine number of classes based on dataset
    num_classes = 10
    if args.dataset.lower() == 'imagenet':
        num_classes = 1000
    elif args.dataset.upper() == 'MNIST' or args.dataset.upper() == 'CIFAR-10':
        num_classes = 10
    
    try:
        model = get_model(args.model, num_classes=num_classes, pretrained_path=pretrained_path)
    except Exception as e:
        sse_error(f"模型加载失败: {str(e)}")
        return

    # 4. Load Data
    # Inception v3 requires 299x299
    img_size = 299 if 'inception' in args.model.lower() else 224
    
    try:
        train_loader, test_loader, classes = get_dataloaders(
            args.input_dir,
            dataset_name=args.dataset,
            batch_size=args.batch_size, 
            image_size=img_size
        )
    except Exception as e:
        sse_error(f"数据加载失败: {str(e)}")
        return

    # 5. Execute process
    if args.process in ['train', 'optimize', 'develop']:
        train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=args.device,
            epochs=args.epochs,
            lr=args.lr,
            output_path=args.output_dir,
            model_name=args.model
        )
    else:
        sse_error(f"不支持的处理类型: {args.process}")

if __name__ == '__main__':
    main()
