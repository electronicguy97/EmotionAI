import argparse, torch, os
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from utils.utils import mkdir_model, save_history, save_checkpoint
from preprocess.dataload import create_model, create_data_loaders
from train import train_model

def main(args):
    # GPU settings
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        args.num_device = str(args.gpu) 
    os.environ["CUDA_VISIBLE_DEVICES"]= args.num_device  # Set the GPUs 2 and 3 to use
    device = torch.device(args.device)
    print('Device:', device)
    print('Current cuda device:', torch.cuda.current_device())
    print('Count of using GPUs:', torch.cuda.device_count())

    # Create data loaders
    image_datasets, dataloaders_dict = create_data_loaders(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers,train_name = args.train_name, val_name = args.val_name)

    # Create model and load weights
    model = create_model(args.model_name, args.num_workers)
    model.to(device)
    print(model)

    # Model and training-related settings
    # mkdir_model(args.base_dir, args.model_name, 0)
    # model_folder = args.base_dir + args.model_name + "/"
        
    model_folder = os.path.join(args.base_dir, args.model_name)
    os.makedirs(model_folder, exist_ok=True)
    model_file = model_folder + args.model_name + ".pth"
    train_history = model_folder + args.model_name + "_" + "history_train"
    val_history = model_folder + args.model_name + "_" + "history_val"
    epoch_history = model_folder + args.model_name + '_' + "epoch_his"
    criterion = nn.CrossEntropyLoss()
    if args.optimizer == "Adam":
        optimizer_ft = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2), eps=args.eps)
    elif args.optimizer == "SGD":
        optimizer_ft = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)
    elif args.optimizer == "RMSprop":
        optimizer_ft = optim.RMSprop(model.parameters(), lr=args.learning_rate, alpha=args.alpha)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=args.num_epochs)
    
    
    # Train and evaluate the model
    model, train_hist, val_hist, best_acc, _ , _ = train_model(model, dataloaders_dict, criterion, optimizer_ft, scheduler, device,model_folder=model_folder, batch_size=args.batch_size, num_epochs=args.num_epochs, is_inception=False, train_name = args.train_name, val_name = args.val_name)

    # Save the trained model
    torch.save(model.state_dict(), model_file)

    # Save history data
    save_history(train_hist, train_history)
    save_history(val_hist, val_history)
    

    print("Task completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training and validation of RepVGG model")
    parser.add_argument("--data-dir", type=str, default="../../../data/image", help="Data directory path")
    parser.add_argument('-bs',"--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=7, help="Number of processes for data loader")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use (cuda or cpu)")
    parser.add_argument("--base-dir", type=str, default="./Models", help="Base directory path for model saving")
    parser.add_argument('-m',"--momentum", type=float, default=0.9, help="Momentum")
    # parser.add_argument("--step-size", type=int, default=10, help="Step size for scheduler")
    # parser.add_argument('-g',"--gamma", type=float, default=0.1, help="Gamma value for scheduler")
    parser.add_argument('-e',"--num-epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--num-device", type=str, default='0', help="Number of Using GPU Device")
    parser.add_argument('-tn',"--train-name", type=str, default='train', help="train folder name")
    parser.add_argument('-vn',"--val-name", type=str, default='val', help="val folder nam")
    parser.add_argument('-opt',"--optimizer", type=str, default="Adam", choices=["Adam", "SGD", "RMSprop"], help="Optimizer choice (Adam, SGD, RMSprop)")
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('-b1', '--beta1', type=float, default=0.9, help='Beta1 for Adam optimizer (default: 0.9)')
    parser.add_argument('-b2', '--beta2', type=float, default=0.999, help='Beta2 for Adam optimizer (default: 0.999)')
    parser.add_argument('--eps', type=float, default=1e-8, help='Epsilon for Adam optimizer (default: 1e-8)')
    parser.add_argument('-a', '--alpha', type=float, default=0.99, help='Alpha for RMSprop optimizer (default: 0.99)')
    parser.add_argument('-mn', '--model-name', type=str, default='Repvgg', choices=['Repvgg', 'VIT'],help='Chocie model [Repvgg, VIT]')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use')
    
    args = parser.parse_args()
    main(args)
