from torchvision import transforms
from torch.utils.data import DataLoader
from model.repvgg import repvgg_model_convert, create_RepVGG_A0
import os, torch
from .dataset import CustomDataset
from tqdm import tqdm
import timm
import torch.nn as nn

def create_data_loaders(data_dir, batch_size, num_workers, train_name, val_name):
    # Define data transformations for training and validation datasets
    data_transforms = {
        train_name: transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        val_name: transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Create training and validation datasets
    image_datasets = {x: CustomDataset(os.path.join(data_dir, x), transform=data_transforms[x]) for x in [train_name, val_name]}

    # Create training and validation data loaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True) for x in [train_name, val_name]}

    # Print the number of images in the training dataset
    print("Number of images in train dataset:", len(image_datasets[train_name]))
    # Print some sample images and labels
    for phase in [train_name, val_name]:
        print(f"Creating {phase} dataloader...")
        with tqdm(total=len(dataloaders_dict[phase]), desc=f"Loading {phase} data", unit='batch') as pbar:
            for images, labels in dataloaders_dict[phase]:
                pbar.update(1)

    # Print the shape of a batch of images and labels from the training dataset
    for images, labels in dataloaders_dict[train_name]:
        print("Image batch shape:", images.shape)
        print("Label batch shape:", labels.shape)
        # Print some sample images and labels
        for img, label in zip(images, labels):
            print(img)
            print("Image shape:", img.shape)
            print("Label:", label)
        
            break 
        break 
    return image_datasets, dataloaders_dict

def create_model(model_name, num_class):
    # Create the RepVGG-A0 model and load pre-trained weights
    if model_name == 'Repvgg':
        model = create_RepVGG_A0()
    elif model_name == 'VIT':
        model_vit_base = 'vit_base_patch16_224'
        # model_name = 'resnet18'
        # loading pretrained model.
        model = timm.create_model(model_vit_base, pretrained=False)
        model.head = nn.Linear(768, num_class)
    else:
        raise ValueError("Invalid model name. Please provide a valid model name.")
    
    return model
