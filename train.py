import time
import torch
import copy
import sys
import os
from utils.utils import save_history
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

def train_model(model, dataloaders, criterion, optimizer, lr_scheduler, device, train_name, val_name, batch_size=32, num_epochs=25, is_inception=False,
                is_loaded=False, load_state_ws=None, model_folder="", best_acc=0.0):
    history = {val_name: [], train_name: []}
    loss_history = {val_name: [], train_name: []}
    accuracy_history = {val_name: [], train_name: []}
    precision_history = {val_name: [], train_name: []}
    recall_history = {val_name: [], train_name: []}
    f1_history = {val_name: [], train_name: []}
    r2_history = {val_name: [], train_name: []}

    if is_loaded and load_state_ws is not None:
        state_dict = torch.load(load_state_ws)
        model.load_state_dict(state_dict)
        model.eval()
        print('Model loaded successfully')

    print('Starting Training')
    print('-' * 12)

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = best_acc
    
    for epoch in range(num_epochs):
        epoch_since = time.time()
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 12)
        for phase in [train_name, val_name]:
            total = len(dataloaders[phase])
            current = 0
            if phase == train_name:
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            total_predictions = []
            total_labels = []

            dl = dataloaders[phase]
            totalIm = 0
            for inputs, labels in tqdm(dl):
                totalIm += len(inputs)
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == train_name):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)
                    total_predictions.extend(preds.cpu().numpy())
                    total_labels.extend(labels.cpu().numpy())

                    if phase == train_name:
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / totalIm
            epoch_acc = running_corrects.double() / totalIm
            epoch_precision = precision_score(total_labels, total_predictions, average= None, zero_division=1)
            epoch_recall = recall_score(total_labels, total_predictions, average= None, zero_division=1)
            epoch_f1 = f1_score(total_labels, total_predictions, average= None, zero_division=1)
            epoch_r2 = r2_score(total_labels, total_predictions)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Precision: {epoch_precision.mean():.4f} Recall: {epoch_recall.mean():.4f} F1: {epoch_f1.mean():.4f} R2: {epoch_r2.mean():.4f}')

            if phase == train_name:
                history[train_name].append(epoch_acc)
                loss_history[train_name].append(epoch_loss)
                accuracy_history[train_name].append(epoch_acc)
                precision_history[train_name].append(epoch_precision)
                recall_history[train_name].append(epoch_recall)
                f1_history[train_name].append(epoch_f1)
                r2_history[train_name].append(epoch_r2)
            else:
                history[val_name].append(epoch_acc)
                loss_history[val_name].append(epoch_loss)
                accuracy_history[val_name].append(epoch_acc)
                precision_history[val_name].append(epoch_precision)
                recall_history[val_name].append(epoch_recall)
                f1_history[val_name].append(epoch_f1)
                r2_history[val_name].append(epoch_r2)

            if phase == val_name and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        if lr_scheduler:
            for param_group in optimizer.param_groups:
                lr_scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        interval_epoch = time.time() - epoch_since
        print(f'Epoch {epoch + 1} complete in {interval_epoch // 60:.0f}m {interval_epoch % 60:.0f}s with a learning rate of {lr}')
        print('-' * 12)

        save_training_progress(model_folder, epoch, model, optimizer, history, loss_history, accuracy_history, precision_history, recall_history, f1_history, r2_history)

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val accuracy: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    save_final_results(model_folder, model, optimizer, history, loss_history, accuracy_history, precision_history, recall_history, f1_history, r2_history)

    return model, history[train_name], history[val_name], accuracy_history[train_name], accuracy_history[val_name], best_acc

def save_training_progress(model_folder, epoch, model, optimizer, history, loss_history, accuracy_history, precision_history, recall_history, f1_history, r2_history):
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'loss_history': loss_history,
        'accuracy_history': accuracy_history,
        'precision_history': precision_history,
        'recall_history': recall_history,
        'f1_history': f1_history,
        'r2_history': r2_history
    }
    torch.save(checkpoint, os.path.join(model_folder, f'checkpoint_epoch_{epoch + 1}.pt'))


def save_final_results(model_folder, model, optimizer, history, loss_history, accuracy_history, precision_history, recall_history, f1_history, r2_history):
    torch.save(model.state_dict(), os.path.join(model_folder, 'final_model.pt'))
    torch.save(optimizer.state_dict(), os.path.join(model_folder, 'final_optimizer.pt'))
    save_history(history, os.path.join(model_folder, 'final_history.pt'))
    save_history(loss_history, os.path.join(model_folder, 'final_loss_history.pt'))
    save_history(accuracy_history, os.path.join(model_folder, 'final_accuracy_history.pt'))
    save_history(precision_history, os.path.join(model_folder, 'final_precision_history.pt'))
    save_history(recall_history, os.path.join(model_folder, 'final_recall_history.pt'))
    save_history(f1_history, os.path.join(model_folder, 'final_f1_history.pt'))
    save_history(r2_history, os.path.join(model_folder, 'final_r2_history.pt'))
