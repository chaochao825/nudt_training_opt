import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
from utils.sse import sse_print, sse_epoch_progress, save_json_results

def train_model(model, train_loader, test_loader, device, epochs=10, lr=0.001, output_path='./output', model_name='model'):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_acc = 0.0
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }

    sse_print("training_started", {"message": f"开始训练模型: {model_name}"}, progress=20, message="开始训练")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        start_time = time.time()
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Special handling for Inception v3
            if 'inception' in model_name.lower():
                outputs, aux_outputs = model(inputs)
                loss1 = criterion(outputs, labels)
                loss2 = criterion(aux_outputs, labels)
                loss = loss1 + 0.4 * loss2
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if (i + 1) % 10 == 0:
                progress = int(20 + (epoch / epochs) * 70 + (i / len(train_loader)) * (70 / epochs))
                sse_print("batch_update", {
                    "epoch": epoch + 1,
                    "batch": i + 1,
                    "loss": loss.item()
                }, progress=progress, message=f"Epoch {epoch+1}/{epochs} - Batch {i+1}")

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        # Validation
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
        
        val_loss = test_loss / len(test_loader)
        val_acc = 100. * test_correct / test_total
        
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        history['test_loss'].append(val_loss)
        history['test_acc'].append(val_acc)
        
        sse_epoch_progress(epoch + 1, epochs, epoch_type="Epoch")
        sse_print("epoch_summary", {
            "epoch": epoch + 1,
            "train_loss": epoch_loss,
            "train_acc": epoch_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        }, progress=int(20 + ((epoch+1)/epochs) * 70), message=f"Epoch {epoch+1} 完成 - Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(output_path, f'best_{model_name}.pth'))

    # Final result
    final_results = {
        "model_name": model_name,
        "best_accuracy": best_acc,
        "history": history,
        "status": "completed"
    }
    save_json_results(final_results, output_path)
    
    sse_print("training_completed", final_results, progress=100, message="训练任务已完成")
    return history

