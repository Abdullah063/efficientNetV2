import torch.nn as nn
import torch.optim as optim
import pandas as pd
from datetime import datetime
from config import *
from dataset import get_dataloaders
from model import get_model
from train import train_epoch, evaluate


def main():
    # Veri yükleyicileri
    train_loader, val_loader, test_loader, classes = get_dataloaders()
    print(f"Sınıf sayısı: {len(classes)}")
    print(f"Sınıflar: {classes}")

    # Model, loss ve optimizer
    model = get_model(len(classes))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, verbose=True)

    # Sonuçları kaydetmek için sözlük
    results = {
        'epoch': [], 'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'learning_rate': []
    }

    # Eğitim döngüsü
    best_val_acc = 0
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        print(f'\nEpoch {epoch + 1}/{NUM_EPOCHS}')

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Sonuçları kaydet
        results['epoch'].append(epoch + 1)
        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        results['val_loss'].append(val_loss)
        results['val_acc'].append(val_acc)
        results['learning_rate'].append(current_lr)

        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f'Early stopping triggered after epoch {epoch + 1}')
            break

    # Sonuçları kaydet
    results_df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df.to_csv(f'training_results_{timestamp}.csv', index=False)

    # Test değerlendirmesi
    model.load_state_dict(torch.load('best_model.pth'))
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print(f'\nTest Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%')

    with open(f'test_results_{timestamp}.txt', 'w') as f:
        f.write(f'Test Loss: {test_loss:.4f}\n')
        f.write(f'Test Accuracy: {test_acc:.2f}%\n')


if __name__ == '__main__':
    main()