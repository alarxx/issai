import torch

def save(model,
            optimizer=None,
            train_losses=None,
            test_losses=None,
            train_accuracies=None,
            test_accuracies=None,
            index=-1,
            filename="checkpoint"):
    savings = {
        'model_state_dict': model.state_dict()
    }
    if optimizer is not None:
        savings["optimizer_state_dict"] = optimizer.state_dict()
    if train_losses is not None:
        savings['train_losses'] = train_losses
    if test_losses is not None:
        savings['test_losses'] = test_losses
    if train_accuracies is not None:
        savings['train_accuracies'] = train_accuracies
    if test_accuracies is not None:
        savings['test_accuracies'] = test_accuracies
    if index >= 0:
        filename = f"{filename}_{index}"
    torch.save(savings, f"{filename}.pth")
    print(f"Saved in file: {filename}.pth")
