import torch

def model_norm(model_1, model_2):
    device = next(model_1.parameters()).device
    squared_sum = torch.tensor(0.0, device=device)
    for name, layer in model_1.named_parameters():
        other = model_2.state_dict()[name].data.to(device)
        squared_sum += torch.sum(torch.pow(layer.data - other, 2))
    return torch.sqrt(squared_sum).item()
