import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader


def get_cifar10_loaders(batch_size=64, img_size=224, num_workers=2):
    transform = T.Compose([
    T.Resize((img_size, img_size)),
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader