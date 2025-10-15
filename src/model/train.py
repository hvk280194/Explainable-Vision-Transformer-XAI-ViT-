import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from src.model.vit_model import ViTClassifier
from src.utils.dataset import get_cifar10_loaders




def train_one_epoch(model, loader, opt, loss_fn, device):
    model.train()
    running = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
        running += loss.item() * x.size(0)
    return running / len(loader.dataset)




def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)
        return correct / total




def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = get_cifar10_loaders(batch_size=args.batch_size, img_size=args.img_size)
    model = ViTClassifier(num_classes=10, pretrained=True).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()


    best_acc = 0.0
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, opt, loss_fn, device)
        acc = evaluate(model, test_loader, device)
        print(f'Epoch {epoch+1}/{args.epochs} — train_loss: {train_loss:.4f} — test_acc: {acc:.4f}')
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'checkpoints/best.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--img-size', type=int, default=224)
    args = parser.parse_args()
    main(args)