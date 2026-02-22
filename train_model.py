import os
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
import torchvision

from dataloader import DataLoaderCreator


    
class ModelCreator(object):
    def __init__(self, args=None):
        self.args = args
        self.model = self.create_model()

    def create_model(self, model_name=None, num_classes=None):
        model_name = model_name or self.args.model
        num_classes = num_classes or self.args.num_classes
        return torchvision.models.resnet18(weights=None, num_classes=num_classes)


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y_super, _ in loader:
            x = x.to(device)
            y_super = y_super.to(device)

            logits = model(x)
            loss = F.cross_entropy(logits, y_super)

            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(dim=1) == y_super).sum().item()
            total += x.size(0)

    return total_loss / total, correct / total


def train_and_save(args, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    ckpt_dir = save_dir + "_epochs"
    os.makedirs(ckpt_dir, exist_ok=True)

    device = torch.device(args.device)

    dlc = DataLoaderCreator(args, auto_create=False)
    train_loader = dlc.train_data_loader(shuffle=True)

    # 🔹 IMPORTANT: use validation loader instead of test loader
    val_loader = dlc.val_data_loader()   # <-- YOU MUST IMPLEMENT THIS

    model = ModelCreator(args).model.to(device)

    opt = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=5e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=args.epochs
    )

    best_val_loss = float("inf")
    patience = 5
    patience_counter = 0

    for epoch in range(args.epochs):

        # -------------------
        # TRAIN
        # -------------------
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for x, y_super, _ in tqdm(train_loader, desc=f"epoch {epoch}"):
            x = x.to(device)
            y_super = y_super.to(device)

            logits = model(x)
            loss = F.cross_entropy(logits, y_super)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(dim=1) == y_super).sum().item()
            total += x.size(0)

        train_loss = total_loss / total
        train_acc = correct / total

        scheduler.step()

        # -------------------
        # VALIDATION
        # -------------------
        val_loss, val_acc = evaluate(model, val_loader, device)

        print(
            f"Train: loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"Val: loss={val_loss:.4f} acc={val_acc:.4f}"
        )

        # -------------------
        # Save Best Model
        # -------------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_dir + ".pt")
            print("✅ Saved new best model")
        else:
            patience_counter += 1

        # -------------------
        # Early Stopping
        # -------------------
        if patience_counter >= patience:
            print("⛔ Early stopping triggered")
            break

        # optional: save all epochs
        torch.save(model.state_dict(),
                   os.path.join(ckpt_dir, f"epoch{epoch}.pt"))

    print("Best validation loss:", best_val_loss)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="imbalancedsupercifar100")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--num_classes", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.1)
    args = parser.parse_args()

    # IMPORTANT: match their path conventions
    save_base = f"save1/001_train_model/{args.model}_{args.dataset_name}"
    train_and_save(args, save_base)

if __name__ == "__main__":
    main()
