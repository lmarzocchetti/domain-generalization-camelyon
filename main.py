import sys
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from wilds.datasets.wilds_dataset import WILDSSubset
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader, get_eval_loader
import torchvision
import torchvision.transforms as transforms

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")

def set_seed(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

class TripletDataset():
    def __init__(self, dataset: WILDSSubset):
        self.dataset = dataset
        self.collate = dataset.collate
        self.domains = [0, 3, 4]
        self.label_domain_dict: dict[tuple[int, int], pd.DataFrame] = {}
        
        self.reset_array_filtering()
    
    def reset_array_filtering(self):
        frame_for_filtering = pd.DataFrame(
            np.array([np.array([i, y, meta[0]]) for (i, (y, meta)) in enumerate(zip(self.dataset.y_array, self.dataset.metadata_array))]), 
            columns=["idx", "label", "domain"]
        )
        self.label_domain_dict[(0, 0)] = frame_for_filtering.loc[(frame_for_filtering["label"] == 0) & (frame_for_filtering["domain"] == 0)]
        self.label_domain_dict[(0, 3)] = frame_for_filtering.loc[(frame_for_filtering["label"] == 0) & (frame_for_filtering["domain"] == 3)]
        self.label_domain_dict[(0, 4)] = frame_for_filtering.loc[(frame_for_filtering["label"] == 0) & (frame_for_filtering["domain"] == 4)]
        self.label_domain_dict[(1, 0)] = frame_for_filtering.loc[(frame_for_filtering["label"] == 1) & (frame_for_filtering["domain"] == 0)]
        self.label_domain_dict[(1, 3)] = frame_for_filtering.loc[(frame_for_filtering["label"] == 1) & (frame_for_filtering["domain"] == 3)]
        self.label_domain_dict[(1, 4)] = frame_for_filtering.loc[(frame_for_filtering["label"] == 1) & (frame_for_filtering["domain"] == 4)]
    
    def get_positive(self, label: int, domain: int):
        new_domain: int = random.choice([d for d in self.domains if d != domain])
        
        positive = self.label_domain_dict[(label, new_domain)].sample()["idx"].item()
        
        return self.dataset.__getitem__(positive)
        
    def get_negative(self, label: int, domain: int):
        negative = self.label_domain_dict[(1 - label, domain)].sample()["idx"].item()
        
        return self.dataset.__getitem__(negative)
    
    def __getitem__(self, idx):
        (x_anchor, y_anchor, meta_anchor) = self.dataset.__getitem__(idx)
        label = y_anchor.item()
        domain = meta_anchor[0].item()
        x_positive, _, _ = self.get_positive(label, domain)
        x_negative, _, _ = self.get_negative(label, domain)
        return (x_anchor, y_anchor), x_positive, x_negative
    
    def __len__(self):
        return len(self.dataset)
    
class TripletModel(nn.Module):
    def __init__(self, backbone: torchvision.models.ResNet) -> None:
        super().__init__()
        self.feature_extractor = backbone
        self.cl_head = torch.nn.Sequential(
            # nn.LayerNorm(1024, bias=False),
            nn.Linear(2048, 4096, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(4096),
            nn.Linear(4096, 1024, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256, bias=False),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor = None, negative: torch.Tensor = None):
        anchor_ft = self.feature_extractor(anchor)
        if positive is None or negative is None:
            return self.cl_head(anchor_ft)
        positive_ft = self.feature_extractor(positive)
        negative_ft = self.feature_extractor(negative)
        return anchor_ft, positive_ft, negative_ft, self.cl_head(anchor_ft)

class ClassificationModel():
    def __init__(self, feature_extractor: torchvision.models.ResNet, learning_rate, margin=15.0):
        super(ClassificationModel, self).__init__()
        self.margin = margin
        if torch.cuda.device_count() > 1:
            print("Using Data Parallel")
            self.cl_head = nn.DataParallel(TripletModel(feature_extractor)).to(device)
        else:
            self.cl_head = TripletModel(feature_extractor).to(device)
        self.optimizer = torch.optim.AdamW(self.cl_head.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.1)
        self.triplet = torch.nn.TripletMarginLoss(margin=margin)

    def train(self, train_loader, epoch):
        self.cl_head.train()
        if epoch < 5:
            if isinstance(self.cl_head, nn.DataParallel):
                self.cl_head.module.feature_extractor.eval()
                self.cl_head.module.feature_extractor.requires_grad_(False)
            else:
                self.cl_head.feature_extractor.eval()
                self.cl_head.feature_extractor.requires_grad_(False)
        else:
            if isinstance(self.cl_head, nn.DataParallel):
                self.cl_head.module.feature_extractor.requires_grad_(True)
            else:
                self.cl_head.feature_extractor.requires_grad_(True)
        for i, batch in enumerate(train_loader):
            anchor, positive, negative = batch
            (x_anchor_p, y_anchor_p) = anchor
            x_positive_p = positive
            x_negative_p = negative
            
            (x_anchor, y_anchor) = x_anchor_p.to(device), y_anchor_p.float().to(device)
            x_positive = x_positive_p.to(device)
            x_negative = x_negative_p.to(device)
            self.optimizer.zero_grad()
            anchor_features, positive_features, negative_features, output = self.cl_head(x_anchor, x_positive, x_negative)
            triplet_loss_val = self.triplet(anchor_features, positive_features, negative_features) * 100
            ce_loss_val = torch.nn.functional.binary_cross_entropy(output.squeeze(), y_anchor, reduction='sum')
            total_loss = triplet_loss_val + ce_loss_val
            total_loss.backward()
            self.optimizer.step()
            if i % 10 == 0:
                print(f"Epoch {epoch}, Batch {i}, Triplet Loss: {triplet_loss_val}, CE Loss: {ce_loss_val}, Total Loss: {total_loss}")
                sys.stdout.flush()
                
    
    def test(self, test_loader, epoch):
        self.cl_head.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for batch in enumerate(test_loader):
                (_, (x, y, _)) = batch
                x, y = x.to(device), y.to(device)
                output = self.cl_head(x)
                test_loss += torch.nn.functional.binary_cross_entropy(output.squeeze(), y.squeeze().float(), reduction='sum').item()
                pred = output.round()
                correct += pred.eq(y.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        print(f"Epoch {epoch}, Test Loss: {test_loss}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.2f}%)")
    
    def save_model(self, path: str, epoch: int):
        print(f"Saving model at epoch {epoch} to {path}")
        if isinstance(self.cl_head, nn.DataParallel):
            torch.save({
                "epoch": epoch,
                "model_state_dict": self.cl_head.module.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                }, path)
        else:
            torch.save({
                "epoch": epoch,
                "model_state_dict": self.cl_head.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                }, path)
    
    def load_model(self, path: str):
        print(f"Loading model from {path}")
        checkpoint = torch.load(path)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if isinstance(self.cl_head, nn.DataParallel):
            self.cl_head.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.cl_head.load_state_dict(torch.load(path))
        EPOCH = checkpoint['epoch']


def resnet_for_feature_extraction() -> torchvision.models.ResNet:
    model = torchvision.models.resnet50(weights='DEFAULT')
    model.fc = torch.nn.Identity()
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser(description="Camelyon17 Training with Triplet Loss")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch Size for Training")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of Epochs to Train")
    parser.add_argument("--checkpoint_epochs", type=int, default=1, help="Number of Epochs between Checkpoints")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning Rate")
    parser.add_argument("--seed", type=int, default=42, help="Random Seed")
    parser.add_argument("--resume_training", type=str, default=None, help="Path to Model to Resume Training")
    parser.add_argument("--save_model_path", type=str, default="", help="Path to Save Model (specify with / at the end)")
    parser.add_argument("--root_dataset", type=str, default="/work/ai4bio2023/lmarzocchetti/data/camelyon17/camelyon_labeled", help="Path to Root Dataset")
    parser.add_argument("--download", type=bool, default=False, help="Download Dataset")
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    dataset = get_dataset(dataset="camelyon17", root_dir=args.root_dataset, download=args.download)
    train_data = dataset.get_subset(
        "train",
        transform=transforms.Compose(
            [
                transforms.RandomRotation(90),
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )
    )
    trplt_dataset = TripletDataset(train_data)
    train_loader = get_train_loader("standard", trplt_dataset, batch_size=args.batch_size)
    
    val_data = dataset.get_subset(
        "val",
        transform=transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )
    )
    val_loader = get_eval_loader("standard", val_data, batch_size=args.batch_size)

    densenet_extractor = resnet_for_feature_extraction()
    model = ClassificationModel(densenet_extractor, args.learning_rate)
    
    if args.resume_training is not None:
        model.load_model(args.resume_training)

    for epoch in range(1, args.num_epochs + 1):
        print(f"Starting Epoch {epoch}")
        model.train(train_loader, epoch)
        
        if epoch % args.checkpoint_epochs == 0:
           model.save_model(f"{args.save_model_path}model_epoch_{epoch}.pt", epoch)
        
        model.test(val_loader, epoch)
        model.scheduler.step()
        trplt_dataset.reset_array_filtering()

    test_data = dataset.get_subset(
        "test",
        transform=transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )
    )
    test_loader = get_eval_loader("standard", test_data, batch_size=args.batch_size)
    
    print("Finished Training, Starting Testing")
    model.test(test_loader, epoch)
    
if __name__ == "__main__":
    main()
