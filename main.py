import dataloader
import model
import argparse
import torch
import torch.optim as optim
import numpy as np
from datetime import datetime
import random
import os
import metrics

from model import GEncoder 
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)  # gpu
    torch.backends.cudnn.deterministic = True


class MarginLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super(MarginLoss, self).__init__()
        self.margin = margin

    def forward(self, pos_pred, neg_pred):
        loss = torch.mean(torch.relu(self.margin - (pos_pred - neg_pred)))
        return loss


# Mixed Loss 
class MixedLoss(torch.nn.Module):
    def __init__(self, alpha=0.5, margin=1.0):  # positive_weight
        super(MixedLoss, self).__init__()
        self.alpha = alpha 
        self.margin_loss = MarginLoss(margin=margin)

    def forward(self, pos_pred, neg_pred):
        positive_loss = torch.mean(-torch.log(pos_pred + 1e-10))  
        negative_loss = torch.mean(torch.exp(neg_pred) - 1)  # Negative Loss
        margin_loss = self.margin_loss(pos_pred, neg_pred)  
        loss = self.alpha * (positive_loss + negative_loss) + (1 - self.alpha) * margin_loss
        return loss


def train(train_loader, epoch_id, lr, weight_decay, type_m='user'):
    rec_model.train()
    optimizer = optim.Adam(rec_model.parameters(), lr, weight_decay=weight_decay)
    losses = []

    for _, (u, pi_ni) in enumerate(train_loader):
        users, pos, neg = u.to(device), pi_ni[:, 0].to(device), pi_ni[:, 1].to(device)
        if type_m == 'user':
            pos_prediction = rec_model(None, users, pos)
            neg_prediction = rec_model(None, users, neg)
        else:
            pos_prediction = rec_model(users, None, pos)
            neg_prediction = rec_model(users, None, neg)

        rec_model.zero_grad()
        
        loss = loss_fn(pos_prediction, neg_prediction)  #loss function
        if args.beta > 0 and type_m == 'group':
            loss += args.beta * rec_model.compute_contrastive_loss(users, pos, args.threshold, args.temperature, args.mask_ratio, args.passes)

        losses.append(loss.detach().cpu())
        loss.backward()
        optimizer.step()
    
    
    print(f"[Epoch {epoch_id}] {type_m} loss: {torch.mean(torch.stack(losses)):.5f}")
    


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, help="[Mafengwo, CAMRa2011]", default="CAMRa2011")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--device", type=str, help="[cuda:0, ..., cpu]", default="cuda:0")

parser.add_argument("--emb_dim", type=int, default=32)
parser.add_argument("--epoch", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--drop_ratio", type=float, default=0.2)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--weight_decay", type=float, default=0.0)
parser.add_argument("--topK", type=list, default=[1, 5, 10])
parser.add_argument("--num_negatives", type=int, default=2)
parser.add_argument("--num_layers", type=int, default=3)
parser.add_argument("--num_heads", type=int, default=4)
parser.add_argument("--alpha", type=float, default=0.5)
parser.add_argument("--beta", type=float, default=0.025)
parser.add_argument("--margin", type=float, default=1.0)
parser.add_argument("--threshold", type=int, default=5)
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--mask_ratio", type=float, default=0.8)
parser.add_argument("--passes", type=int, default=1)


args = parser.parse_args()
set_seed(args.seed)
device = torch.device(args.device)

print('= ' * 20)
print('## Starting Time:', datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)
print(args)

#dataset load
dataset = dataloader.GroupDataset(num_negatives=args.num_negatives, dataset=args.dataset, device = device)
group_member_dict = dataset.group_member_dict


rec_model = GEncoder(
    num_users=dataset.num_users,
    num_items=dataset.num_items,
    num_groups=dataset.num_groups,
    emb_dim=args.emb_dim,
    group_member_dict=group_member_dict,
    drop_ratio=args.drop_ratio,
    num_layers=args.num_layers,
    num_heads=args.num_heads
)
rec_model.to(device)
dataset.model = rec_model  

loss_fn = MixedLoss(alpha=args.alpha, margin=args.margin)
args.mask_ratio = 1 - args.mask_ratio

# Training Loop
best = 0
for epoch in range(args.epoch):
    print(f"Starting Epoch {epoch}", flush=True)  
    train(dataset.get_user_dataloader(args.batch_size), epoch, args.lr, args.weight_decay, type_m='user')
    train(dataset.get_group_dataloader(args.batch_size), epoch, args.lr, args.weight_decay, type_m='group')

    u_hits, u_ndcgs = metrics.evaluate(rec_model, dataset.user_test_ratings, dataset.user_test_negatives, device, args.topK, type_m='user')
    print(f"[Epoch {epoch}] User, Hit@{args.topK}: {u_hits}, NDCG@{args.topK}: {u_ndcgs}")

    g_hits, g_ndcgs = metrics.evaluate(rec_model, dataset.group_test_ratings, dataset.group_test_negatives, device, args.topK, type_m='group')
    print(f"[Epoch {epoch}] Group, Hit@{args.topK}: {g_hits}, NDCG@{args.topK}: {g_ndcgs}")
    print()
    if best < g_hits[-1]:
        best = g_hits[-1]
        best_scores = [u_hits, u_ndcgs, g_hits, g_ndcgs]

print()
print('## Finishing Time:', datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)
print('= ' * 20)
print('Best Scores: User Hit:', best_scores[0], "User NDCG:", best_scores[1],'Group Hit:', best_scores[2], "Group NDCG:",best_scores[3])
print('= ' * 20)
print("Done!")
