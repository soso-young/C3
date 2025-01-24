import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class GEncoder(nn.Module):
    def __init__(self, num_users, num_items, emb_dim, group_member_dict, num_groups, drop_ratio=0.2, num_layers=2, num_heads=2):
        super(GEncoder, self).__init__()

        self.user_embedding = nn.Embedding(num_users, emb_dim)
        self.item_embedding = nn.Embedding(num_items, emb_dim)
        
        self.group_member_dict = group_member_dict 
        
        # LayerNorm, Dropout
        self.layer_norm = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(drop_ratio)

        # Transformer Encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads, dropout=drop_ratio, batch_first=True),
            num_layers=num_layers)

        # Output Layer
        self.output_layer = nn.Linear(emb_dim, 1)

      
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)

    def user_forward(self, users, items):
        users_embed = self.user_embedding(users) 
        items_embed = self.item_embedding(items) 
        bsz, emb_dim = users_embed.size()         
        
        combined_embed = torch.cat((users_embed, items_embed), dim=1).view(bsz, 2, emb_dim)  # (batch_size, 2, emb_dim)


        # Transformer Encoder
        transformer_output = self.transformer(combined_embed)
        transformer_output = transformer_output.mean(dim=1)
        
        return torch.sigmoid(self.output_layer(transformer_output))  # (batch_size, 1)
    
    def group_forward(self, groups, items, return_output=False):
        device = items.device
        batch_size = len(groups)
        group_embs = []
        
        group_ids = groups.tolist()
        item_ids = items.tolist()
        
        max_members = max(len(self.group_member_dict[group_id]) for group_id in group_ids)
        
        members_embed_tensor = torch.zeros((batch_size, max_members, self.user_embedding.embedding_dim)).to(device)
        items_embed_tensor = torch.zeros((batch_size, 1, self.item_embedding.embedding_dim)).to(device)
        masks = torch.zeros((batch_size, max_members+1)).to(device)
        
        for i, (group_id, item_id) in enumerate(zip(group_ids, item_ids)):
            members = self.group_member_dict[group_id]
            num_members = len(members)
            
            members_embed = self.user_embedding(torch.LongTensor(members).to(device))
            items_embed = self.item_embedding(torch.LongTensor([item_id]).to(device))
            
            members_embed_tensor[i, :num_members, :] = members_embed
            items_embed_tensor[i, :, :] = items_embed
            masks[i, :num_members] = 1
            masks[i, -1] = 1
        
        combined_embed = torch.cat((members_embed_tensor, items_embed_tensor), dim=1)
        combined_embed = self.layer_norm(self.dropout(combined_embed))
        
        transformer_output = self.transformer(combined_embed, src_key_padding_mask=(masks == 0))
        transformer_output = (transformer_output * masks.unsqueeze(2)).mean(dim=1)
        
        #test
        # print(transformer_output)
        output = torch.sigmoid(self.output_layer(transformer_output))
        
        if return_output:
            return output, transformer_output
        
        return output
       
       
    def forward(self, groups, users, items, return_output=False):
        if groups is not None:
            return self.group_forward(groups, items, return_output)
        return self.user_forward(users, items)

    def compute_contrastive_loss(self, groups, items, threshold=5, temperature=1.0, mask_ratio=0.2, passes=True):
        device = items.device
        batch_size = groups.size(0)
        
        group_ids = groups.tolist()
        item_ids = items.tolist()
        max_members = max(len(self.group_member_dict[group_id]) for group_id in group_ids)
        
        members_embed_tensor = torch.zeros((batch_size, max_members, self.user_embedding.embedding_dim)).to(device)
        items_embed_tensor = torch.zeros((batch_size, 1, self.item_embedding.embedding_dim)).to(device)
        masks = torch.zeros((batch_size, max_members+1)).to(device)
        
        members_embed_tensor2 = torch.zeros((batch_size, max_members, self.user_embedding.embedding_dim)).to(device)
        items_embed_tensor2 = torch.zeros((batch_size, 1, self.item_embedding.embedding_dim)).to(device)
        masks2 = torch.zeros((batch_size, max_members+1)).to(device)
        
        cnt=0
        for i, (group_id, item_id) in enumerate(zip(group_ids, item_ids)):
            members = self.group_member_dict[group_id]
            num_members = len(members)
            
            if num_members > threshold:
                augmented_members = random.sample(members, int(num_members * mask_ratio))
                augmented_members2 = random.sample(members, int(num_members * mask_ratio))
                
            elif passes:
                augmented_members = members
                augmented_members2 = members
                
            else:
                continue
            
            cnt+=1
            num_augmented_members = len(augmented_members)
            
            aug_members_embed = self.user_embedding(torch.LongTensor(augmented_members).to(device))
            items_embed = self.item_embedding(torch.LongTensor([item_id]).to(device))
            
            aug_members_embed2 = self.user_embedding(torch.LongTensor(augmented_members2).to(device))
            items_embed2 = self.item_embedding(torch.LongTensor([item_id]).to(device))
            
            members_embed_tensor[i, :num_augmented_members, :] = aug_members_embed
            items_embed_tensor[i, :, :] = items_embed
            masks[i, :num_members] = 1
            masks[i, -1] = 1
            
            members_embed_tensor2[i, :num_augmented_members, :] = aug_members_embed2
            items_embed_tensor2[i, :, :] = items_embed2
            masks2[i, :num_members] = 1
            masks2[i, -1] = 1
        
        
        members_embed_tensor = members_embed_tensor[:cnt, :, :]
        members_embed_tensor2 = members_embed_tensor2[:cnt, :, :]
        
        items_embed_tensor = items_embed_tensor[:cnt, :, :]
        items_embed_tensor2 = items_embed_tensor2[:cnt, :, :]
        
        masks = masks[:cnt, :]
        masks2 = masks2[:cnt, :]
        batch_size = len(members_embed_tensor)
        
        combined_embed = torch.cat((members_embed_tensor, items_embed_tensor), dim=1)
        combined_embed = self.layer_norm(self.dropout(combined_embed))
        combined_embed2 = torch.cat((members_embed_tensor2, items_embed_tensor2), dim=1)
        combined_embed2 = self.layer_norm(self.dropout(combined_embed2))
        
        transformer_output = self.transformer(combined_embed, src_key_padding_mask=(masks == 0))
        transformer_output = (transformer_output * masks.unsqueeze(2)).mean(dim=1)
        transformer_output2 = self.transformer(combined_embed2, src_key_padding_mask=(masks2 == 0))
        transformer_output2 = (transformer_output2 * masks2.unsqueeze(2)).mean(dim=1)
        
        embeddings = torch.cat([transformer_output, transformer_output2], dim=0)
        sim_matrix = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)

        mask = torch.eye(batch_size * 2, dtype=torch.bool).to(device)
        sim_matrix = sim_matrix.masked_fill(mask, -1e9)
        
        logits = sim_matrix / temperature
        targets = torch.cat([torch.arange(batch_size, 2* batch_size), torch.arange(0, batch_size)], dim=0).to(device)
        
        loss_fn = nn.CrossEntropyLoss()
        contrastive_loss = loss_fn(logits, targets) 
        
        return contrastive_loss
        