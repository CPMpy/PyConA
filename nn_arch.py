import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from pycona.benchmarks import *
from pycona.predictor.feature_representation import FeaturesRelDim
import numpy as np
from sklearn.tree import DecisionTreeClassifier


from pycona.utils import get_scope, get_var_name, get_var_dims

def featurize_constraint(c):

    con_name = c.name
    scope = get_scope(c)
    var_names = [get_var_name(var) for var in scope]
    vars_dims = [get_var_dims(var) for var in scope]


    return (con_name, [(var_name, var_indices) for var_name, var_indices in zip(var_names, vars_dims)])


# ----------------------------
# 1. Constraint Tokenizer with global variable embeddings
# ----------------------------
class ConstraintTokenizer:
    def __init__(self, relations, max_index=50):
        """
        relations: list of strings, e.g. ["==","!="]
        max_index: maximum index value in constraints
        """
        self.relation2id = {r: i for i, r in enumerate(relations)}
        # index and relation spaces are offset after variables and a reserved PAD=0
        self.index_offset = 100  # additional spacing before index tokens per variable space
        self.vocab_size_indices = self.index_offset + max_index + 1
        self.relation_vocab_size = len(relations)
    
    def encode(self, constraint, var2id):
        """
        constraint: (relation:str, list of variables)
            each variable: (name:str, list of indices)
        var2id: dict mapping variable names (global per problem) -> IDs
        returns: list of integer tokens
        """
        if not isinstance(constraint, tuple) or len(constraint) != 2:
            raise ValueError(f"Expected constraint to be tuple of (relation, variables), got {type(constraint)}")
            
        relation, variables = constraint
        if relation not in self.relation2id:
            raise ValueError(f"Unknown relation: {relation}. Expected one of {list(self.relation2id.keys())}")
            
        # Reserve 0 for PAD. Variable IDs occupy [1, var_vocab_size].
        # Index tokens start at 1 + var_vocab_size + index_offset.
        # Relation tokens start at 1 + var_vocab_size + vocab_size_indices.
        var_vocab_size = len(var2id)
        var_base = 1
        index_base = 1 + var_vocab_size
        relation_base = 1 + var_vocab_size + self.vocab_size_indices

        tokens = []
        for var in variables:  # list of variables
            if not isinstance(var, tuple) or len(var) != 2:
                raise ValueError(f"Expected variable to be tuple of (name, indices), got {var}")
                
            var_name, indices = var
            if var_name not in var2id:
                raise ValueError(f"Unknown variable name: {var_name}")
                
            tokens.append(var_base + var2id[var_name])              # global variable embedding ID (offset by 1)
            for idx in indices:
                if not isinstance(idx, int):
                    raise ValueError(f"Expected integer index, got {type(idx)}")
                tokens.append(index_base + self.index_offset + idx)   # index tokens after var space
                
        # append relation token
        tokens.append(relation_base + self.relation2id[relation])
        return tokens

    def get_vocab_size(self, var_vocab_size):
        """
        total vocab = variable IDs + indices + relation tokens
        """
        # +1 for PAD at index 0
        return 1 + var_vocab_size + self.vocab_size_indices + self.relation_vocab_size


# ----------------------------
# 2. Transformer Encoder
# ----------------------------
class ConstraintEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, nhead=4, nlayers=2, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        # Learned positional embeddings to help order-sensitive relations
        self.pos_embed = nn.Embedding(512, emb_dim)  # supports sequences up to 512 tokens
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=nhead, dim_feedforward=256, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, nlayers)

    def forward(self, token_batch):
        """
        token_batch: [B, L] tensor
        returns: [B, D] embeddings
        """
        # Embedding + mask
        x = self.embed(token_batch)        # [B, L, D]
        # Positional embeddings (clamped to max length supported)
        B, L = token_batch.size(0), token_batch.size(1)
        positions = torch.arange(L, device=token_batch.device).clamp(max=self.pos_embed.num_embeddings - 1)
        pos = self.pos_embed(positions).unsqueeze(0).expand(B, L, -1)
        x = x + pos
        pad_mask = token_batch.eq(0)       # [B, L]
        x = self.transformer(x, src_key_padding_mask=pad_mask)
        # Masked mean pooling
        lengths = (~pad_mask).sum(dim=1).clamp(min=1).unsqueeze(-1)  # [B,1]
        x = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        x = x.sum(dim=1) / lengths
        # Stabilize with dropout and L2 normalization
        x = self.dropout(x)
        x = F.normalize(x, p=2, dim=1)
        return x


# ----------------------------
# 2b. Slot-based Constraint Encoder (structure-aware)
# ----------------------------
class IndexValueEncoder(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., 1] in [0,1]
        return self.net(x)


class SlotConstraintEncoder(nn.Module):
    def __init__(self, num_relations: int, max_index: int, emb_dim: int = 128, nhead: int = 4, nlayers: int = 2, max_vars: int = 64, max_dims: int = 6, dropout: float = 0.1):
        super().__init__()
        self.emb_dim = emb_dim
        self.max_vars = max_vars
        self.max_dims = max_dims
        self.max_index = max_index

        # Embeddings
        self.var_embed = nn.Embedding(max_vars, emb_dim)                 # per-episode variable IDs 0..(num_vars-1)
        self.dim_embed = nn.Embedding(max_dims, emb_dim)                 # dimension position 0..D-1
        # Continuous index value encoder (shared MLP), index normalized to [0,1]
        self.index_mlp = IndexValueEncoder(emb_dim)
        self.relation_embed = nn.Embedding(num_relations, emb_dim)       # relation types
        self.dim_token = nn.Embedding(max_dims, emb_dim)                 # per-dimension sequence token

        # Positional embeddings over slot sequence (relation + vars)
        self.pos_embed = nn.Embedding(512, emb_dim)
        # Relative position bias (uniform across heads) — keep initialized to zero
        self.max_rel_pos = 128
        self.rel_pos_bias = nn.Embedding(2 * self.max_rel_pos - 1, 1)
        with torch.no_grad():
            self.rel_pos_bias.weight.zero_()

        # No explicit pairwise channels; rely on attention to discover relations

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=nhead, dim_feedforward=256, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, nlayers)
        self.dropout = nn.Dropout(dropout)

    def build_slot_embeddings(self, var_ids: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """
        var_ids: [B, V] integers in [0, num_vars_in_episode)
        indices: [B, V, D] integers (0..max_index). 0 can be a valid index; use masks externally.
        returns: [B, V, D_model] slot embeddings
        """
        B, V, D = indices.shape
        V_emb = self.var_embed(var_ids.clamp(min=0, max=self.max_vars - 1))     # [B, V, E]
        # Build per-dimension embeddings and sum across dims
        dim_ids = torch.arange(D, device=indices.device).clamp(max=self.max_dims - 1)
        dim_emb = self.dim_embed(dim_ids)                                        # [D, E]
        dim_emb = dim_emb.unsqueeze(0).unsqueeze(0).expand(B, V, D, -1)          # [B, V, D, E]
        # Normalize indices to [0,1] and encode via MLP
        idx_norm = indices.clamp(min=0).float() / max(1.0, float(self.max_index))   # [B, V, D]
        idx_emb = self.index_mlp(idx_norm.unsqueeze(-1))                            # [B, V, D, E]
        per_var_emb = V_emb + (dim_emb + idx_emb).sum(dim=2)                     # [B, V, E]
        return per_var_emb

    def forward(self, var_ids: torch.Tensor, indices: torch.Tensor, relation_ids: torch.Tensor, slot_mask: torch.Tensor) -> torch.Tensor:
        """
        var_ids: [B, V]
        indices: [B, V, D]
        relation_ids: [B]
        slot_mask: [B, V] boolean or 0/1 (1 = real slot)
        returns: [B, E]
        """
        B, V, D = indices.shape
        # Build [REL] token
        rel_tok = self.relation_embed(relation_ids)              # [B, E]

        # Build per-dimension sub-slots: for each dim d, a dim token then V sub-slots for that dim
        var_emb = self.var_embed(var_ids.clamp(min=0, max=self.max_vars - 1))        # [B, V, E]
        dim_ids = torch.arange(D, device=indices.device).clamp(max=self.max_dims - 1)
        dim_pos_emb = self.dim_embed(dim_ids).unsqueeze(0).unsqueeze(0).expand(B, V, D, -1)  # [B, V, D, E]
        idx_norm = indices.clamp(min=0).float() / max(1.0, float(self.max_index))             # [B, V, D]
        idx_emb = self.index_mlp(idx_norm.unsqueeze(-1))                                      # [B, V, D, E]
        sub_slots = var_emb.unsqueeze(2) + dim_pos_emb + idx_emb                               # [B, V, D, E]

        seq_parts = [rel_tok.unsqueeze(1)]
        mask_parts = [torch.ones(B, 1, dtype=torch.bool, device=indices.device)]  # REL is real
        for d in range(D):
            # dimension token
            dim_tok = self.dim_token(torch.tensor(d, device=indices.device)).unsqueeze(0).expand(B, -1)  # [B, E]
            seq_parts.append(dim_tok.unsqueeze(1))
            mask_parts.append(torch.ones(B, 1, dtype=torch.bool, device=indices.device))
            # sub-slots for this dimension
            slots_d = sub_slots[:, :, d, :]  # [B, V, E]
            seq_parts.append(slots_d)
            mask_parts.append(slot_mask)     # [B, V]

        seq = torch.cat(seq_parts, dim=1)    # [B, L, E]
        real_mask = torch.cat(mask_parts, dim=1)  # [B, L] True=real

        # Positional encodings
        L = seq.size(1)
        pos = self.pos_embed(torch.arange(L, device=seq.device).clamp(max=self.pos_embed.num_embeddings - 1))
        seq = seq + pos.unsqueeze(0)

        # Padding mask: True = pad
        pad_mask = ~real_mask.bool()

        # No relative position bias (disabled)
        x = self.transformer(seq, src_key_padding_mask=pad_mask)

        # Pool: masked mean (including relation token)
        valid_lengths = (~pad_mask).sum(dim=1).clamp(min=1).unsqueeze(-1)  # [B,1]
        x_masked = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        pooled = x_masked.sum(dim=1) / valid_lengths
        pooled = self.dropout(pooled)
        pooled = F.normalize(pooled, p=2, dim=1)
        return pooled

# ----------------------------
# 3. Prototypical loss
# ----------------------------
def prototypical_loss(support_emb, support_labels, query_emb, query_labels, temperature: float = 0.1):
    # Ensure we have both positive and negative examples in support set
    unique_support_labels = torch.unique(support_labels)
    if len(unique_support_labels) < 2:
        raise ValueError(f"Support set must contain both positive and negative examples. Got labels: {unique_support_labels}")
    
    # Ensure all query labels exist in support set
    unique_query_labels = torch.unique(query_labels)
    missing_labels = set(unique_query_labels.tolist()) - set(unique_support_labels.tolist())
    if missing_labels:
        raise ValueError(f"Query set contains labels not in support set: {missing_labels}")
    
    # Create prototypes for each class
    # Build normalized prototypes (cosine similarity)
    prototypes = []
    for c in unique_support_labels:
        mask = support_labels == c
        if not mask.any():
            raise ValueError(f"No support examples for class {c}")
        proto = support_emb[mask].mean(0)
        prototypes.append(F.normalize(proto, p=2, dim=0))
    prototypes = torch.stack(prototypes)  # [C, D]

    # Cosine similarity classification
    query_norm = F.normalize(query_emb, p=2, dim=1)
    logits = temperature * (query_norm @ prototypes.t())  # [Nq, C]
    log_p_y = F.log_softmax(logits, dim=1)

    # Create label mapping ensuring all classes are included
    all_classes = torch.sort(unique_support_labels)[0]  # Sort to ensure consistent mapping
    label_map = {c.item(): i for i, c in enumerate(all_classes)}
    
    # Map query labels to indices
    y = torch.tensor([label_map[l.item()] for l in query_labels])
    
    # Calculate loss and accuracy
    loss = F.nll_loss(log_p_y, y)
    acc = (log_p_y.argmax(1) == y).float().mean().item()
    
    return loss, acc


# ----------------------------
# 4. Episodic Training Loop
# ----------------------------
def train_prototypical(encoder, optimizer, dataset, tokenizer, n_episodes=1000, k_support=5, k_query=15):
    """
    dataset: dict mapping problem_id -> list of (constraint, label)
    constraint: (relation:str, list of variables)
    """
    for episode in range(n_episodes):
        # 1. sample random problem
        problem_id = random.choice(list(dataset.keys()))
        examples = dataset[problem_id]

        # 2. create global variable mapping
        var_names = set()
        for relation, variables, label in examples:
            constr = (relation, variables)
            for var_name, _ in constr[1]:
                var_names.add(var_name)
        var2id = {v: i for i, v in enumerate(var_names)}

        # 3. split support / query
        pos = [ex for ex in examples if ex[2] == 1]
        neg = [ex for ex in examples if ex[2] == 0]
        print(f"Problem {problem_id} has {len(pos)} positive and {len(neg)} negative examples")
        
        # Ensure we have enough examples of each class
        min_required = min(k_support, k_query)
        if len(pos) < min_required or len(neg) < min_required:
            print(f"Skipping episode - not enough examples. Need {min_required} of each class. Pos: {len(pos)}, Neg: {len(neg)}")
            continue
            
        # Sample equal numbers of positive and negative examples
        n_support = min(k_support, min(len(pos), len(neg)))
        n_query = min(k_query, min(len(pos) - n_support, len(neg) - n_support))
        
        # First sample support set
        pos_support = random.sample(pos, n_support)
        neg_support = random.sample(neg, n_support)
        support = pos_support + neg_support
        
        # Remove support examples from available pools
        pos_remaining = [x for x in pos if x not in pos_support]
        neg_remaining = [x for x in neg if x not in neg_support]
        
        # Then sample query set from remaining examples
        pos_query = random.sample(pos_remaining, n_query)
        neg_query = random.sample(neg_remaining, n_query)
        query = pos_query + neg_query
        
        print(f"Selected {len(support)} support ({n_support} each class) and {len(query)} query ({n_query} each class) examples")
        
        print(f"Selected {len(support)} support and {len(query)} query examples")

        # 4. build slot tensors for the slot-based encoder
        def build_slot_batch(examples):
            # Determine maximum dims among variables in this episode
            max_dims = 0
            for relation, variables, label in examples:
                for _, idxs in variables:
                    max_dims = max(max_dims, len(idxs))

            var_ids_list, idx_list, rel_ids, labels = [], [], [], []
            rel_id_map = tokenizer.relation2id

            for relation, variables, label in examples:
                # map var names to episode IDs using var2id
                var_ids = [var2id[vn] for vn, _ in variables]
                # pad vars to same length per batch example
                V = len(var_ids)
                # build indices matrix [V, max_dims]
                idx_mat = []
                for _, idxs in variables:
                    row = list(idxs) + [0] * (max_dims - len(idxs))
                    idx_mat.append(row)
                var_ids_list.append(torch.tensor(var_ids, dtype=torch.long))
                idx_list.append(torch.tensor(idx_mat, dtype=torch.long))
                rel_ids.append(rel_id_map[relation])
                labels.append(label)

            # pad to max variables across examples
            max_vars = max(v.size(0) for v in var_ids_list) if var_ids_list else 0
            PAD_VAR = 0
            PAD_IDX = 0
            padded_vars = []
            padded_idxs = []
            slot_masks = []
            for v_ids, idxs in zip(var_ids_list, idx_list):
                pad_len = max_vars - v_ids.size(0)
                if pad_len > 0:
                    v_ids = torch.cat([v_ids, torch.full((pad_len,), PAD_VAR, dtype=torch.long)], dim=0)
                    pad_rows = torch.full((pad_len, idxs.size(1)), PAD_IDX, dtype=torch.long)
                    idxs = torch.cat([idxs, pad_rows], dim=0)
                padded_vars.append(v_ids)
                padded_idxs.append(idxs)
                slot_masks.append(torch.tensor([1]* (max_vars - pad_len) + [0]*pad_len, dtype=torch.bool))

            var_batch = torch.stack(padded_vars) if padded_vars else torch.empty(0, 0, dtype=torch.long)
            idx_batch = torch.stack(padded_idxs) if padded_idxs else torch.empty(0, 0, 0, dtype=torch.long)
            slot_mask = torch.stack(slot_masks) if slot_masks else torch.empty(0, 0, dtype=torch.bool)
            rel_batch = torch.tensor(rel_ids, dtype=torch.long)
            label_batch = torch.tensor(labels, dtype=torch.long)
            return var_batch, idx_batch, rel_batch, slot_mask, label_batch

        support_vars, support_idx, support_rel, support_mask, support_y = build_slot_batch(support)
        query_vars, query_idx, query_rel, query_mask, query_y = build_slot_batch(query)

        # 5. forward + loss
        encoder.train()
        sup_emb = encoder(support_vars, support_idx, support_rel, support_mask)
        qry_emb = encoder(query_vars, query_idx, query_rel, query_mask)
        loss, acc = prototypical_loss(sup_emb, support_y, qry_emb, query_y, temperature=1.0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if episode % 50 == 0:
            print(f"Episode {episode}, loss {loss.item():.4f}, acc {acc:.3f}")


# ----------------------------
# 4b. Evaluation Loop (no grad)
# ----------------------------
def evaluate_prototypical(encoder, dataset, tokenizer, n_episodes=200, k_support=5, k_query=15):
    encoder.eval()
    total_loss = 0.0
    total_acc = 0.0
    valid_episodes = 0

    for episode in range(n_episodes):
        # 1. sample random problem
        problem_id = random.choice(list(dataset.keys()))
        examples = dataset[problem_id]

        # 2. create global variable mapping
        var_names = set()
        for relation, variables, label in examples:
            constr = (relation, variables)
            for var_name, _ in constr[1]:
                var_names.add(var_name)
        var2id = {v: i for i, v in enumerate(var_names)}

        # 3. split support / query (balanced)
        pos = [ex for ex in examples if ex[2] == 1]
        neg = [ex for ex in examples if ex[2] == 0]

        min_required = min(k_support, k_query)
        if len(pos) < min_required or len(neg) < min_required:
            continue

        n_support = min(k_support, min(len(pos), len(neg)))
        n_query = min(k_query, min(len(pos) - n_support, len(neg) - n_support))
        if n_query <= 0:
            continue

        pos_support = random.sample(pos, n_support)
        neg_support = random.sample(neg, n_support)
        support = pos_support + neg_support

        pos_remaining = [x for x in pos if x not in pos_support]
        neg_remaining = [x for x in neg if x not in neg_support]

        pos_query = random.sample(pos_remaining, n_query)
        neg_query = random.sample(neg_remaining, n_query)
        query = pos_query + neg_query

        # 4. build slot tensors (same as training)
        def build_slot_batch(examples):
            max_dims = 0
            for relation, variables, label in examples:
                for _, idxs in variables:
                    max_dims = max(max_dims, len(idxs))

            var_ids_list, idx_list, rel_ids, labels = [], [], [], []
            rel_id_map = tokenizer.relation2id

            for relation, variables, label in examples:
                var_ids = [var2id[vn] for vn, _ in variables]
                V = len(var_ids)
                idx_mat = []
                for _, idxs in variables:
                    row = list(idxs) + [0] * (max_dims - len(idxs))
                    idx_mat.append(row)
                var_ids_list.append(torch.tensor(var_ids, dtype=torch.long))
                idx_list.append(torch.tensor(idx_mat, dtype=torch.long))
                rel_ids.append(rel_id_map[relation])
                labels.append(label)

            max_vars = max(v.size(0) for v in var_ids_list) if var_ids_list else 0
            PAD_VAR = 0
            PAD_IDX = 0
            padded_vars = []
            padded_idxs = []
            slot_masks = []
            for v_ids, idxs in zip(var_ids_list, idx_list):
                pad_len = max_vars - v_ids.size(0)
                if pad_len > 0:
                    v_ids = torch.cat([v_ids, torch.full((pad_len,), PAD_VAR, dtype=torch.long)], dim=0)
                    pad_rows = torch.full((pad_len, idxs.size(1)), PAD_IDX, dtype=torch.long)
                    idxs = torch.cat([idxs, pad_rows], dim=0)
                padded_vars.append(v_ids)
                padded_idxs.append(idxs)
                slot_masks.append(torch.tensor([1]* (max_vars - pad_len) + [0]*pad_len, dtype=torch.bool))

            var_batch = torch.stack(padded_vars) if padded_vars else torch.empty(0, 0, dtype=torch.long)
            idx_batch = torch.stack(padded_idxs) if padded_idxs else torch.empty(0, 0, 0, dtype=torch.long)
            slot_mask = torch.stack(slot_masks) if slot_masks else torch.empty(0, 0, dtype=torch.bool)
            rel_batch = torch.tensor(rel_ids, dtype=torch.long)
            label_batch = torch.tensor(labels, dtype=torch.long)
            return var_batch, idx_batch, rel_batch, slot_mask, label_batch

        with torch.no_grad():
            support_vars, support_idx, support_rel, support_mask, support_y = build_slot_batch(support)
            query_vars, query_idx, query_rel, query_mask, query_y = build_slot_batch(query)

            sup_emb = encoder(support_vars, support_idx, support_rel, support_mask)
            qry_emb = encoder(query_vars, query_idx, query_rel, query_mask)
            loss, acc = prototypical_loss(sup_emb, support_y, qry_emb, query_y, temperature=1.0)

        total_loss += loss.item()
        total_acc += acc
        valid_episodes += 1

    if valid_episodes == 0:
        print("No valid evaluation episodes could be formed.")
        return float('nan'), float('nan')

    return total_loss / valid_episodes, total_acc / valid_episodes

# ----------------------------
# 4c. Few-shot holdout evaluation on full problems (support_frac)
# ----------------------------
def evaluate_fewshot_holdout(
    encoder,
    dataset,
    tokenizer,
    support_frac: float = 0.05,
    min_support_per_class: int = 10,
    temperature: float = 1.0,
    refine_steps: int = 2,
):
    """
    For each problem: take ~support_frac of each class as labeled (support),
    classify the rest (query) using prototypes, and aggregate metrics.
    Returns a dict of aggregated metrics and per-problem breakdowns.
    """
    total_tp = total_fp = total_tn = total_fn = 0
    per_problem_metrics = []

    def build_slot_batch(examples, var2id):
        # Determine maximum dims among variables in this set
        max_dims = 0
        for relation, variables, label in examples:
            for _, idxs in variables:
                max_dims = max(max_dims, len(idxs))

        var_ids_list, idx_list, rel_ids, labels = [], [], [], []
        rel_id_map = tokenizer.relation2id

        for relation, variables, label in examples:
            var_ids = [var2id[vn] for vn, _ in variables]
            idx_mat = []
            for _, idxs in variables:
                row = list(idxs) + [0] * (max_dims - len(idxs))
                idx_mat.append(row)
            var_ids_list.append(torch.tensor(var_ids, dtype=torch.long))
            idx_list.append(torch.tensor(idx_mat, dtype=torch.long))
            rel_ids.append(rel_id_map[relation])
            labels.append(label)

        max_vars = max(v.size(0) for v in var_ids_list) if var_ids_list else 0
        PAD_VAR = 0
        PAD_IDX = 0
        padded_vars = []
        padded_idxs = []
        slot_masks = []
        for v_ids, idxs in zip(var_ids_list, idx_list):
            pad_len = max_vars - v_ids.size(0)
            if pad_len > 0:
                v_ids = torch.cat([v_ids, torch.full((pad_len,), PAD_VAR, dtype=torch.long)], dim=0)
                pad_rows = torch.full((pad_len, idxs.size(1)), PAD_IDX, dtype=torch.long)
                idxs = torch.cat([idxs, pad_rows], dim=0)
            padded_vars.append(v_ids)
            padded_idxs.append(idxs)
            slot_masks.append(torch.tensor([1]* (max_vars - pad_len) + [0]*pad_len, dtype=torch.bool))

        var_batch = torch.stack(padded_vars) if padded_vars else torch.empty(0, 0, dtype=torch.long)
        idx_batch = torch.stack(padded_idxs) if padded_idxs else torch.empty(0, 0, 0, dtype=torch.long)
        slot_mask = torch.stack(slot_masks) if slot_masks else torch.empty(0, 0, dtype=torch.bool)
        rel_batch = torch.tensor(rel_ids, dtype=torch.long)
        label_batch = torch.tensor(labels, dtype=torch.long)
        return var_batch, idx_batch, rel_batch, slot_mask, label_batch

    for problem_id, examples in dataset.items():
        # Build per-problem var2id
        var_names = set()
        for relation, variables, label in examples:
            for var_name, _ in variables:
                var_names.add(var_name)
        var2id = {v: i for i, v in enumerate(var_names)}

        # Stratified split by class
        pos = [ex for ex in examples if ex[2] == 1]
        neg = [ex for ex in examples if ex[2] == 0]
        if len(pos) == 0 or len(neg) == 0:
            continue
        n_pos_sup = max(min_support_per_class, int(round(len(pos) * support_frac)))
        n_neg_sup = max(min_support_per_class, int(round(len(neg) * support_frac)))
        n_pos_sup = min(n_pos_sup, len(pos))
        n_neg_sup = min(n_neg_sup, len(neg))
        # Ensure at least one query remains for each class if possible
        if len(pos) - n_pos_sup <= 0 or len(neg) - n_neg_sup <= 0:
            n_pos_sup = min(n_pos_sup, max(1, len(pos) - 1))
            n_neg_sup = min(n_neg_sup, max(1, len(neg) - 1))
            if len(pos) - n_pos_sup <= 0 or len(neg) - n_neg_sup <= 0:
                continue

        pos_support = random.sample(pos, n_pos_sup)
        neg_support = random.sample(neg, n_neg_sup)
        support = pos_support + neg_support
        pos_remaining = [x for x in pos if x not in pos_support]
        neg_remaining = [x for x in neg if x not in neg_support]
        query = pos_remaining + neg_remaining
        if len(query) == 0:
            continue

        with torch.no_grad():
            encoder.eval()
            sup_vars, sup_idx, sup_rel, sup_mask, sup_y = build_slot_batch(support, var2id)
            qry_vars, qry_idx, qry_rel, qry_mask, qry_y = build_slot_batch(query, var2id)
            sup_emb = encoder(sup_vars, sup_idx, sup_rel, sup_mask)
            qry_emb = encoder(qry_vars, qry_idx, qry_rel, qry_mask)

            # Build prototypes from support
            unique_support_labels = torch.unique(sup_y)
            if len(unique_support_labels) < 2:
                continue
            classes_sorted = torch.sort(unique_support_labels)[0]
            prototypes = []
            for c in classes_sorted:
                mask = sup_y == c
                proto = F.normalize(sup_emb[mask].mean(0), p=2, dim=0)
                prototypes.append(proto)
            prototypes = torch.stack(prototypes)  # [C, E]

            # Cosine logits
            qry_norm = F.normalize(qry_emb, p=2, dim=1)
            logits = temperature * (qry_norm @ prototypes.t())

            # Optional transductive refinement of prototypes using query soft assignments
            for _ in range(max(0, refine_steps)):
                probs = F.softmax(logits, dim=1)  # [Nq, C]
                # Update prototypes with support (hard) + query (soft) contributions
                new_protos = []
                for ci, c in enumerate(classes_sorted):
                    sup_mask = (sup_y == c)
                    sup_count = sup_mask.sum().item()
                    sup_sum = sup_emb[sup_mask].sum(dim=0) if sup_count > 0 else torch.zeros_like(prototypes[0])
                    qry_weights = probs[:, ci].unsqueeze(1)  # [Nq,1]
                    qry_sum = (qry_weights * qry_emb).sum(dim=0)
                    denom = sup_count + probs[:, ci].sum().clamp(min=1e-6)
                    proto = (sup_sum + qry_sum) / denom
                    new_protos.append(F.normalize(proto, p=2, dim=0))
                prototypes = torch.stack(new_protos)
                logits = temperature * (qry_norm @ prototypes.t())

            pred_idx = logits.argmax(dim=1)

            # Map back to original labels order
            classes = classes_sorted.tolist()
            idx_to_label = {i: c for i, c in enumerate(classes)}
            y_pred = torch.tensor([idx_to_label[i.item()] for i in pred_idx], dtype=torch.long)

        # Metrics per problem
        y_true = qry_y
        tp = int(((y_pred == 1) & (y_true == 1)).sum().item())
        tn = int(((y_pred == 0) & (y_true == 0)).sum().item())
        fp = int(((y_pred == 1) & (y_true == 0)).sum().item())
        fn = int(((y_pred == 0) & (y_true == 1)).sum().item())

        total_tp += tp; total_fp += fp; total_tn += tn; total_fn += fn

        per_problem_metrics.append({
            'problem': problem_id,
            'accuracy': (tp + tn) / max(1, tp + tn + fp + fn),
            'precision_pos': tp / max(1, tp + fp),
            'recall_pos': tp / max(1, tp + fn),
            'f1_pos': (2*tp) / max(1, 2*tp + fp + fn),
        })

    # Aggregate macro/weighted
    total = total_tp + total_tn + total_fp + total_fn
    accuracy = (total_tp + total_tn) / max(1, total)
    precision_pos = total_tp / max(1, total_tp + total_fp)
    recall_pos = total_tp / max(1, total_tp + total_fn)
    f1_pos = (2*total_tp) / max(1, 2*total_tp + total_fp + total_fn)

    precision_neg = total_tn / max(1, total_tn + total_fn)
    recall_neg = total_tn / max(1, total_tn + total_fp)
    f1_neg = (2*total_tn) / max(1, 2*total_tn + total_fp + total_fn)

    macro_f1 = 0.5 * (f1_pos + f1_neg)

    support_pos = total_tp + total_fn
    support_neg = total_tn + total_fp
    weighted_f1 = (
        (f1_pos * support_pos + f1_neg * support_neg) / max(1, support_pos + support_neg)
    )

    return {
        'accuracy': accuracy,
        'precision_pos': precision_pos,
        'recall_pos': recall_pos,
        'f1_pos': f1_pos,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'per_problem': per_problem_metrics,
    }

# ----------------------------
# 4d. Few-shot holdout with linear (ridge) head on embeddings
# ----------------------------
def evaluate_fewshot_holdout_linear(
    encoder,
    dataset,
    tokenizer,
    support_frac: float = 0.05,
    min_support_per_class: int = 10,
    ridge_lambda: float = 1.0,
):
    total_tp = total_fp = total_tn = total_fn = 0
    per_problem_metrics = []

    def build_slot_batch(examples, var2id):
        max_dims = 0
        for relation, variables, label in examples:
            for _, idxs in variables:
                max_dims = max(max_dims, len(idxs))
        var_ids_list, idx_list, rel_ids, labels = [], [], [], []
        rel_id_map = tokenizer.relation2id
        for relation, variables, label in examples:
            var_ids = [var2id[vn] for vn, _ in variables]
            idx_mat = []
            for _, idxs in variables:
                row = list(idxs) + [0] * (max_dims - len(idxs))
                idx_mat.append(row)
            var_ids_list.append(torch.tensor(var_ids, dtype=torch.long))
            idx_list.append(torch.tensor(idx_mat, dtype=torch.long))
            rel_ids.append(rel_id_map[relation])
            labels.append(label)
        max_vars = max(v.size(0) for v in var_ids_list) if var_ids_list else 0
        PAD_VAR = 0; PAD_IDX = 0
        padded_vars = []; padded_idxs = []; slot_masks = []
        for v_ids, idxs in zip(var_ids_list, idx_list):
            pad_len = max_vars - v_ids.size(0)
            if pad_len > 0:
                v_ids = torch.cat([v_ids, torch.full((pad_len,), PAD_VAR, dtype=torch.long)], dim=0)
                pad_rows = torch.full((pad_len, idxs.size(1)), PAD_IDX, dtype=torch.long)
                idxs = torch.cat([idxs, pad_rows], dim=0)
            padded_vars.append(v_ids)
            padded_idxs.append(idxs)
            slot_masks.append(torch.tensor([1]* (max_vars - pad_len) + [0]*pad_len, dtype=torch.bool))
        var_batch = torch.stack(padded_vars) if padded_vars else torch.empty(0, 0, dtype=torch.long)
        idx_batch = torch.stack(padded_idxs) if padded_idxs else torch.empty(0, 0, 0, dtype=torch.long)
        slot_mask = torch.stack(slot_masks) if slot_masks else torch.empty(0, 0, dtype=torch.bool)
        rel_batch = torch.tensor(rel_ids, dtype=torch.long)
        label_batch = torch.tensor(labels, dtype=torch.long)
        return var_batch, idx_batch, rel_batch, slot_mask, label_batch

    for problem_id, examples in dataset.items():
        var_names = set()
        for relation, variables, label in examples:
            for var_name, _ in variables:
                var_names.add(var_name)
        var2id = {v: i for i, v in enumerate(var_names)}

        pos = [ex for ex in examples if ex[2] == 1]
        neg = [ex for ex in examples if ex[2] == 0]
        if len(pos) == 0 or len(neg) == 0:
            continue
        n_pos_sup = max(min_support_per_class, int(round(len(pos) * support_frac)))
        n_neg_sup = max(min_support_per_class, int(round(len(neg) * support_frac)))
        n_pos_sup = min(n_pos_sup, len(pos))
        n_neg_sup = min(n_neg_sup, len(neg))
        if len(pos) - n_pos_sup <= 0 or len(neg) - n_neg_sup <= 0:
            n_pos_sup = min(n_pos_sup, max(1, len(pos) - 1))
            n_neg_sup = min(n_neg_sup, max(1, len(neg) - 1))
            if len(pos) - n_pos_sup <= 0 or len(neg) - n_neg_sup <= 0:
                continue

        pos_support = random.sample(pos, n_pos_sup)
        neg_support = random.sample(neg, n_neg_sup)
        support = pos_support + neg_support
        pos_remaining = [x for x in pos if x not in pos_support]
        neg_remaining = [x for x in neg if x not in neg_support]
        query = pos_remaining + neg_remaining
        if len(query) == 0:
            continue

        with torch.no_grad():
            encoder.eval()
            sup_vars, sup_idx, sup_rel, sup_mask, sup_y = build_slot_batch(support, var2id)
            qry_vars, qry_idx, qry_rel, qry_mask, qry_y = build_slot_batch(query, var2id)
            sup_emb = encoder(sup_vars, sup_idx, sup_rel, sup_mask)  # [Ns, D]
            qry_emb = encoder(qry_vars, qry_idx, qry_rel, qry_mask)  # [Nq, D]

        # Fit ridge-regularized linear classifier w,b on support
        Xs = sup_emb
        ys = sup_y.float()
        Ns, D = Xs.shape
        # Add bias term by augmenting X with ones
        ones = torch.ones(Ns, 1, dtype=Xs.dtype, device=Xs.device)
        Xs_aug = torch.cat([Xs, ones], dim=1)  # [Ns, D+1]
        I = torch.eye(D+1, dtype=Xs.dtype, device=Xs.device)
        I[-1, -1] = 0.0  # no regularization on bias
        XtX = Xs_aug.t().mm(Xs_aug)
        XtY = Xs_aug.t().mv(ys)
        w = torch.linalg.solve(XtX + ridge_lambda * I, XtY)  # [D+1]

        # Scores for support to tune threshold for F1
        s_sup = Xs_aug.mv(w)
        # Find threshold that maximizes F1 on support
        thresholds = torch.quantile(s_sup, torch.linspace(0.0, 1.0, steps=min(101, max(3, Ns))))
        best_f1 = -1.0
        best_t = 0.0
        for t in thresholds:
            pred = (s_sup >= t).long()
            tp = int(((pred == 1) & (sup_y == 1)).sum().item())
            fp = int(((pred == 1) & (sup_y == 0)).sum().item())
            fn = int(((pred == 0) & (sup_y == 1)).sum().item())
            f1 = (2*tp) / max(1, 2*tp + fp + fn)
            if f1 > best_f1:
                best_f1 = f1; best_t = float(t.item())

        # Evaluate on query with tuned threshold
        Nq = qry_emb.shape[0]
        ones_q = torch.ones(Nq, 1, dtype=qry_emb.dtype, device=qry_emb.device)
        Xq_aug = torch.cat([qry_emb, ones_q], dim=1)
        s_q = Xq_aug.mv(w)
        y_pred = (s_q >= best_t).long()

        # Metrics per problem
        y_true = qry_y
        tp = int(((y_pred == 1) & (y_true == 1)).sum().item())
        tn = int(((y_pred == 0) & (y_true == 0)).sum().item())
        fp = int(((y_pred == 1) & (y_true == 0)).sum().item())
        fn = int(((y_pred == 0) & (y_true == 1)).sum().item())

        total_tp += tp; total_fp += fp; total_tn += tn; total_fn += fn
        per_problem_metrics.append({
            'problem': problem_id,
            'accuracy': (tp + tn) / max(1, tp + tn + fp + fn),
            'precision_pos': tp / max(1, tp + fp),
            'recall_pos': tp / max(1, tp + fn),
            'f1_pos': (2*tp) / max(1, 2*tp + fp + fn),
        })

    total = total_tp + total_tn + total_fp + total_fn
    accuracy = (total_tp + total_tn) / max(1, total)
    precision_pos = total_tp / max(1, total_tp + total_fp)
    recall_pos = total_tp / max(1, total_tp + total_fn)
    f1_pos = (2*total_tp) / max(1, 2*total_tp + total_fp + total_fn)

    precision_neg = total_tn / max(1, total_tn + total_fn)
    recall_neg = total_tn / max(1, total_tn + total_fp)
    f1_neg = (2*total_tn) / max(1, 2*total_tn + total_fp + total_fn)
    macro_f1 = 0.5 * (f1_pos + f1_neg)

    support_pos = total_tp + total_fn
    support_neg = total_tn + total_fp
    weighted_f1 = (
        (f1_pos * support_pos + f1_neg * support_neg) / max(1, support_pos + support_neg)
    )

    return {
        'accuracy': accuracy,
        'precision_pos': precision_pos,
        'recall_pos': recall_pos,
        'f1_pos': f1_pos,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'per_problem': per_problem_metrics,
    }

# ----------------------------
# 5. Example usage
# ----------------------------
# Tokenizer will be built dynamically from the dataset below

# example dataset: problem_id -> list of (constraint, label)
# constraint: (relation, [(var_name, [idx1, idx2]), ...])

instance, oracle = construct_latin_squares(9)
instance2, oracle2 = construct_latin_squares(4)

print("Constructing bias...")
instance.construct_bias()
instance2.construct_bias()

print("Featurizing constraints...")
print("Creating dataset...")
dataset = {}

# First instance
constraints1 = []
for c in instance.bias:
    constraint = featurize_constraint(c)
    label = 1 if c in set(oracle.constraints) else 0
    # Create a tuple of (relation, variables, label) as expected by the training loop
    constraints1.append((constraint[0], constraint[1], label))

random.shuffle(constraints1)

if constraints1:
    dataset[instance.name] = constraints1
    print(f"Added {len(constraints1)} constraints for {instance.name}")
    print(f"Sample constraint: {constraints1[0]}")

# Second instance
constraints2 = []
for c in instance2.bias:
    constraint = featurize_constraint(c)
    label = 1 if c in set(oracle2.constraints) else 0
    # Create a tuple of (relation, variables, label) as expected by the training loop
    constraints2.append((constraint[0], constraint[1], label))

random.shuffle(constraints2)

if constraints2:
    dataset[instance2.name] = constraints2
    print(f"Added {len(constraints2)} constraints for {instance2.name}")
    print(f"Sample constraint: {constraints2[0]}")

if not dataset:
    raise ValueError("No valid constraints found in any instance")

#print("Dataset: ", dataset)
print("Building tokenizer from dataset...")
for k in list(dataset.keys()):
    random.shuffle(dataset[k])
all_relations = set()
all_vars = set()
max_index_seen = 0
for ex_list in dataset.values():
    for relation, var_list, label in ex_list:
        all_relations.add(relation)
        for var_name, indices in var_list:
            all_vars.add(var_name)
            if indices:
                max_index_seen = max(max_index_seen, max(indices))

tokenizer = ConstraintTokenizer(relations=sorted(all_relations), max_index=max_index_seen)

print(f"Relations: {sorted(all_relations)}")
print(f"Max index seen: {max_index_seen}")
print("Calculating vocab size...")
vocab_size = tokenizer.get_vocab_size(len(all_vars))
print("Num variables: ", len(all_vars))

print("encoder initializing...")
encoder = SlotConstraintEncoder(
    num_relations=len(tokenizer.relation2id),
    max_index=max_index_seen,
    emb_dim=256,
    nhead=4,
    nlayers=3,
    dropout=0.1
)
print("optimizer initializing...")
optimizer = torch.optim.AdamW(encoder.parameters(), lr=3e-4, weight_decay=1e-2)

print("training...")
# train
train_prototypical(encoder, optimizer, dataset, tokenizer, n_episodes=1200, k_support=3, k_query=3)

print("evaluating...")
eval_loss, eval_acc = evaluate_prototypical(encoder, dataset, tokenizer, n_episodes=400, k_support=3, k_query=3)
print(f"Eval — loss: {eval_loss:.4f}, acc: {eval_acc:.3f}")

print("few-shot holdout (5%)...")
hold = evaluate_fewshot_holdout(encoder, dataset, tokenizer, support_frac=0.05, min_support_per_class=1, temperature=10.0)
print(f"Holdout — acc: {hold['accuracy']:.3f}, f1_pos: {hold['f1_pos']:.3f}, macro_f1: {hold['macro_f1']:.3f}, weighted_f1: {hold['weighted_f1']:.3f}")

print("few-shot holdout (5%) with linear head...")
hold_lin = evaluate_fewshot_holdout_linear(encoder, dataset, tokenizer, support_frac=0.05, min_support_per_class=1, ridge_lambda=1.0)
print(f"HoldoutLinear — acc: {hold_lin['accuracy']:.3f}, f1_pos: {hold_lin['f1_pos']:.3f}, macro_f1: {hold_lin['macro_f1']:.3f}, weighted_f1: {hold_lin['weighted_f1']:.3f}")

print("few-shot holdout (5%) DecisionTree + FeaturesRelDim...")
def evaluate_dt_fewshot(dataset, support_frac=0.05, min_support_per_class=1, random_state=0):
    total_tp = total_fp = total_tn = total_fn = 0
    for problem_id, examples in dataset.items():
        # Reconstruct a temporary ProblemInstance-like interface via the actual instances used earlier
        # We rely on the oracle constraints membership created above.
        # For featurization we need the instance corresponding to the problem_id.
        if problem_id == instance.name:
            feat = FeaturesRelDim(); feat.instance = instance
        elif problem_id == instance2.name:
            feat = FeaturesRelDim(); feat.instance = instance2
        else:
            continue

        # Build constraint objects list back from examples: we don't store original c, so skip per-problem beyond these two
        # Instead, we can featurize using relation/variables alone is not supported; so restrict to the two problems above.
        pos = [ex for ex in examples if ex[2] == 1]
        neg = [ex for ex in examples if ex[2] == 0]
        if len(pos) == 0 or len(neg) == 0:
            continue
        n_pos_sup = max(min_support_per_class, int(round(len(pos) * support_frac)))
        n_neg_sup = max(min_support_per_class, int(round(len(neg) * support_frac)))
        n_pos_sup = min(n_pos_sup, len(pos)); n_neg_sup = min(n_neg_sup, len(neg))
        if len(pos) - n_pos_sup <= 0 or len(neg) - n_neg_sup <= 0:
            n_pos_sup = min(n_pos_sup, max(1, len(pos) - 1))
            n_neg_sup = min(n_neg_sup, max(1, len(neg) - 1))
            if len(pos) - n_pos_sup <= 0 or len(neg) - n_neg_sup <= 0:
                continue

        # Recover original constraints for featurization: we can map back via constructing bias order matching.
        # Since we built dataset from instance.bias in the same order, we can recreate constraint lists:
        if problem_id == instance.name:
            constraint_list = list(instance.bias)
        else:
            constraint_list = list(instance2.bias)

        # Build index mapping from (relation, variables) to constraint object
        keyed = {}
        for c in constraint_list:
            rel_name = c.name
            scope = get_scope(c)
            vars_dims = [(get_var_name(var), get_var_dims(var)) for var in scope]
            keyed[(rel_name, tuple((vn, tuple(d)) for vn, d in vars_dims))] = c

        def to_constraint(example):
            relation, var_list, label = example
            key = (relation, tuple((vn, tuple(dims)) for vn, dims in var_list))
            return keyed.get(key, None), label

        # Materialize support/query constraint objects
        support_ex = random.sample(pos, n_pos_sup) + random.sample(neg, n_neg_sup)
        query_ex = [e for e in examples if e not in support_ex]
        support_cs = [to_constraint(e) for e in support_ex]
        query_cs = [to_constraint(e) for e in query_ex]
        support_cs = [(c, y) for c, y in support_cs if c is not None]
        query_cs = [(c, y) for c, y in query_cs if c is not None]
        if len(query_cs) == 0 or len(support_cs) == 0:
            continue

        Xs = [feat.featurize_constraint(c) for c, _ in support_cs]
        ys = [y for _, y in support_cs]
        Xq = [feat.featurize_constraint(c) for c, _ in query_cs]
        yq = [y for _, y in query_cs]

        clf = DecisionTreeClassifier(random_state=random_state, class_weight='balanced', max_depth=None)
        clf.fit(Xs, ys)
        yp = clf.predict(Xq)

        import numpy as np
        yq = np.array(yq); yp = np.array(yp)
        tp = int(((yp == 1) & (yq == 1)).sum())
        tn = int(((yp == 0) & (yq == 0)).sum())
        fp = int(((yp == 1) & (yq == 0)).sum())
        fn = int(((yp == 0) & (yq == 1)).sum())
        total_tp += tp; total_fp += fp; total_tn += tn; total_fn += fn

    total = total_tp + total_tn + total_fp + total_fn
    acc = (total_tp + total_tn) / max(1, total)
    f1_pos = (2*total_tp) / max(1, 2*total_tp + total_fp + total_fn)
    precision_pos = total_tp / max(1, total_tp + total_fp)
    recall_pos = total_tp / max(1, total_tp + total_fn)
    return {'accuracy': acc, 'f1_pos': f1_pos, 'precision_pos': precision_pos, 'recall_pos': recall_pos}

dt_res = evaluate_dt_fewshot(dataset, support_frac=0.15, min_support_per_class=1)
print(f"DT(5%) — acc: {dt_res['accuracy']:.3f}, f1_pos: {dt_res['f1_pos']:.3f}, precision_pos: {dt_res['precision_pos']:.3f}, recall_pos: {dt_res['recall_pos']:.3f}")





