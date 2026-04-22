import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv

try:
    from .artifact_utils import get_artifact_paths
    from .artifact_runtime import load_or_build_graph_examples
except ImportError:
    from artifact_utils import get_artifact_paths
    from artifact_runtime import load_or_build_graph_examples


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class QueryAwareGraphSAGE(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.bn1   = nn.BatchNorm1d(hidden_dim)       # stabilizes training

        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.bn2   = nn.BatchNorm1d(hidden_dim)

        self.dropout = dropout
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        residual = x                                   # skip connection
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x + residual)                      # residual add after norm+relu
        x = F.dropout(x, p=self.dropout, training=self.training)

        return self.classifier(x).squeeze(-1)


def build_pyg_data_from_example(
    example,
    model,
    query_prefix="Represent this sentence for searching relevant passages: ",
):
    """
    Convert one question graph into a PyG Data object.

    Node features:
        [chunk_embedding ; query_embedding ; cosine_similarity]

    Labels:
        1 if supporting chunk, else 0
    """
    chunks = example["context_chunks"]
    chunk_embeddings = example["context_chunk_embeddings"]
    G = example["graph"]

    if len(chunks) == 0:
        return None

    query_text = query_prefix + example["question"]
    query_embedding = model.encode(
        query_text,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    sim_scores = np.dot(chunk_embeddings, query_embedding).astype(np.float32)
    query_repeated = np.tile(query_embedding, (len(chunks), 1))
    sim_scores = sim_scores.reshape(-1, 1)

    node_features = np.concatenate(
        [chunk_embeddings, query_repeated, sim_scores], axis=1
    ).astype(np.float32)

    labels = np.array(
        [1.0 if chunk.metadata.get("is_supporting", False) else 0.0 for chunk in chunks],
        dtype=np.float32,
    )

    edges = list(G.edges())
    if len(edges) == 0:
        edge_index = torch.arange(len(chunks), dtype=torch.long).unsqueeze(0).repeat(2, 1)
    else:
        edge_list = [[u, v] for u, v in edges] + [[v, u] for u, v in edges]
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    data = Data(
        x=torch.tensor(node_features, dtype=torch.float),
        edge_index=edge_index,
        y=torch.tensor(labels, dtype=torch.float),
    )
    data.question_id   = example["id"]
    data.question_type = example["type"]
    data.gold_titles   = example["supporting_facts"]["title"]
    data.chunk_titles  = [chunk.metadata["title"] for chunk in chunks]

    return data


def build_pyg_dataset(graph_examples, model):
    dataset = []
    for example in graph_examples:
        data = build_pyg_data_from_example(example, model)
        if data is not None:
            dataset.append(data)
    return dataset


def split_dataset(dataset, train_ratio=0.8, seed=42):
    # shuffle before splitting to prevent ordering bias
    rng = random.Random(seed)
    shuffled = dataset[:]
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * train_ratio)
    return shuffled[:split_idx], shuffled[split_idx:]


def compute_global_pos_weight(train_dataset: list) -> torch.Tensor:
    """
    Compute pos_weight once over the full training set — not per batch.
    Per-batch computation causes the loss scale to fluctuate, making
    training noisy and the gradient signal inconsistent.
    """
    total_pos = sum(data.y.sum().item() for data in train_dataset)
    total_neg = sum((data.y == 0).sum().item() for data in train_dataset)
    pos_weight = total_neg / (total_pos + 1e-8)
    return torch.tensor([max(pos_weight, 1.0)], dtype=torch.float)


def compute_metrics_from_logits(logits, labels, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()

    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    return {
        "accuracy":  (preds == labels).float().mean().item(),
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
    }


def train_one_epoch(model, loader, optimizer, pos_weight, device, max_grad_norm=1.0):
    model.train()
    total_loss = 0.0
    pw = pos_weight.to(device)

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        logits = model(batch.x, batch.edge_index)
        loss = F.binary_cross_entropy_with_logits(logits, batch.y, pos_weight=pw)
        loss.backward()

        # gradient clipping prevents exploding gradients in deep GNN stacks
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, pos_weight, device):
    model.eval()
    total_loss = 0.0
    all_logits, all_labels = [], []
    pw = pos_weight.to(device)

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index)
        loss = F.binary_cross_entropy_with_logits(logits, batch.y, pos_weight=pw)
        total_loss += loss.item()
        all_logits.append(logits.cpu())
        all_labels.append(batch.y.cpu())

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    return {
        "loss": total_loss / len(loader),
        **compute_metrics_from_logits(all_logits, all_labels),
    }


if __name__ == "__main__":
    set_seed(42)
    SPLIT = "train"
    MAX_SAMPLES = 10000
    CHUNK_SIZE = 300
    CHUNK_OVERLAP = 50
    MIN_TEXT_LENGTH = 20
    MODEL_NAME = "BAAI/bge-base-en-v1.5"
    BATCH_SIZE = 64
    SEMANTIC_K = 2
    SEMANTIC_MIN_SIM = 0.40
    KEYWORD_OVERLAP_THRESHOLD = 3
    paths = get_artifact_paths(
        split=SPLIT,
        max_samples=MAX_SAMPLES,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    graph_examples, embed_model = load_or_build_graph_examples(
        split=SPLIT,
        max_samples=MAX_SAMPLES,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        min_text_length=MIN_TEXT_LENGTH,
        model_name=MODEL_NAME,
        batch_size=BATCH_SIZE,
        semantic_k=SEMANTIC_K,
        semantic_min_sim=SEMANTIC_MIN_SIM,
        keyword_overlap_threshold=KEYWORD_OVERLAP_THRESHOLD,
    )

    pyg_dataset = build_pyg_dataset(graph_examples, embed_model)
    train_dataset, val_dataset = split_dataset(pyg_dataset, train_ratio=0.8)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=16, shuffle=False)

    # global pos_weight computed from training set only
    pos_weight = compute_global_pos_weight(train_dataset)
    print(f"Global pos_weight: {pos_weight.item():.4f}")

    input_dim = pyg_dataset[0].x.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = QueryAwareGraphSAGE(
        input_dim=input_dim,
        hidden_dim=256,
        dropout=0.2,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # verbose=True was removed in PyTorch 2.2 — log LR manually instead
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    epochs           = 20
    patience         = 4
    best_val_loss    = float("inf")
    best_epoch       = 0
    best_state_dict  = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        train_loss  = train_one_epoch(model, train_loader, optimizer, pos_weight, device)
        val_metrics = evaluate(model, val_loader, pos_weight, device)

        scheduler.step(val_metrics["loss"])

        # manual LR log — shows exactly when the scheduler reduces the rate
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val P: {val_metrics['precision']:.4f} | "
            f"Val R: {val_metrics['recall']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss    = val_metrics["loss"]
            best_epoch       = epoch
            best_state_dict  = copy.deepcopy(model.state_dict())
            torch.save(best_state_dict, paths["gnn_checkpoint"])
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch}.")
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    print(f"\nBest val loss: {best_val_loss:.4f} at epoch {best_epoch}")
    print(f"Saved best model -> {paths['gnn_checkpoint']}")
