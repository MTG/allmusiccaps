import math
import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
from torch.distributed.nn.functional import all_gather as ddp_all_gather


def gather_with_grad(x, cat: bool = True) -> torch.Tensor | list[torch.Tensor]:
    if not (dist.is_initialized() and dist.get_world_size() > 1):
        return [x] if not cat else x

    # tuple of tensors, grads preserved
    parts = ddp_all_gather(x)

    if cat:
        return torch.cat(parts, dim=0)
    else:
        return parts


class ChunkedDDPSigmoidLoss(nn.Module):
    """Bidirectional distributed sigmoid contrastive loss (SigLIP-style).

    Each GPU holds a local batch of (audio, text) pairs. Embeddings are gathered
    across all GPUs and the loss is computed for every (local, remote) shard pair
    in both directions (audio→text and text→audio).

    The per-pair loss is:
        L = -LogSigmoid(label * logit)
    where logit = z_0 · z_1ᵀ * exp(t_prime) + bias, and label is +1 for matching
    pairs (diagonal of the same-device block) and -1 for all non-matching pairs.

    Learnable parameters t_prime and bias are initialised as recommended in
    https://arxiv.org/pdf/2303.15343 (SigLIP): t_prime = log(10), bias = -10.

    Reference implementation:
        https://github.com/ahmdtaha/distributed_sigmoid_loss
    """

    def __init__(self) -> None:
        super().__init__()

        # Init as recommended in https://arxiv.org/pdf/2303.15343j
        self.t_prime = nn.Parameter(torch.tensor(math.log(10)))  # log 10
        self.bias = nn.Parameter(torch.tensor(-10.0))

        self.sigmoid_loss = nn.LogSigmoid()

    def _compute_device_loss(
        self, z_0: torch.Tensor, z_1: torch.Tensor, t: torch.Tensor, same_device: bool
    ) -> torch.Tensor:
        """Compute sigmoid loss for one (local, remote) shard pair.

        Args:
            z_0: L2-normalized local embeddings, shape (local_bs, dim).
            z_1: L2-normalized remote embeddings, shape (remote_bs, dim).
            t: scalar temperature exp(t_prime).
            same_device: True when z_1 comes from this rank; enables positive
                labels on the diagonal.

        Returns:
            Scalar sum of per-element losses for this shard pair.
        """
        logits = z_0 @ z_1.T * t + self.bias
        bs = z_0.shape[0]

        if same_device:
            # diagonal +1 (matching pairs), off-diagonal -1 (negatives)
            labels = 2 * torch.eye(bs) - torch.ones(bs)
        else:
            # all pairs are negatives on a different-device shard
            labels = -torch.ones_like(logits)

        labels = labels.to(logits.device)

        loss = -self.sigmoid_loss(labels * logits)
        return loss.sum()

    def forward(self, z_a: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        """Compute the bidirectional sigmoid loss over all GPUs.

        Args:
            z_a: Audio embeddings for the local batch, shape (local_bs, dim).
            z_t: Text embeddings for the local batch, shape (local_bs, dim).

        Returns:
            Scalar mean loss (summed over all shard pairs, divided by local_bs).
        """
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0

        # Normalize once; gathered tensors inherit the normalization.
        z_a = F.normalize(z_a, dim=1)
        z_t = F.normalize(z_t, dim=1)
        t = self.t_prime.exp()

        # NOTE: We don't need a full all_gather here.
        # Actually, the chunked computation is designed to allow for larger batches by
        # only gathering one shard at a time in the loop bellow.
        # Keepting it as is for simplicity, but it could be optimized if interested in
        # maximizing batch size in the future.
        z_t_all = gather_with_grad(z_t, cat=False)
        z_a_all = gather_with_grad(z_a, cat=False)

        bs = z_a.shape[0]

        total_loss = 0
        for i in range(world_size):
            # audio→text and text→audio for each remote shard
            loss_at = self._compute_device_loss(
                z_a, z_t_all[i], t, same_device=i == rank
            )
            loss_ta = self._compute_device_loss(
                z_t, z_a_all[i], t, same_device=i == rank
            )
            total_loss += (loss_at + loss_ta) / 2

        return total_loss / bs


class DDPSigmoidLoss(nn.Module):
    """Bidirectional distributed sigmoid contrastive loss (SigLIP-style).

    Each GPU holds a local batch of (audio, text) pairs. Embeddings are gathered
    across all GPUs and the loss is computed for every (local, remote) shard pair
    in both directions (audio→text and text→audio).

    The per-pair loss is:
        L = -LogSigmoid(label * logit)
    where logit = z_0 · z_1ᵀ * exp(t_prime) + bias, and label is +1 for matching
    pairs (diagonal of the same-device block) and -1 for all non-matching pairs.

    Learnable parameters t_prime and bias are initialised as recommended in
    https://arxiv.org/pdf/2303.15343 (SigLIP): t_prime = log(10), bias = -10.

    Reference implementation:
        https://github.com/ahmdtaha/distributed_sigmoid_loss
    """

    def __init__(self) -> None:
        super().__init__()

        # Init as recommended in https://arxiv.org/pdf/2303.15343j
        self.t_prime = nn.Parameter(torch.tensor(math.log(10)))  # log 10
        self.bias = nn.Parameter(torch.tensor(-10.0))

    def forward(self, z_a: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        """Compute the bidirectional sigmoid loss over all GPUs.

        Args:
            z_a: Audio embeddings for the local batch, shape (local_bs, dim).
            z_t: Text embeddings for the local batch, shape (local_bs, dim).

        Returns:
            Scalar mean loss (summed over all shard pairs, divided by local_bs).
        """
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0

        z_a = F.normalize(z_a, dim=1)
        z_t = F.normalize(z_t, dim=1)
        t = self.t_prime.exp()

        z_t_all = gather_with_grad(z_t, cat=True)  # (global_bs, dim)
        z_a_all = gather_with_grad(z_a, cat=True)  # (global_bs, dim)

        bs = z_a.shape[0]
        global_bs = z_a_all.shape[0]

        # Two matmuls total instead of 2*world_size
        logits_at = z_a @ z_t_all.T * t + self.bias  # (local_bs, global_bs)
        logits_ta = z_t @ z_a_all.T * t + self.bias

        # +1 only at this rank's diagonal block, -1 everywhere else
        labels = -torch.ones(bs, global_bs, device=z_a.device)
        labels[:, rank * bs : (rank + 1) * bs] = (
            2 * torch.eye(bs, device=z_a.device) - 1
        )

        loss = (
            -F.logsigmoid(labels * logits_at) - F.logsigmoid(labels * logits_ta)
        ).sum() / 2
        return loss / bs


class InfoNCELoss(nn.Module):
    """InfoNCE loss function.

    This function expect features of shape: (2 * batch_size, feat_dim):

    features = [
        F0_1,
        F0_2,
        F0_N,
        F1_1,
        F1_2,
        F1_N,
    ]
    """

    def __init__(self, temp: float) -> None:
        super().__init__()
        self.temp = temp

    def forward(self, z_a: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        features = torch.cat([z_a, z_t], dim=0)

        # our implemnentation only works for 2 views
        n_views = 2
        batch_size = features.shape[0] // n_views

        # create two stacked diagonal matrices
        labels = torch.cat(
            [torch.arange(batch_size) for _ in range(n_views)], dim=0
        ).to(self.device)

        # create a matrix of shape (2 * batch_size, 2 * batch_size) with a diagonal and two sub-diagonals
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

        # normalize features and compute similarity matrix
        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(
            similarity_matrix.shape[0], -1
        )

        # select the positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select the negatives
        negatives = similarity_matrix[~labels.bool()].view(
            similarity_matrix.shape[0], -1
        )

        # rearange similirities: 1st column are the positives, the rest are the negatives
        logits = torch.cat([positives, negatives], dim=1)

        # create labels as class indices: the target is always the first column (0)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        # normalize the logits by the temperature
        logits = logits / self.temp

        loss = F.cross_entropy(logits, labels)

        return loss


class MultimodalInfoNCELoss(nn.Module):
    """
    Bidirectional multimodal InfoNCE (CLIP-style), memory-efficient.

    Args:
        features_0: tensor (local_batch, dim) from modality 0 (e.g., images).
        features_1: tensor (local_batch, dim) from modality 1 (e.g., texts).

    Returns:
        loss: scalar tensor (average of both directions)
    """

    def __init__(self, temp: float) -> None:
        super().__init__()
        self.temp = temp

    def forward(
        self,
        features_0: torch.Tensor,
        features_1: torch.Tensor,
    ) -> torch.Tensor:
        # print("reached multimodal loss..")
        local_batch = features_0.shape[0]
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0

        # print("world size", world_size, "rank", rank)

        # ---- gather features across GPUs ----
        if world_size > 1:
            features_0 = gather_with_grad(features_0)
            features_1 = gather_with_grad(features_1)

        global_batch = features_0.shape[0]

        # normalize features
        features_0 = F.normalize(features_0, dim=1)
        features_1 = F.normalize(features_1, dim=1)

        # print("done normalizing..")

        # get local chunk for this rank
        if world_size > 1:
            start = rank * local_batch
            end = (rank + 1) * local_batch
            local_feat0 = features_0[start:end]
            local_feat1 = features_1[start:end]
        else:
            local_feat0 = features_0
            local_feat1 = features_1

        # ---- similarities ----
        # local modality 0 vs all modality 1
        sim_0_to_1 = torch.matmul(local_feat0, features_1.T) / self.temp
        # print("done sim 0 to 1..")

        # local modality 1 vs all modality 0
        sim_1_to_0 = torch.matmul(local_feat1, features_0.T) / self.temp
        # print("done sim 1 to 0..")

        # targets: ground-truth alignment is "diagonal" across gathered batches
        targets = torch.arange(global_batch, device=features_0.device)

        # for local chunk, offset the target indices accordingly
        local_targets = targets[rank * local_batch : (rank + 1) * local_batch]

        # compute losses
        loss_0_to_1 = F.cross_entropy(sim_0_to_1, local_targets)
        loss_1_to_0 = F.cross_entropy(sim_1_to_0, local_targets)

        # final loss
        loss = (loss_0_to_1 + loss_1_to_0) / 2

        # Only return the total loss for consistency with other loss functions
        # return loss, loss_0_to_1, loss_1_to_0
        return loss


class MultiPositiveInfoNCELoss(nn.Module):
    """Multi-positive InfoNCE loss for multi-view embeddings.

    Expects a stacked tensor z of shape (V, B, D) where V is the number of
    views and B is the batch size.  For each anchor, all other views of the
    same sample are treated as positives and every other sample (across all
    views) as negatives.

    Loss per anchor:
        L_i = -log( sum(exp(s_pos)) / sum(exp(s_all_non_self)) )
            = -logsumexp(s_pos) + logsumexp(s_all_non_self)

    When running in DDP, embeddings are gathered across GPUs so that the
    negative pool spans the full global batch.
    """

    def __init__(self, temp: float = 0.1) -> None:
        super().__init__()
        self.temp = temp

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (V, B, D) stacked view embeddings.

        Returns:
            Scalar loss averaged over all anchors.
        """
        V, B, D = z.shape

        # (V*B, D) — local layout: [v0_b0..v0_b(B-1), v1_b0..v1_b(B-1), ...]
        z_flat = z.reshape(V * B, D)

        # DDP: single gather for efficiency
        z_flat = gather_with_grad(z_flat, cat=True)  # (V*B_global, D)
        N = z_flat.shape[0]

        # After gathering, the layout is:
        #   [rank0: v0_b0..v0_b(B-1), v1_b0..v1_b(B-1), ...
        #    rank1: v0_b0..v0_b(B-1), v1_b0..v1_b(B-1), ...]
        # i.e. each rank contributes a V*B chunk. Within each chunk position p,
        # the local sample index is p % B and the rank offset is (k // (V*B)) * B.
        # So the global sample id = (k // (V * B)) * B + (k % B).
        sample_ids = (torch.arange(N, device=z.device) // (V * B)) * B + (
            torch.arange(N, device=z.device) % B
        )

        # L2-normalise
        z_flat = F.normalize(z_flat, dim=-1)

        # Similarity matrix  (N, N)
        sim = z_flat @ z_flat.T / self.temp

        # Build masks
        # positive_mask[i, j] = True when i and j correspond to the same
        # sample but different views.
        positive_mask = sample_ids.unsqueeze(0) == sample_ids.unsqueeze(1)  # (N, N)

        # Remove self-similarities from positives
        self_mask = torch.eye(N, dtype=torch.bool, device=z.device)
        positive_mask = positive_mask & ~self_mask

        # For the denominator we want all non-self entries
        # Set self-similarities to -inf so they don't contribute
        sim_denom = sim.masked_fill(self_mask, float("-inf"))

        # logsumexp over all non-self entries (denominator)
        log_denom = torch.logsumexp(sim_denom, dim=1)  # (N,)

        # For the numerator: logsumexp over positives only
        # Set non-positive entries to -inf
        sim_numer = sim.masked_fill(~positive_mask, float("-inf"))
        log_numer = torch.logsumexp(sim_numer, dim=1)  # (N,)

        loss = (log_denom - log_numer).mean()
        return loss


class MultiPositiveSigmoidLoss(nn.Module):
    """Multi-positive sigmoid contrastive loss for multi-view embeddings.

    Mirrors :class:`MultiPositiveInfoNCELoss` but uses SigLIP-style per-pair
    sigmoid loss instead of softmax. Expects a stacked tensor ``z`` of shape
    ``(V, B, D)``. For each anchor, all other views of the same sample are
    labeled +1 (positives) and every view of every other sample is labeled -1
    (negatives). In DDP, embeddings are gathered across GPUs so the negative
    pool spans the full global batch.

    Uses learnable temperature ``t_prime`` and bias following SigLIP
    (https://arxiv.org/abs/2303.15343).
    """

    def __init__(self) -> None:
        super().__init__()
        self.t_prime = nn.Parameter(torch.tensor(math.log(10)))
        self.bias = nn.Parameter(torch.tensor(-10.0))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        V, B, D = z.shape

        z_flat = z.reshape(V * B, D)
        z_flat = gather_with_grad(z_flat, cat=True)  # (V*B_global, D)
        N = z_flat.shape[0]

        sample_ids = (torch.arange(N, device=z.device) // (V * B)) * B + (
            torch.arange(N, device=z.device) % B
        )

        z_flat = F.normalize(z_flat, dim=-1)

        t = self.t_prime.exp()
        logits = z_flat @ z_flat.T * t + self.bias  # (N, N)

        # +1 where same sample & different view; -1 elsewhere.
        same_sample = sample_ids.unsqueeze(0) == sample_ids.unsqueeze(1)
        self_mask = torch.eye(N, dtype=torch.bool, device=z.device)
        labels = -torch.ones_like(logits)
        labels[same_sample & ~self_mask] = 1.0

        # Exclude self-pairs from the loss.
        per_pair = -F.logsigmoid(labels * logits)
        per_pair = per_pair.masked_fill(self_mask, 0.0)

        # Normalise by the number of non-self pairs per anchor.
        return per_pair.sum() / (N * (N - 1))
