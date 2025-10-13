# spherical_pose_net.py
"""End-to-end spherical panorama pose estimation network.

This module copies the provided ``SphericalTransformer`` backbone and extends it
with a hierarchical soft-matching stage and a differentiable rotation solver.
All components are compatible with CPU/GPU execution and operate on batched
inputs.
"""

# spherical_transformer.py (backbone to keep)
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_lonlat_grid(H, W, device):
    u = torch.linspace(0, W - 1, W, device=device)
    v = torch.linspace(0, H - 1, H, device=device)
    uu, vv = torch.meshgrid(u, v, indexing='xy')  # [W,H]
    uu = uu.t().contiguous()  # [H,W]
    vv = vv.t().contiguous()  # [H,W]
    phi = (uu + 0.5) / W * 2 * math.pi - math.pi
    theta = (vv + 0.5) / H * math.pi - math.pi / 2
    return phi, theta  # [H,W], [H,W]


def bearings_from_lonlat(phi, theta):
    cos_t = torch.cos(theta)
    x = cos_t * torch.cos(phi)
    y = cos_t * torch.sin(phi)
    z = torch.sin(theta)
    p = torch.stack([x, y, z], dim=-1)  # [H,W,3]
    p = F.normalize(p, dim=-1)
    return p


def lat_attention_bias(theta, eps=1e-6):
    w = torch.log(torch.clamp(torch.abs(torch.sin(theta)), min=eps))  # [H,W]
    return w


class SphericalPositionalEncoding(nn.Module):
    def __init__(self, num_freqs=8, include_xyz=True):
        super().__init__()
        self.num_freqs = num_freqs
        self.include_xyz = include_xyz

    def forward(self, phi, theta):
        feats = []
        for k in range(self.num_freqs):
            f = 2.0 ** k
            feats += [torch.sin(f * phi), torch.cos(f * phi),
                      torch.sin(f * theta), torch.cos(f * theta)]
        pos = torch.stack(feats, dim=-1)  # [H,W,4*num_freqs]
        if self.include_xyz:
            xyz = bearings_from_lonlat(phi, theta)  # [H,W,3]
            pos = torch.cat([pos, xyz], dim=-1)     # [H,W,4F+3]
        return pos  # [H,W,Cpos]


class PatchEmbed(nn.Module):
    def __init__(self, in_ch=3, embed_dim=256, patch_size=8):
        super().__init__()
        ps = patch_size
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, embed_dim//2, kernel_size=ps, stride=ps, padding=0, bias=False),
            nn.BatchNorm2d(embed_dim//2),
            nn.GELU(),
            nn.Conv2d(embed_dim//2, embed_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

    def forward(self, x):
        x = self.proj(x)  # [B,C,H',W']
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)  # [B,HW,C]
        return x, x_flat, (H, W)


class SphericalMHA(nn.Module):
    def __init__(self, dim, num_heads=8, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, lat_bias=None, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # [B,N,H,D]
        q = q.transpose(1, 2)  # [B,H,N,D]
        k = k.transpose(1, 2)  # [B,H,N,D]
        v = v.transpose(1, 2)  # [B,H,N,D]
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B,H,N,N]
        if lat_bias is not None:
            lb = lat_bias.unsqueeze(1).unsqueeze(1)  # [B,1,1,N]
            attn = attn + lb
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = attn @ v  # [B,H,N,D]
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SphericalMHA(dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, dim),
            nn.Dropout(drop),
        )

    def forward(self, x, lat_bias=None, mask=None):
        x = x + self.attn(self.norm1(x), lat_bias=lat_bias, mask=mask)
        x = x + self.mlp(self.norm2(x))
        return x


class SphericalTransformer(nn.Module):
    def __init__(
        self,
        in_ch=3,
        embed_dim=256,
        depth=8,
        num_heads=8,
        mlp_ratio=4.0,
        patch_size=8,
        pos_freqs=8,
        include_xyz=True,
        drop=0.0,
        attn_drop=0.0,
    ):
        super().__init__()
        self.patch = PatchEmbed(in_ch=in_ch, embed_dim=embed_dim, patch_size=patch_size)
        self.posenc = SphericalPositionalEncoding(num_freqs=pos_freqs, include_xyz=include_xyz)
        pos_ch = 4 * pos_freqs + (3 if include_xyz else 0)
        self.pos_proj = nn.Linear(pos_ch, embed_dim)
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, drop, attn_drop)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Identity()

    @staticmethod
    def _downsample_lonlat(phi, theta, H_out, W_out):
        H, W = phi.shape
        phi = phi.unsqueeze(0).unsqueeze(0)
        theta = theta.unsqueeze(0).unsqueeze(0)
        phi_ds = F.interpolate(phi, size=(H_out, W_out), mode='bilinear', align_corners=False)[0,0]
        theta_ds = F.interpolate(theta, size=(H_out, W_out), mode='bilinear', align_corners=False)[0,0]
        return phi_ds, theta_ds

    def forward(self, x):
        B, _, H, W = x.shape
        device = x.device
        fmap, tokens, (Hp, Wp) = self.patch(x)  # fmap [B,C,Hp,Wp]; tokens [B,N,C]
        phi, theta = build_lonlat_grid(H, W, device)
        phi_ds, theta_ds = self._downsample_lonlat(phi, theta, Hp, Wp)  # [Hp,Wp]
        pos = self.posenc(phi_ds, theta_ds).reshape(Hp*Wp, -1)  # [N,Cpos]
        pos = self.pos_proj(pos).unsqueeze(0).expand(B, -1, -1)  # [B,N,C]
        tokens = tokens + pos
        lat_bias_map = lat_attention_bias(theta_ds)  # [Hp,Wp]
        lat_bias_tokens = lat_bias_map.flatten().unsqueeze(0).expand(B, -1)  # [B,N]
        for blk in self.blocks:
            tokens = blk(tokens, lat_bias=lat_bias_tokens)
        tokens = self.norm(tokens)  # [B,N,C]
        out = tokens.transpose(1, 2).reshape(B, -1, Hp, Wp)  # [B,C,Hp,Wp]
        out = self.head(out)
        return out, tokens, (Hp, Wp), (phi_ds, theta_ds), lat_bias_tokens


# -----------------------------------------------------------------------------
# Utility helpers for spherical pose estimation
# -----------------------------------------------------------------------------

def downsampled_bearings(phi_ds: torch.Tensor, theta_ds: torch.Tensor) -> torch.Tensor:
    """Convert downsampled longitude/latitude grids to unit bearing vectors.

    Args:
        phi_ds: Longitude angles of shape ``[Hp, Wp]``.
        theta_ds: Latitude angles of shape ``[Hp, Wp]``.

    Returns:
        Tensor of shape ``[Hp * Wp, 3]`` containing unit direction vectors.
    """
    bearings = bearings_from_lonlat(phi_ds, theta_ds)  # [Hp,Wp,3]
    bearings = bearings.view(-1, 3)
    return bearings


def chunked_topk_similarities(q: torch.Tensor, k: torch.Tensor, topk: int, chunk_size: int = 1024):
    """Compute top-k similarities without materializing the full N x N matrix.

    Args:
        q: Query tensor of shape ``[B, N, C]``.
        k: Key tensor of shape ``[B, N, C]``.
        topk: Number of top elements to retain per query token.
        chunk_size: Number of query tokens processed at once.

    Returns:
        A tuple ``(topk_scores, topk_indices)`` each of shape ``[B, N, topk]``.
    """
    B, N, C = q.shape
    scores_chunks = []
    indices_chunks = []
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        q_chunk = q[:, start:end, :]  # [B,chunk,C]
        sim = torch.matmul(q_chunk, k.transpose(1, 2))  # [B,chunk,N]
        topk_val, topk_idx = torch.topk(sim, k=topk, dim=-1)
        scores_chunks.append(topk_val)
        indices_chunks.append(topk_idx)
    topk_scores = torch.cat(scores_chunks, dim=1)
    topk_indices = torch.cat(indices_chunks, dim=1)
    return topk_scores, topk_indices


class HierarchicalSoftMatcher(nn.Module):
    """Coarse-to-fine soft matcher for spherical tokens with locality constraints."""

    def __init__(
        self,
        dim: int,
        topk_coarse: int = 128,
        topk_fine: int = 16,
        local_deg: float = 15.0,
        use_sinkhorn: bool = False,
        temperature: float = 0.07,
        sinkhorn_iters: int = 3,
    ) -> None:
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.topk_coarse = topk_coarse
        self.topk_fine = topk_fine
        self.local_rad = math.radians(local_deg)
        self.use_sinkhorn = use_sinkhorn
        self.temperature = temperature
        self.sinkhorn_iters = sinkhorn_iters

    def forward(
        self,
        feat_a: torch.Tensor,
        feat_b: torch.Tensor,
        dirs_a: torch.Tensor,
        dirs_b: torch.Tensor,
    ) -> torch.Tensor:
        """Compute sparse soft correspondences between two token sets.

        Args:
            feat_a: Query token features ``[B, N, C]``.
            feat_b: Key token features ``[B, N, C]``.
            dirs_a: Unit bearings for tokens in A ``[B, N, 3]``.
            dirs_b: Unit bearings for tokens in B ``[B, N, 3]``.

        Returns:
            Dense tensor ``P`` of shape ``[B, N, N]`` with sparse support
            determined by coarse-to-fine matching. Probabilities are
            row-normalized.
        """
        B, N, C = feat_a.shape
        device = feat_a.device
        q = self.q_proj(feat_a)
        k = self.k_proj(feat_b)

        _, coarse_indices = chunked_topk_similarities(q, k, self.topk_coarse)
        topk_total = coarse_indices.shape[-1]

        P = torch.zeros(B, N, N, device=device, dtype=feat_a.dtype)
        chunk_size = 1024
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            length = end - start
            q_chunk = q[:, start:end, :]  # [B,length,C]
            idx_chunk = coarse_indices[:, start:end, :]  # [B,length,topk]

            # Gather candidate key features and directions
            k_cand = torch.gather(
                k.unsqueeze(1).expand(-1, length, -1, -1),
                2,
                idx_chunk.unsqueeze(-1).expand(-1, -1, -1, C),
            )  # [B,length,topk,C]
            dir_a_chunk = dirs_a[:, start:end, :].unsqueeze(2)  # [B,length,1,3]
            dirs_b_cand = torch.gather(
                dirs_b.unsqueeze(1).expand(-1, length, -1, -1),
                2,
                idx_chunk.unsqueeze(-1).expand(-1, -1, -1, 3),
            )  # [B,length,topk,3]

            cosang = (dir_a_chunk * dirs_b_cand).sum(-1).clamp(-1.0, 1.0)
            ang = torch.acos(cosang)
            local_mask = ang <= self.local_rad
            if not torch.any(local_mask):
                local_mask = torch.ones_like(local_mask, dtype=torch.bool)
            valid_mask = local_mask.any(-1, keepdim=True)
            effective_mask = torch.where(valid_mask, local_mask, torch.ones_like(local_mask, dtype=torch.bool))

            logits_full = (q_chunk.unsqueeze(2) * k_cand).sum(-1) / self.temperature
            logits_full = logits_full.masked_fill(~effective_mask, -1e9)

            if self.topk_fine < topk_total:
                topk_keep = self.topk_fine
                topk_vals, topk_pos = torch.topk(logits_full, k=topk_keep, dim=-1)
                idx_chunk = torch.gather(idx_chunk, -1, topk_pos)
                effective_mask = torch.gather(effective_mask, -1, topk_pos)
                logits = topk_vals
            else:
                logits = logits_full

            fallback_mask = effective_mask.any(-1, keepdim=True)
            effective_mask = torch.where(fallback_mask, effective_mask, torch.ones_like(effective_mask, dtype=torch.bool))
            logits = logits.masked_fill(~effective_mask, -1e9)
            probs = torch.softmax(logits, dim=-1)

            if self.use_sinkhorn:
                logp = torch.log(probs + 1e-8)
                for _ in range(self.sinkhorn_iters):
                    logp = logp - torch.logsumexp(logp, dim=-1, keepdim=True)
                    logp = logp - torch.logsumexp(logp, dim=-2, keepdim=True)
                probs = torch.exp(logp)

            # Zero-out entries outside the effective mask to avoid scattering garbage
            probs = probs * effective_mask.float()
            P[:, start:end, :].scatter_(2, idx_chunk, probs)

        P = P / (P.sum(dim=-1, keepdim=True) + 1e-8)
        return P


class SoftRotationSolver(nn.Module):
    """Differentiable soft Procrustes solver producing rotations in ``SO(3)``."""

    def forward(self, P: torch.Tensor, dirs_a: torch.Tensor, dirs_b: torch.Tensor) -> torch.Tensor:
        """Solve for rotations aligning ``dirs_a`` to ``dirs_b`` under weights ``P``.

        Args:
            P: Matching probabilities ``[B, N, N]`` (row-normalized).
            dirs_a: Unit bearings for view A ``[B, N, 3]``.
            dirs_b: Unit bearings for view B ``[B, N, 3]``.

        Returns:
            Rotation matrices ``[B, 3, 3]`` in ``SO(3)``.
        """
        B, N, _ = dirs_a.shape
        weight = P / (P.sum(dim=-1, keepdim=True) + 1e-8)
        # Compute weighted cross-covariance H = Y^T (W X)
        WX = torch.matmul(weight, dirs_b)  # [B, N, 3]
        H = torch.matmul(dirs_a.transpose(1, 2), WX)  # [B,3,3]
        U, S, Vh = torch.linalg.svd(H)
        R = torch.matmul(Vh, U.transpose(-2, -1))
        det = torch.det(R)
        mask = det < 0
        if mask.any():
            Vh[mask, :, -1] *= -1
            R = torch.matmul(Vh, U.transpose(-2, -1))
        return R


class SphericalPoseNet(nn.Module):
    """Full pipeline combining backbone, matcher, and rotation solver."""

    def __init__(
        self,
        backbone: SphericalTransformer,
        matcher: HierarchicalSoftMatcher,
        solver: SoftRotationSolver,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.matcher = matcher
        self.solver = solver

    @classmethod
    def build_default(
        cls,
        embed_dim: int = 256,
        depth: int = 4,
        num_heads: int = 8,
        patch_size: int = 8,
        matcher_topk_coarse: int = 128,
        matcher_topk_fine: int = 16,
        matcher_local_deg: float = 15.0,
        temperature: float = 0.07,
        use_sinkhorn: bool = False,
    ) -> "SphericalPoseNet":
        backbone = SphericalTransformer(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            patch_size=patch_size,
        )
        matcher = HierarchicalSoftMatcher(
            dim=embed_dim,
            topk_coarse=matcher_topk_coarse,
            topk_fine=matcher_topk_fine,
            local_deg=matcher_local_deg,
            temperature=temperature,
            use_sinkhorn=use_sinkhorn,
        )
        solver = SoftRotationSolver()
        return cls(backbone, matcher, solver)

    def forward(self, img_a: torch.Tensor, img_b: torch.Tensor):
        """Estimate the relative rotation between two panoramas.

        Args:
            img_a: Tensor ``[B, 3, H, W]``.
            img_b: Tensor ``[B, 3, H, W]``.

        Returns:
            Tuple ``(R_pred, P, extras)`` where ``R_pred`` is ``[B, 3, 3]``,
            ``P`` is the dense soft correspondence matrix ``[B, N, N]`` and
            ``extras`` contains intermediate values including token directions.
        """
        _, tokens_a, (Hp, Wp), (phi_ds, theta_ds), _ = self.backbone(img_a)
        _, tokens_b, _, _, _ = self.backbone(img_b)
        B = img_a.shape[0]
        bearings = downsampled_bearings(phi_ds, theta_ds)  # [N,3]
        dirs = bearings.unsqueeze(0).expand(B, -1, -1)
        P = self.matcher(tokens_a, tokens_b, dirs, dirs)
        R_pred = self.solver(P, dirs, dirs)

        rot_dirs = torch.matmul(dirs, R_pred.transpose(1, 2))  # [B,N,3]
        term1 = (rot_dirs ** 2).sum(-1, keepdim=True)
        term2 = (dirs ** 2).sum(-1).unsqueeze(1)
        cross = torch.matmul(rot_dirs, dirs.transpose(1, 2))
        loss_matrix = term1 + term2 - 2.0 * cross
        loss = (loss_matrix * P).sum(dim=(-2, -1)) / (P.sum(dim=(-2, -1)) + 1e-8)
        loss = loss.mean()

        extras = {
            "Hp": Hp,
            "Wp": Wp,
            "phi_ds": phi_ds,
            "theta_ds": theta_ds,
            "dirs": dirs,
            "loss": loss.detach(),
        }
        return R_pred, P, {"loss": loss, **extras}


if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SphericalPoseNet.build_default(embed_dim=64, depth=2, num_heads=4, patch_size=8)
    model = model.to(device)
    B, H, W = 2, 128, 256
    img_a = torch.randn(B, 3, H, W, device=device)
    img_b = torch.randn(B, 3, H, W, device=device)
    R_pred, P, extras = model(img_a, img_b)
    print("Rotation matrix shape:", R_pred.shape)
    print("Soft correspondence shape:", P.shape)
    print("Loss:", extras["loss"].item())
