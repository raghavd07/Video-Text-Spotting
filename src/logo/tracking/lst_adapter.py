"""LST-Matcher adapter: GoMatching's long/short-term association transformers on LOGO arrays.

GoMatching's LSTMatcher (third_party/GoMatching/gomatching/modeling/roi_heads/lstmatcher.py) needs
detectron2/adet Instances. This module re-hosts its association path on plain tensors:
  * long_term_matcher / short_term_matcher are GoMatching's own Transformer class (roi_heads/transformer.py,
    pure torch, loaded by file path; third_party is not edited);
  * asso_head / asso_predictor / local_asso_predictor keep GoMatching's parameter names and maths, but
    asso_head takes the VPMM vector [N, 2d] instead of DeepSolo query features [N, 25, 256];
  * training targets come from per-detection GT track ids (0 = no GT, see label_detections) instead of
    IoU against GT boxes inside the matcher;
  * associate() reproduces GoMatching.batch_inference / run_short_term_match / run_long_term_match.
Dropped: rescoring head (our fusion replaces it), box/temporal position embedding (NO_POS_EMB: True in
every GoMatching config), detection NMS (upstream), MIN_TRACK_LEN filter (voting.min_len).

Frame dict (train_step clip items and associate input):
  quads [n,8]|[n,4,2] px, feats Tensor [n,2d] (VPMM output), scores [n] (score_fused), img_wh (w,h),
  track_ids [n] int (training only; 0 = no GT).
Config (tracker{}; missing keys use LST_DEFAULTS = GoMatching config.py + GoMatching_ICDAR15.yaml);
overlap_thr is GoMatching's VIDEO_TEST.OVERLAP_THRESH. tracker.thr is the input score threshold applied by
logo.tracking.run_track before associate(), not an LST setting.
"""
from __future__ import annotations

import importlib.util
import sys
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn

from logo.tracking.simple_tracker import match, quad_iou_matrix
from logo.utils.paths import project_root

LST_DEFAULTS = {
    # architecture (MODEL.ASSO_HEAD.*)
    "fc_dim": 1024, "num_fc": 2, "num_encoder_layers": 1, "num_decoder_layers": 1, "num_heads": 8,
    "dropout": 0.1, "norm": False, "num_weight_layers": 0, "no_decoder_self_att": True,
    "no_encoder_self_att": False,
    # training (ASSO_THRESH, ASSO_WEIGHT, ASSO_WEIGHT_LOCAL, NEG_UNMATCHED, CLIP_GRADIENTS.CLIP_VALUE)
    "asso_thr": 0.3, "asso_weight": 0.5, "asso_weight_local": 0.5, "neg_unmatched": True, "grad_clip": 0.1,
    # inference (INPUT.VIDEO.TEST_LEN, VIDEO_TEST.*)
    "test_len": 6, "overlap_thr": 0.2, "not_mult_thr": True, "with_iou": True, "decay_time": 0.9,
    "max_center_dist": 1.0,
}
_TRANSFORMER_PY = ("third_party", "GoMatching", "gomatching", "modeling", "roi_heads", "transformer.py")


def lst_settings(cfg: dict) -> dict:
    """Accepts the full config or just its tracker{} section."""
    t = cfg.get("tracker", cfg) or {}
    s = {k: type(v)(t.get(k, v)) for k, v in LST_DEFAULTS.items()}
    if s["test_len"] < 2:
        raise ValueError("test_len must be >= 2")
    if s["fc_dim"] % s["num_heads"]:
        raise ValueError("fc_dim must be divisible by num_heads")
    return s


def gomatching_transformer_cls():
    """GoMatching's Transformer class, imported from its file without touching third_party."""
    name = "_gomatching_roi_transformer"
    if name not in sys.modules:
        path = project_root().joinpath(*_TRANSFORMER_PY)
        if not path.is_file():
            raise ImportError(f"{path} not found; clone GoMatching into third_party/GoMatching")
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        sys.modules[name] = mod
    return sys.modules[name].Transformer


def label_detections(quads, gt_quads, gt_ids, iou: float = 0.5) -> np.ndarray:
    """GT track id per detection (Hungarian on quad IoU >= iou), 0 where unmatched."""
    out = np.zeros(len(quads), dtype=np.int64)
    for r, c in match(quad_iou_matrix(quads, gt_quads), iou):
        out[r] = int(gt_ids[c])
    return out


def quads_to_boxes(quads, img_wh) -> torch.Tensor:
    """[n,8]|[n,4,2] px quads -> [n,4] axis-aligned x1y1x2y2 normalized by (w, h), as _get_boxes_time."""
    q = torch.as_tensor(np.asarray(quads, dtype=np.float32).reshape(-1, 4, 2))
    w, h = float(img_wh[0]), float(img_wh[1])
    return torch.cat([q.min(dim=1).values, q.max(dim=1).values], dim=1) / q.new_tensor([w, h, w, h])


def box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """IoU [n,m] of x1y1x2y2 boxes (detectron2 pairwise_iou semantics)."""
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    wh = (torch.min(a[:, None, 2:], b[None, :, 2:]) - torch.max(a[:, None, :2], b[None, :, :2])).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return torch.where(inter > 0, inter / union.clamp(min=1e-12), torch.zeros_like(inter))


class MLP(nn.Module):
    """GoMatching association_head.MLP (same parameter names)."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0.0):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        if self.num_layers > 0:
            h = [hidden_dim] * (num_layers - 1)
            self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
            if self.dropout > 0.0:
                self.dropouts = nn.ModuleList(nn.Dropout(dropout) for _ in range(self.num_layers - 1))
        else:
            self.layers = []

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
            if i < self.num_layers - 1 and self.dropout > 0.0:
                x = self.dropouts[i](x)
        return x


class ATTWeightHead(nn.Module):
    """GoMatching association_head.ATTWeightHead: association logits q_proj(query) . k_proj(key)^T."""

    def __init__(self, feature_dim, num_layers, dropout):
        super().__init__()
        self.q_proj = MLP(feature_dim, feature_dim, feature_dim, num_layers, dropout)
        self.k_proj = MLP(feature_dim, feature_dim, feature_dim, num_layers, dropout)

    def forward(self, query, key):
        return torch.bmm(self.q_proj(query), self.k_proj(key).transpose(1, 2))  # B x M x N


class EmbedHead(nn.Module):
    """FCHead4Query with a flat [N, in_dim] input: fc1..fcK Linear + ReLU, c2_xavier_fill init."""

    def __init__(self, in_dim, fc_dim, num_fc):
        super().__init__()
        if num_fc == 0 and in_dim != fc_dim:
            raise ValueError(f"num_fc=0 needs in_dim == fc_dim, got {in_dim} vs {fc_dim}")
        self.fcs = []
        for k in range(num_fc):
            fc = nn.Linear(in_dim if k == 0 else fc_dim, fc_dim)
            nn.init.kaiming_uniform_(fc.weight, a=1)
            nn.init.zeros_(fc.bias)
            self.add_module(f"fc{k + 1}", fc)
            self.fcs.append(fc)

    def forward(self, x):
        for layer in self.fcs:
            x = F.relu(layer(x))
        return x


def _activate_asso(asso_output):
    """Per-frame softmax with an extra zero (background) logit, background column dropped."""
    return [torch.cat([a, a.new_zeros((a.shape[0], 1))], dim=1).softmax(dim=1)[:, :-1] for a in asso_output]


class LSTAdapter(nn.Module):
    def __init__(self, in_dim: int, cfg: dict):
        super().__init__()
        s = self.s = lst_settings(cfg)
        Transformer = gomatching_transformer_cls()
        self.feature_dim = s["fc_dim"]
        self.asso_head = EmbedHead(in_dim, s["fc_dim"], s["num_fc"])

        def matcher():
            return Transformer(
                d_model=s["fc_dim"], nhead=s["num_heads"], num_encoder_layers=s["num_encoder_layers"],
                num_decoder_layers=s["num_decoder_layers"], dim_feedforward=s["fc_dim"], dropout=s["dropout"],
                return_intermediate_dec=True, norm=s["norm"], no_decoder_self_att=s["no_decoder_self_att"],
                no_encoder_self_att=s["no_encoder_self_att"])

        self.asso_predictor = ATTWeightHead(s["fc_dim"], s["num_weight_layers"], s["dropout"])
        self.long_term_matcher = matcher()
        self.local_asso_predictor = ATTWeightHead(s["fc_dim"], s["num_weight_layers"], s["dropout"])
        self.short_term_matcher = matcher()
        self.reset()

    # ---------------------------------------------------------------- shared
    def _keep(self, frame) -> np.ndarray:
        """Indices with score_fused > asso_thr (GoMatching objectness_logits > ASSO_THRESH)."""
        return np.nonzero(np.asarray(frame["scores"], dtype=np.float64).reshape(-1) > self.s["asso_thr"])[0]

    def _forward_transformer(self, embs, query_frame=None, short_term=False):
        """lstmatcher._forward_transformer without pos-emb: embs T x [n_t, F] -> L x [M, N] logits."""
        n_t = [len(x) for x in embs]
        query_inds = None
        M = sum(n_t)
        if query_frame is not None:
            query_inds = list(range(sum(n_t[:query_frame]), sum(n_t[:query_frame + 1])))
            M = len(query_inds)
        src = torch.cat(embs, dim=0)[None]
        tr, head = (self.short_term_matcher, self.local_asso_predictor) if short_term \
            else (self.long_term_matcher, self.asso_predictor)
        feats, memory = tr(src, pos_embed=None, query_embed=None, query_inds=query_inds)
        return [head(x, memory).view(M, sum(n_t)).float() for x in feats]

    # ---------------------------------------------------------------- training
    @staticmethod
    def _asso_gt(ids):
        """ids T x [n_t] -> (ret [K,T] det index or n_t, match_cues [N] instance index or -1)."""
        n_t = [len(x) for x in ids]
        flat = torch.cat(ids)
        inst_ids = torch.unique(flat[flat > 0])
        ret = flat.new_tensor(n_t)[None].repeat(len(inst_ids), 1)
        match_cues = flat.new_full((len(flat),), -1)
        base = 0
        for t, ids_t in enumerate(ids):
            for k, inst_id in enumerate(inst_ids):
                hit = torch.nonzero(ids_t == inst_id).flatten()
                if len(hit) > 1:
                    raise ValueError(f"track id {int(inst_id)} appears {len(hit)} times in clip frame {t}")
                if len(hit) == 1:
                    ret[k, t] = hit[0]
                    match_cues[base + hit[0]] = k
            base += n_t[t]
        return ret, match_cues

    def _detr_asso_loss(self, asso_pred, asso_gt, match_cues, n_t):
        src_inds = torch.where(match_cues >= 0)[0]
        target_inds = match_cues[src_inds]
        loss = 0
        num_objs = 0
        zero = asso_pred.new_zeros((asso_pred.shape[0], 1))
        for t, pred_t in enumerate(asso_pred.split(n_t, dim=1)):
            with_bg = torch.cat([pred_t, zero], dim=1)  # M x (n_t + 1)
            if self.s["neg_unmatched"]:
                gt_t = asso_gt.new_full((asso_pred.shape[0],), n_t[t])
                gt_t[src_inds] = asso_gt[target_inds, t]
            else:
                with_bg = with_bg[src_inds]
                gt_t = asso_gt[target_inds, t]
            num_objs += (gt_t != n_t[t]).float().sum()
            loss += F.cross_entropy(with_bg, gt_t, reduction="none")
        return loss.sum() / (num_objs + 1e-4)

    def _clip_losses(self, embs, ids):
        asso_gt, cues = self._asso_gt(ids)
        n_t = [len(x) for x in embs]
        return sum(self._detr_asso_loss(x, asso_gt, cues, n_t) for x in self._forward_transformer(embs))

    def losses(self, clip: list[dict]) -> dict:
        """lstmatcher._forward_asso training branch: {loss_long_asso, loss_short_asso, loss}."""
        keeps = [self._keep(f) for f in clip]
        dev = next(self.parameters()).device
        ids = [torch.as_tensor(np.asarray(f["track_ids"], dtype=np.int64).reshape(-1)[k], device=dev)
               for f, k in zip(clip, keeps)]
        zero = torch.zeros((), device=dev)
        n_all = sum(len(k) for k in keeps)
        if n_all == 0 or not any((x > 0).any() for x in ids):
            return {"loss_long_asso": zero, "loss_short_asso": zero, "loss": zero}
        feats = torch.cat([f["feats"][torch.as_tensor(k, dtype=torch.long, device=f["feats"].device)]
                           for f, k in zip(clip, keeps)]).to(dev)
        emb = self.asso_head(feats).split([len(k) for k in keeps], dim=0)

        loss_long = self._clip_losses(list(emb), ids)
        loss_short, eff = zero, 0
        for cur in range(1, len(clip)):
            pair_ids = ids[cur - 1: cur + 1]
            if not any((x > 0).any() for x in pair_ids):
                continue
            eff += 1
            asso_gt, cues = self._asso_gt(pair_ids)
            n_t = [len(x) for x in pair_ids]
            for x in self._forward_transformer(list(emb[cur - 1: cur + 1]), short_term=True):
                loss_short = loss_short + self._detr_asso_loss(x, asso_gt, cues, n_t)
        loss_short = loss_short / (eff + 1e-4)
        out = {"loss_long_asso": self.s["asso_weight"] * loss_long,
               "loss_short_asso": self.s["asso_weight_local"] * loss_short}
        out["loss"] = out["loss_long_asso"] + out["loss_short_asso"]
        return out

    def train_step(self, clip: list[dict], optimizer=None, scaler=None) -> dict:
        """Clip losses; with an optimizer also backward (AMP scaler optional), full-model grad clip, step."""
        out = self.losses(clip)
        if optimizer is None:
            return out
        optimizer.zero_grad(set_to_none=True)
        if out["loss"].requires_grad:
            params = [p for g in optimizer.param_groups for p in g["params"]]
            if scaler is not None:
                scaler.scale(out["loss"]).backward()
                scaler.unscale_(optimizer)
            else:
                out["loss"].backward()
            if self.s["grad_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(params, self.s["grad_clip"])
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
        return {k: v.detach() for k, v in out.items()}

    # ---------------------------------------------------------------- inference
    def reset(self) -> None:
        """Start a new video."""
        self._window: deque = deque(maxlen=self.s["test_len"])
        self._frame_idx = 0
        self._next_id = 1

    def _assign(self, asso_nonk, k_boxes, nonk_boxes, ids, center_gate: bool) -> torch.Tensor:
        """Trajectory scores + Hungarian (run_short_term_match / run_long_term_match tail)."""
        Np = len(ids)
        unique_ids = torch.unique(ids)
        id_inds = (unique_ids[None, :] == ids[:, None]).float()  # Np x M
        traj_score = asso_nonk @ id_inds  # n_k x M
        last_inds = (id_inds * torch.arange(Np, device=ids.device)[:, None]).max(dim=0)[1]
        if self.s["with_iou"]:
            traj_score = torch.max(traj_score, box_iou(k_boxes, nonk_boxes[last_inds]))
        if center_gate and self.s["max_center_dist"] > 0:
            k_ct = (k_boxes[:, :2] + k_boxes[:, 2:]) / 2
            k_s = ((k_boxes[:, 2:] - k_boxes[:, :2]) ** 2).sum(dim=1)
            nonk_ct = (nonk_boxes[:, :2] + nonk_boxes[:, 2:]) / 2
            dist = ((k_ct[:, None] - nonk_ct[None, :]) ** 2).sum(dim=2) / (k_s[:, None] + 1e-8)
            valid = torch.mm((dist < self.s["max_center_dist"]).float(), id_inds).clamp_(max=1.0).bool()
            traj_score[~valid] = 0
        out = ids.new_full((len(k_boxes),), -1)
        for i, j in zip(*linear_sum_assignment(-traj_score.cpu().numpy())):
            thr = self.s["overlap_thr"] if self.s["not_mult_thr"] else self.s["overlap_thr"] * id_inds[:, j].sum()
            if traj_score[i, j] > thr:
                out[i] = unique_ids[j]
        return out

    def _short_term(self, prev: dict, cur: dict) -> torch.Tensor:
        if len(prev["ids"]) == 0:
            return cur["ids"].new_full((len(cur["ids"]),), -1)
        asso = self._forward_transformer([prev["emb"], cur["emb"]], query_frame=1, short_term=True)[-1]
        asso = torch.cat(_activate_asso(asso.split([len(prev["ids"]), len(cur["ids"])], dim=1)), dim=1)
        return self._assign(asso[:, :len(prev["ids"])], cur["boxes"], prev["boxes"], prev["ids"], False)

    def _long_term(self, window: list, cur_ids: torch.Tensor) -> torch.Tensor:
        T, k = len(window), len(window) - 1
        taken = set(int(i) for i in torch.unique(cur_ids))
        sel = [torch.tensor([int(i) not in taken for i in f["ids"]], dtype=torch.bool, device=cur_ids.device)
               for f in window[:-1]] + [cur_ids == -1]
        embs = [f["emb"][m] for f, m in zip(window, sel)]
        n_t = [len(x) for x in embs]
        Np = sum(n_t) - n_t[k]
        if Np == 0:
            return cur_ids
        asso = self._forward_transformer(embs, query_frame=k)[-1]
        asso = torch.cat(_activate_asso(asso.split(n_t, dim=1)), dim=1)[:, :Np]  # frame k is last
        if self.s["decay_time"] > 0:
            dts = torch.cat([asso.new_full((n_t[t],), T - t - 2) for t in range(k)])
            asso = asso * (self.s["decay_time"] ** dts[None, :])
        ids = torch.cat([f["ids"][m] for f, m in zip(window[:-1], sel[:-1])])
        boxes = torch.cat([f["boxes"][m] for f, m in zip(window[:-1], sel[:-1])])
        found = self._assign(asso, window[-1]["boxes"][sel[-1]], boxes, ids, True)
        out = cur_ids.clone()
        out[sel[-1]] = found
        return out

    @torch.no_grad()
    def associate(self, frame_feats: dict) -> np.ndarray:
        """Next frame of the current video -> int64 [n] track ids (-1: score_fused <= asso_thr, not tracked)."""
        if self.training:
            raise RuntimeError("call .eval() before associate()")
        n = len(np.asarray(frame_feats["scores"]).reshape(-1))
        keep = self._keep(frame_feats)
        dev = next(self.parameters()).device
        kt = torch.as_tensor(keep, dtype=torch.long, device=frame_feats["feats"].device)
        feats = frame_feats["feats"][kt].to(dev)
        boxes = quads_to_boxes(np.asarray(frame_feats["quads"], dtype=np.float32).reshape(-1, 8)[keep],
                               frame_feats["img_wh"]).to(dev)
        cur = {"boxes": boxes, "emb": self.asso_head(feats),
               "ids": torch.full((len(keep),), -1, dtype=torch.long, device=dev)}
        self._window.append(cur)
        if len(keep) and len(self._window) > 1:
            ids = self._short_term(self._window[-2], cur)
            if self._frame_idx >= 2 and (ids == -1).any():  # GoMatching: frame 1 never runs long-term
                ids = self._long_term(list(self._window), ids)
            cur["ids"] = ids
        for i in torch.nonzero(cur["ids"] == -1).flatten():
            cur["ids"][i] = self._next_id
            self._next_id += 1
        self._frame_idx += 1
        out = np.full(n, -1, dtype=np.int64)
        out[keep] = cur["ids"].cpu().numpy()
        return out
