# lrp_captum.py
from captum.attr import LRP
from captum.attr._utils.lrp_rules import EpsilonRule
import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt

def _attach_epsilon_rules(model, eps=1e-6):
    """Attach EpsilonRule to all Linear and BatchNorm1d layers."""
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.BatchNorm1d)):
            m.rule = EpsilonRule(eps)

    # ------- sanity-check -------
    missing = [
        m for m in model.modules()
        if isinstance(m, nn.BatchNorm1d) and not hasattr(m, "rule")
    ]
    if missing:
        names = [name for name, mod in model.named_modules() if mod in missing]
        raise RuntimeError(
            f"LRP: the following BatchNorm1d layers still lack a rule: {names}"
        )

def explain_features(net: nn.Module,
                     loader,
                     device: torch.device,
                     feature_names=None,
                     target_index: int = 0,
                     max_batches: int = 50,
                     eps: float = 1e-6):
    """
    Compute mean absolute LRP relevance per input feature.
    """
    net.eval()

    # 1️⃣ attach ε-rules before constructing LRP
    # _attach_epsilon_rules(net, eps)

    # 2️⃣ build LRP object
    lrp = LRP(net)

    # 3️⃣ collect attributions
    relevances = []
    for i, (x, _) in enumerate(loader):
        if i >= max_batches:
            break
        x = x.to(device).requires_grad_(True)
        attr = lrp.attribute(x, target=target_index)   # (batch, features)
        relevances.append(attr.detach().cpu().numpy())

    # 4️⃣ aggregate & normalise
    rel = np.concatenate(relevances, axis=0)          # (N, features)
    rel_mean = np.mean(np.abs(rel), axis=0)
    rel_mean /= rel_mean.sum() + 1e-12

    # 5️⃣ quick bar-plot
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(len(rel_mean))]
    idx = np.argsort(rel_mean)[::-1]

    # plt.figure(figsize=(8, len(rel_mean) // 6))
    # plt.barh(range(len(rel_mean)), rel_mean[idx])
    # plt.yticks(range(len(rel_mean)),
    #            np.array(feature_names)[idx], fontsize=6)
    # plt.xlabel("Relevance (abs-normalised)")
    # plt.title("Feature importance via LRP")
    # plt.gca().invert_yaxis()
    # plt.tight_layout()
    # plt.show()

    return rel_mean