from captum.attr import IntegratedGradients
import torch, numpy as np, matplotlib.pyplot as plt
from pathlib import Path

def explain_features_ig(net,
                        loader,
                        device,
                        feature_names=None,
                        target_index=0,
                        max_batches=50,
                        steps=50,
                        baseline="zero"):
    """
    Mean absolute Integrated-Gradients attribution per input feature.

    Parameters
    ----------
    net : trained nn.Module in eval() mode
    loader : DataLoader yielding (x, y)
    device : cpu / cuda
    feature_names : list of str (optional)
    target_index : output neuron to explain
    max_batches : cap data for speed (None = all)
    steps : # Riemann steps in IG (20-300 typical)
    baseline : 'zero' or 'random' baseline for IG
    """
    net.eval()
    ig = IntegratedGradients(net)

    # choose a baseline tensor
    for x0, _ in loader:
        baseline_tensor = torch.zeros((1, x0.shape[1]), device=device)
        break

    attr_accum = []
    for i, (x, _) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        x = x.to(device)
        at = ig.attribute(x,
                          baselines=baseline_tensor,
                          target=target_index,
                          n_steps=steps)       # (batch, features)
        attr_accum.append(at.detach().cpu().numpy())

    attr_all = np.concatenate(attr_accum, 0)          # (N, features)
    rel = np.mean(np.abs(attr_all), axis=0)           # mean |IG|
    rel /= rel.sum() + 1e-12

    # ---- quick plot ----
    # names = feature_names or [f"f{i}" for i in range(len(rel))]
    # idx = np.argsort(rel)[::-1]
    # plt.figure(figsize=(8, len(rel)//6))
    # plt.barh(range(len(rel)), rel[idx])
    # plt.yticks(range(len(rel)), np.array(names)[idx], fontsize=6)
    # plt.gca().invert_yaxis()
    # plt.xlabel("Abs-normalised IG relevance")
    # plt.title("Feature importance via Integrated Gradients")
    # plt.tight_layout()
    # plt.show()

    return rel

def explain_all_targets(net, loader, device, feature_names, out_dir, run, steps=50, max_batches=50):
    """
    Compute and save IG-based feature importances for every output neuron.
    """
    # number of outputs
    sample_x, _ = next(iter(loader))
    n_out = net(sample_x.to(device)).shape[1]

    all_rels = {}
    feat_dir = Path(out_dir) / "feature_importance_ig"
    feat_dir.mkdir(exist_ok=True)

    for tgt in range(n_out):
        print(f" Run {run}, target {tgt}: computing IG…")
        rel = explain_features_ig(
            net,
            loader,
            device,
            feature_names=feature_names,
            target_index=tgt,
            max_batches=max_batches,
            steps=steps
        )
        # save per-target
        path = feat_dir / f"run{run}_target{tgt}_relevance.npy"
        np.save(path, rel)
        all_rels[tgt] = rel
        print(f"   → saved {path}")

    return all_rels