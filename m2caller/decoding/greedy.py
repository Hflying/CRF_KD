import torch

def greedy_decode(log_probs, lengths, blank=0):
    # log_probs: [T, B, C]; lengths: [B]
    # Return list[str] per batch
    T, B, C = log_probs.shape
    preds = log_probs.argmax(dim=2)  # [T, B]
    seqs = []
    for b in range(B):
        l = lengths[b].item() if lengths is not None else T
        last = blank
        out = []
        for t in range(l):
            p = preds[t, b].item()
            if p != blank and p != last:
                out.append(p)
            last = p
        seqs.append(out)
    return seqs
