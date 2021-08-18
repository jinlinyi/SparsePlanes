# from https://github.com/Liumouliu/Deep_blind_PnP/blob/master/model/model.py
import torch
# Sinkhorn to estimate the joint probability matrix P
class prob_mat_sinkhorn(torch.nn.Module):
    def __init__(self, mu=0.1, tolerance=1e-9, iterations=20):
        super(prob_mat_sinkhorn, self).__init__()
        self.mu = mu  # the smooth term
        self.tolerance = tolerance  # don't change
        self.iterations = iterations  # max 30 is set, enough for a typical sized mat (e.g., 1000x1000)
        self.eps = 1e-12

    def forward(self, M, r=None, c=None):
        # r, c are the prior 1D prob distribution of 3D and 2D points, respectively
        # M is feature distance between 3D and 2D point
        K = (-M / self.mu).exp()
        # 1. normalize the matrix K
        K = K / K.sum(dim=(-2, -1), keepdim=True).clamp_min_(self.eps)

        # 2. construct the unary prior

        r = r.unsqueeze(-1)
        u = r.clone()
        c = c.unsqueeze(-1)

        i = 0
        u_prev = torch.zeros_like(u)
        while (u - u_prev).norm(dim=-1).max() > self.tolerance:
            if i > self.iterations:
                break
            i += 1
            u_prev = u
            # update the prob vector u, v iteratively
            v = c / K.transpose(-2, -1).matmul(u).clamp_min_(self.eps)
            u = r / K.matmul(v).clamp_min_(self.eps)

        # assemble
        # P = torch.diag_embed(u[:,:,0]).matmul(K).matmul(torch.diag_embed(v[:,:,0]))
        P = (u * K) * v.transpose(-2, -1)
        return P


def log_sinkhorn_iterations(Z, log_mu, log_nu, iters: int):
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores, alpha=None, iters=30, match_threshold=0.05):
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    """ scores is affinity instead of distance """
    if alpha is None:
        alpha = torch.nn.Parameter(torch.tensor(0.3))
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)
    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N


    # scores = Z
    # # Get the matches with score above "match_threshold".
    # max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
    # indices0, indices1 = max0.indices, max1.indices
    # mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
    # mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
    # zero = scores.new_tensor(0)
    # mscores0 = torch.where(mutual0, max0.values.exp(), zero)
    # mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
    # valid0 = mutual0 & (mscores0 > match_threshold)
    # valid1 = mutual1 & valid0.gather(1, indices1)
    # indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
    # indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))
    return Z[:,:-1,:-1]


def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1