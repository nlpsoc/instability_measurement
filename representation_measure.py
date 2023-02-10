# largely copied from https://github.com/moskomule/anatome
import torch
import numpy as np
from scipy import interpolate


def linear_interpolate(x, target_size, dim=-2):
    axis = np.linspace(0, x.shape[dim] - 1, x.shape[dim])
    target_axis = np.linspace(0, x.shape[dim] - 1, target_size)
    f_interpolate = interpolate.interp1d(axis, x, axis=dim)
    aug_x = f_interpolate(target_axis)
    return aug_x


def _zero_mean(x):
    return x - x.mean(dim=1, keepdim=True)


def _svd(x, diag_only=False):
    r"""
    SVD
    Args:
        x: A num_layers x num_examples x num_features_x matrix of features.
        diag_only: whether return coordinates

    Returns:
        u: A num_layers x num_examples x num_examples matrix
        diag: A num_layers x num_examples matrix
        v: A num_layers x num_features x num_examples matrix
    """
    u = v = None
    if diag_only:
        diag = torch.linalg.svdvals(x)
    else:
        u, diag, vh = torch.linalg.svd(x, full_matrices=False)
        v = vh.transpose(-2, -1)
    return u, diag, v


def _svd_reduction(x, accept_rate=0.99):
    u, diag, v = _svd(x, diag_only=False)
    full = diag.abs().sum(dim=-1, keepdims=True)
    rate = diag.abs().cumsum(dim=-1) / full
    num_accept = torch.sum(rate <= accept_rate, dim=-1)
    ones = torch.ones_like(num_accept)
    num_accept = torch.where(num_accept == 0, ones, num_accept)  # fix nan when the first singular value is too large
    transformed_x = torch.bmm(x, v[:, :, :num_accept.max()])  # num_layers, num_features, num_accept
    return transformed_x, num_accept


def cca_by_svd(x, y, diag_only=False):
    r"""
    SVD-based CCA
    Args:
        x: A num_layers x num_examples x num_features_x matrix of features.
        y: A num_layers x num_examples x num_features_y matrix of features.
        diag_only: whether return coordinates

    Returns:
        rho: A num_layers x num_examples matrix
        a: A num_layers x num_feature_x x num_examples (coordinates) matrix
        b: A num_layers x num_feature_y x num_examples (coordinates) matrix
    """
    center_x = _zero_mean(x)
    center_y = _zero_mean(y)
    ux, diag_x, vx = _svd(center_x, diag_only=False)
    uy, diag_y, vy = _svd(center_y, diag_only=False)
    uu = torch.bmm(ux.transpose(1, 2), uy)
    u, rho, v = _svd(uu, diag_only=diag_only)
    a = b = None
    if not diag_only:
        a = torch.bmm(vx * diag_x.reciprocal_().unsqueeze(dim=1), u)
        b = torch.bmm(vy * diag_y.reciprocal_().unsqueeze(dim=1), v)
    return rho, a, b


def cca_dist(x, y):
    """
    CCA distance
    Args:
        x: A num_layers x num_examples x num_features_x matrix of features.
        y: A num_layers x num_examples x num_features_y matrix of features.

    Returns:
        dist: A num_layers vector

    """
    rho = cca_by_svd(x, y, diag_only=True)[0]
    dist = 1 - rho.sum(dim=1) / rho.size(1)
    return dist


def svcca_dist(x, y, accept_rate=0.99):
    r"""
    SVCCA distance
    Args:
        x: A num_layers x num_examples x num_features_x matrix of features.
        y: A num_layers x num_examples x num_features_y matrix of features.
        accept_rate: accept rate of SVCCA

    Returns:
        dist: A num_layers vector
    """
    transformed_x, num_accept_x = _svd_reduction(x, accept_rate=accept_rate)
    transformed_y, num_accept_y = _svd_reduction(y, accept_rate=accept_rate)
    rho_sum = []
    for idx in range(transformed_x.size(0)):
        current_x = transformed_x[idx, :, :num_accept_x[idx]].unsqueeze(dim=0)
        current_y = transformed_y[idx, :, :num_accept_y[idx]].unsqueeze(dim=0)
        current_rho = cca_by_svd(current_x, current_y, diag_only=True)[0]
        rho_sum.append(current_rho.sum())
    min_num_accept = torch.minimum(num_accept_x, num_accept_y)
    dist = 1 - torch.FloatTensor(rho_sum).to(x.device) / min_num_accept
    return dist


def pwcca_dist(x, y):
    r"""
    PWCCA distance
    Args:
        x: A num_layers x num_examples x num_features_x matrix of features.
        y: A num_layers x num_examples x num_features_y matrix of features.

    Returns:
        dist: A num_layers vector
    """
    rho, a, _ = cca_by_svd(x, y, diag_only=False)
    alpha = torch.bmm(x, a).abs_().sum(dim=1)  # num_layers, num_examples (num_coordinates)
    alpha = alpha / alpha.sum(dim=-1, keepdims=True)
    dist = 1 - (alpha * rho).sum(dim=1)
    return dist


def op_dist(x, y):
    r"""
    Orthogonal Procrustes distance
    Args:
        x: A num_layers x num_examples x num_features_x matrix of features.
        y: A num_layers x num_examples x num_features_y matrix of features.

    Returns:
        dist: A num_layers vector
    """
    centered_x = _zero_mean(x)
    centered_y = _zero_mean(y)
    normalized_x = centered_x / torch.linalg.norm(centered_x, ord='fro', dim=(1, 2), keepdim=True)
    normalized_y = centered_y / torch.linalg.norm(centered_y, ord='fro', dim=(1, 2), keepdim=True)
    nuclear_norm = torch.linalg.norm(torch.bmm(normalized_x.transpose(1, 2), normalized_y), dim=(1, 2), ord='nuc')
    return 1 - nuclear_norm


# Adapted from https://github.com/google-research/google-research/blob/master/representation_similarity/Demo.ipynb
def linear_kernel(x):
    return torch.bmm(x, x.transpose(1, 2))  # num_layers, num_examples, num_examples


def center_gram(x_gram):
    num_examples = x_gram.size(1)
    eye_mask = torch.cat([torch.eye(num_examples).unsqueeze(0)
                          for _ in range(x_gram.size(0))], dim=0).bool().to(x_gram.device)
    x_gram = x_gram.masked_fill(eye_mask, 0)
    means = torch.sum(x_gram, dim=1) / (num_examples - 2)
    means = means - torch.sum(means, dim=1, keepdim=True) / (2 * (num_examples - 1))  # num_layers x num_samples
    x_gram = x_gram - (means.unsqueeze(dim=-1) + means.unsqueeze(dim=1))
    x_gram = x_gram.masked_fill(eye_mask, 0)
    return x_gram


def unbiased_linear_cka_dist(x, y):
    r"""
    Batch GPU linear cka
    Args:
        x: A num_layers x num_examples x num_features_x matrix of features.
        y: A num_layers x num_examples x num_features_y matrix of features.

    Returns:
        Use unbiased estimator of HSIC. CKA may still be biased.
    """
    x_gram = center_gram(linear_kernel(x)).view(x.size(0), 1, -1)
    y_gram = center_gram(linear_kernel(y)).view(x.size(0), -1, 1)
    scaled_hsic = torch.bmm(x_gram, y_gram).squeeze(dim=-1).squeeze(dim=-1)
    x_norm = torch.linalg.matrix_norm(x_gram, ord='fro', dim=(1, 2))
    y_norm = torch.linalg.matrix_norm(y_gram, ord='fro', dim=(1, 2))
    return 1 - scaled_hsic / (x_norm * y_norm)


def fast_biased_cka_dist(x, y):
    centered_x = _zero_mean(x)
    centered_y = _zero_mean(y)
    dot_prod = torch.linalg.norm(torch.bmm(centered_x.transpose(1, 2), centered_y), ord='fro', dim=(1, 2)).pow(2)
    norm_x = torch.linalg.norm(torch.bmm(centered_x.transpose(1, 2), centered_x), ord='fro', dim=(1, 2))
    norm_y = torch.linalg.norm(torch.bmm(centered_y.transpose(1, 2), centered_y), ord='fro', dim=(1, 2))
    dist = 1 - dot_prod / (norm_x * norm_y)

    return dist
