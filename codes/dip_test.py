# -*- coding: utf-8 -*-

import numpy as np

def dip(samples, num_bins=100, p=0.95):

    samples = samples / np.abs(samples).max()
    pdf, idxs = np.histogram(samples, bins=num_bins)
    idxs = idxs[:-1] + np.diff(idxs)
    pdf  = pdf / pdf.sum()

    cdf = np.cumsum(pdf, dtype=float)
    assert np.abs(cdf[-1] - 1) < 1e-3

    D = 0
    ans = 0
    check = False
    while True:
        gcm_values, gcm_contact_points   = gcm_cal(cdf, idxs)
        lcm_values, lcm_contact_points = lcm_cal(cdf, idxs)

        d_gcm, gcm_diff = sup_diff(gcm_values, lcm_values, gcm_contact_points)
        d_lcm, lcm_diff = sup_diff(gcm_values, lcm_values, lcm_contact_points)

        if d_gcm > d_lcm:
            xl = gcm_contact_points[d_gcm == gcm_diff][0]
            xr = lcm_contact_points[lcm_contact_points >= xl][0]
            d  = d_gcm
        else:
            xr = lcm_contact_points[d_lcm == lcm_diff][-1]
            xl = gcm_contact_points[gcm_contact_points <= xr][-1]
            d  = d_lcm

        gcm_diff_ranged = np.abs(gcm_values[:xl+1] - cdf[:xl+1]).max()
        lcm_diff_ranged = np.abs(lcm_values[xr:]  - cdf[xr:]).max()

        if d <= D or xr == 0 or xl == cdf.size:
            ans = D
            break
        else:
            D = max(D, gcm_diff_ranged, lcm_diff_ranged)

        cdf = cdf[xl:xr+1]
        idxs = idxs[xl:xr+1]
        pdf = pdf[xl:xr+1]

    p_threshold = p_table(p, samples.size, 10000)
    if ans < p_threshold:
        check = True

    return ans, check


def gcm_cal(cdf, idxs):
    local_cdf = np.copy(cdf)
    local_idxs = np.copy(idxs)
    gcm = [local_cdf[0]]
    contact_points = [0]
    while local_cdf.size > 1:
        distances = local_idxs[1:] - local_idxs[0]
        slopes = (local_cdf[1:] - local_cdf[0]) / distances
        slope_min = slopes.min()
        slope_min_idx = np.where(slopes == slope_min)[0][0] + 1
        gcm.append(local_cdf[0] + distances[:slope_min_idx] * slope_min)
        contact_points.append(contact_points[-1] + slope_min_idx)
        local_cdf = local_cdf[slope_min_idx:]
        local_idxs = local_idxs[slope_min_idx:]
    return np.hstack(gcm), np.hstack(contact_points)


def lcm_cal(cdf, idxs):
    values, points = gcm_cal(1-cdf[::-1], idxs.max() - idxs[::-1])
    return 1 - values[::-1], idxs.size - points[::-1] - 1


def sup_diff(alpha, beta, contact_points):
    diff = np.abs((alpha[contact_points] - beta[contact_points]))
    return diff.max(), diff


def p_table(p, sample_size, n_samples):
    data = [np.random.randn(sample_size) for _ in range(n_samples)]
    dips = np.hstack([dip(samples) for samples in data])
    return np.percentile(dips, p*100)