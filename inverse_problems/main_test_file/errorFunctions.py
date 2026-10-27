import numpy as np

"""
def angularError(result, groundTruth, dimensions):
    domainFactor = float(1.0/result.size)
    result.reshape(dimensions)
    result = ([result, 1.0]/

    sum = 0.0
    for i in range(result.size):
        sumTmp = 0.0
        for j in range(numDim):
            sumTmp += (result[j]-groundTruth[j])**2
        sum += np.sqrt(sumTmp)
    return sum*domainFactor

def averageAngularError(result, groundTruth, dimensions):
    domainFactor = float(1.0/result.size)
    result.reshape(dimensions)
    result = ([result, 1.0]/

    sum = 0.0
    for i in range(result.size):
        sumTmp = 0.0
        for j in range(numDim):
            sumTmp += (result[j]-groundTruth[j])**2
        sum += np.sqrt(sumTmp)
    return sum*domainFactor

def absoluteEndpointError(result, groundTruth, dimensions):
    domainFactor = float(1.0/result.size)
    result.reshape(dimensions)
    sum = 0.0
    for i in range(result.size):
        sumTmp = 0.0
        for j in range(numDim):
            sumTmp += (result[j]-groundTruth[j])**2
        sum += np.sqrt(sumTmp)
    return sum*domainFactor
"""

# Endpoint Error (EPE)
def compute_epe(x_pred, y_pred, x_gt, y_gt, mask=None):
    epe = np.sqrt((x_pred - x_gt)**2 + (y_pred - y_gt)**2)
    return float(epe[mask].mean() if mask is not None else epe.mean())

# Average Angular Error (AAE)
def compute_aae(x_pred, y_pred, x_gt, y_gt, mask=None):
    num = x_pred*x_gt + y_pred*y_gt + 1.0
    den = (np.sqrt(x_pred**2 + y_pred**2 + 1.0) * np.sqrt(x_gt**2   + y_gt**2   + 1.0))
    cos_angle = np.clip(num / (den + 1e-10), -1.0, 1.0)
    aae = np.degrees(np.arccos(cos_angle))
    return float(aae[mask].mean() if mask is not None else aae.mean())

# KITTI Fl outlier rate
def compute_fl(x_pred, y_pred, x_gt, y_gt, mask=None):
    epe = np.sqrt((x_pred - x_gt)**2 + (y_pred - y_gt)**2)
    gt_mag = np.sqrt(x_gt**2 + y_gt**2)
    outlier = (epe > 3.0) & (epe > 0.05 * gt_mag)
    if mask is not None:
        return float(outlier[mask].mean() * 100)
    return float(outlier.mean() * 100)