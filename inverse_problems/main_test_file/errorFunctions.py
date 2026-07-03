import numpy as np

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