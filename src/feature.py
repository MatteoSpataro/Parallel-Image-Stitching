import multiprocessing as mp
import cv2
import numpy as np
import params
from math import fsum
from numba import njit
from numba.typed import List
from scipy.ndimage import maximum_filter
"""
Harris corner detector based on OpenCV documentation:
https://docs.opencv.org/4.x/dc/d0d/tutorial_py_features_harris.html
"""
def harrisCornerDet(img, k=0.04, dimWindow=7):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray) / 255.0
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    Ixx = dx * dx
    Iyy = dy * dy
    # eigenvalues of M
    lambdaOne = cv2.GaussianBlur(Ixx, (dimWindow, dimWindow), sigmaX=1)
    lambdaTwo = cv2.GaussianBlur(Iyy, (dimWindow, dimWindow), sigmaX=1)

    det = lambdaOne * lambdaTwo

    trace = lambdaOne + lambdaTwo

    R = det - k * trace * trace
    return R

def printImageCorners(img, cornerResponse, path, threshold = 0.01):
    imgCorners = img.copy()

    thresholdValue = threshold * cornerResponse.max()

    corners_y, corners_x = np.where(cornerResponse > thresholdValue)

    for x, y in zip(corners_x, corners_y):
        cv2.circle(imgCorners, (x, y), radius=3, color=(0, 0, 255), thickness=-1)

    cv2.imwrite(path, imgCorners)

def extractDescription(img, cornerResponse, threshold=0.01, kernel=3,
                       cutX=params.FEATURE_CUT_X_EDGE, cutY=params.FEATURE_CUT_Y_EDGE):
    height, width = cornerResponse.shape
    maxResponse = cornerResponse.max()

    features = (cornerResponse > threshold * maxResponse).astype(np.uint8)

    features[:cutY, :] = 0  
    features[-cutY:, :] = 0
    features[:, :cutX] = 0
    features[:, -cutX:] = 0

    nms = maximum_filter(cornerResponse, size=3)
    features &= (cornerResponse == nms)

    halfKernel = kernel // 2
    ys, xs = np.where(features)
    valid = (ys >= halfKernel) & (ys < height - halfKernel) & (xs >= halfKernel) & (xs < width - halfKernel)
    ys, xs = ys[valid], xs[valid]

    numFeatures = len(ys)
    descriptors = np.empty((numFeatures, kernel * kernel), dtype=np.float32)
    positions = np.empty((numFeatures, 2), dtype=int)

    for i, (y, x) in enumerate(zip(ys, xs)):
        patch = cornerResponse[y - halfKernel : y + halfKernel + 1, x - halfKernel : x + halfKernel + 1]
        descriptors[i] = patch.flatten()
        positions[i] = [y, x]

    return descriptors, positions

def matching(descriptor1, descriptor2, featurePos1, featurePos2, yRange=10):
    numTasks = 32
    partitionDescriptors = np.array_split(descriptor1, numTasks)
    partitionPositions = np.array_split(featurePos1, numTasks)
    results = []
    
    for i in range(numTasks):
        matches = computeMatch(partitionDescriptors[i], descriptor2, partitionPositions[i], featurePos2, yRange)
        results.extend(matches)
    return results

def matchingPar(descriptor1, descriptor2, featurePos1, featurePos2, pool, yRange=10):
    processes = pool._processes
    partitionDescriptors = np.array_split(descriptor1, processes)
    partitionPositions   = np.array_split(featurePos1, processes)
    tasks = [
        (partitionDescriptors[i],
         descriptor2,
         partitionPositions[i],
         featurePos2,
         yRange)
        for i in range(processes)
    ]
    chunksize = max(1, len(tasks) // processes)

    results = pool.starmap(computeMatch, tasks, chunksize=chunksize)

    matchedPairs = [pair for sub in results for pair in sub]
    return matchedPairs

def computeMatch(descriptor1, descriptor2, feature1, feature2, yRange=10):
    matchedPairs = []
    matchedScores = []
    descriptor2 = np.asarray(descriptor2)
    feature2 = np.asarray(feature2)

    for i in range(len(descriptor1)):
        y = feature1[i][0]
        
        mask = (feature2[:, 0] >= y - yRange) & (feature2[:, 0] <= y + yRange)
        candidateDesc = descriptor2[mask]
        candidatePos = feature2[mask]

        if len(candidateDesc) == 0:
            continue
        elif len(candidateDesc) == 1:
            d = descriptor1[i] - candidateDesc[0]
            dist = np.sum(d**2)
            matchedPairs.append((tuple(feature1[i]), tuple(candidatePos[0])))
            matchedScores.append(dist)
            continue
        elif len(candidateDesc) == 2:
            diffs = candidateDesc - descriptor1[i]
            dists = np.sum(diffs**2, axis=1)
            idx = np.argsort(dists)
            d1, d2 = dists[idx[0]], dists[idx[1]]
            if d2 == 0 or d1 < 0.35 * 256:
                matchedPairs.append((tuple(feature1[i]), tuple(candidatePos[idx[0]])))
                matchedScores.append(d1)
            continue

        diffs = candidateDesc - descriptor1[i]
        dists = np.sum(diffs**2, axis=1)
        top2_idx = np.argpartition(dists, 2)[:2]
        d1, d2 = dists[top2_idx[0]], dists[top2_idx[1]]
        if d1 > d2: d1, d2 = d2, d1
        if d2 == 0 or d1 < 0.35 * 256:
            matchedPairs.append((tuple(feature1[i]), tuple(candidatePos[top2_idx[0]])))
            matchedScores.append(d1)

    matchedScores = np.argsort(matchedScores)
    uniqueTargets = set()
    bestPairs = []
    for idx in matchedScores:
        target = matchedPairs[idx][1]
        if target not in uniqueTargets:
            uniqueTargets.add(target)
            bestPairs.append(matchedPairs[idx])

    return np.array(bestPairs)