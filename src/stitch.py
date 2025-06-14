import numpy as np
import cv2
import params
from params import RANSAC_K, RANSAC_THRESHOLD

def RANSAC(matchedPairs, prevFit, ransacThreshold=RANSAC_THRESHOLD):
    np.random.seed(42)
    matchedPairs = np.asarray(matchedPairs)
    if len(matchedPairs) < 4:
        raise ValueError("Too few matches for RANSAC.")

    isRandom = len(matchedPairs) > 4

    bestFit = np.array([0, 0])
    maxInliers = 0

    points1 = matchedPairs[:, 0]
    points2 = matchedPairs[:, 1]

    for _ in range(RANSAC_K):
        idx = np.random.randint(0, len(matchedPairs)) if isRandom else 0
        fit = points2[idx] - points1[idx]

        predict = points1 + fit
        diffs = points2 - predict

        dists = np.linalg.norm(diffs, axis=1)

        inliers = np.sum(dists < ransacThreshold)
        if inliers > maxInliers:
            maxInliers = inliers
            bestFit = fit

    if maxInliers == 0:
        raise ValueError("No inlier found.")
    if maxInliers < 3:
        raise ValueError("Too few inliers to compute a reliable transformation.")
    if prevFit[1] * bestFit[1] < 0:
        print('[WARNING] Sudden Y-direction change in fit:', bestFit)

    return bestFit

def stitching(img1, img2, shift, blendWindow=params.BLEND_WINDOW):
    padding = [
        (shift[0], 0) if shift[0] > 0 else (0, -shift[0]),
        (shift[1], 0) if shift[1] > 0 else (0, -shift[1]),
        (0, 0)
    ]
    shiftedImg1 = np.lib.pad(img1, padding, 'constant', constant_values=0)

    split = img2.shape[1] + abs(shift[1])
    splited = shiftedImg1[:, split:] if shift[1] > 0 else shiftedImg1[:, :-split]
    shiftedImg1 = shiftedImg1[:, :split] if shift[1] > 0 else shiftedImg1[:, -split:]

    h1, w1, _ = shiftedImg1.shape
    h2, w2, _ = img2.shape

    inverseShift = [h1 - h2, w1 - w2]
    inversePadding = [
        (inverseShift[0], 0) if shift[0] < 0 else (0, inverseShift[0]),
        (inverseShift[1], 0) if shift[1] < 0 else (0, inverseShift[1]),
        (0, 0)
    ]
    shiftedImg2 = np.lib.pad(img2, inversePadding, 'constant', constant_values=0)

    indexX = shiftedImg1.shape[1] // 2
    blendedRows = []

    for y in range(h1):
        blendedRow = pyramidBlendRow(shiftedImg1[y], shiftedImg2[y], indexX, blendWindow)
        blendedRows.append(blendedRow)

    shiftedImg1 = np.asarray(blendedRows)
    shiftedImg1 = np.concatenate((shiftedImg1, splited) if shift[1] > 0 else (splited, shiftedImg1), axis=1)

    return shiftedImg1

def stitchingPar(img1, img2, shift, pool, blendWindow=params.BLEND_WINDOW):
    padding = [
        (shift[0], 0) if shift[0] > 0 else (0, -shift[0]),
        (shift[1], 0) if shift[1] > 0 else (0, -shift[1]),
        (0, 0)
    ]
    shiftedImg1 = np.lib.pad(img1, padding, 'constant', constant_values=0)

    # split between overlapped and not-overlapped parts
    split = img2.shape[1] + abs(shift[1])
    splited = (shiftedImg1[:, split:]
               if shift[1] > 0 else shiftedImg1[:, :-split])
    shiftedImg1 = (shiftedImg1[:, :split]
                   if shift[1] > 0 else shiftedImg1[:, -split:])

    h1, w1, _ = shiftedImg1.shape
    h2, w2, _ = img2.shape
    inverseShift = [h1 - h2, w1 - w2]
    inversePadding = [
        (inverseShift[0], 0) if shift[0] < 0 else (0, inverseShift[0]),
        (inverseShift[1], 0) if shift[1] < 0 else (0, inverseShift[1]),
        (0, 0)
    ]
    shiftedImg2 = np.lib.pad(img2, inversePadding, 'constant', constant_values=0)

    indexX = shiftedImg1.shape[1] // 2
    num_workers = pool._processes
    stripes = np.array_split(np.arange(h1), num_workers)

    tasks = []
    for stripe in stripes:
        block1 = shiftedImg1[stripe]
        block2 = shiftedImg2[stripe]
        tasks.append((block1, block2, indexX, blendWindow, stripe))

    results = pool.starmap(pyramidBlendBlock, tasks)

    blended = np.zeros_like(shiftedImg1)
    for block, stripe in results:
        blended[stripe] = block

    output = (np.concatenate((blended, splited), axis=1)
              if shift[1] > 0 else np.concatenate((splited, blended), axis=1))

    return output
	
def pyramidBlendBlock(block1, block2, indexX, levels, stripe_indices):
    blended = [pyramidBlendRow(r1, r2, indexX, levels) 
               for (r1, r2) in zip(block1, block2)]
    return np.asarray(blended), stripe_indices

def buildGaussianPyramid(img, levels):
    pyramid = [img]
    for _ in range(levels):
        img = cv2.pyrDown(img)
        pyramid.append(img)
    return pyramid

def buildLaplacianPyramid(img, levels):
    g_pyr = buildGaussianPyramid(img, levels)
    l_pyr = []
    for i in range(levels):
        size = (g_pyr[i].shape[1], g_pyr[i].shape[0])
        l_img = g_pyr[i] - cv2.pyrUp(g_pyr[i+1], dstsize=size)
        l_pyr.append(l_img)
    l_pyr.append(g_pyr[-1])
    return l_pyr

def imageFromLaplacian(laplacianPyramid):
    img = laplacianPyramid[-1]
    for i in range(len(laplacianPyramid) - 2, -1, -1):
        size = (laplacianPyramid[i].shape[1], laplacianPyramid[i].shape[0])
        img = cv2.pyrUp(img, dstsize=size) + laplacianPyramid[i]
    return img

def pyramidBlendRow(row1, row2, indexX, levels=4, direction='left'):
    if direction == 'right':
        row1, row2 = row2, row1

    W = row1.shape[0]

    mask = np.zeros((W, 1), dtype=np.float32)
    mask[:indexX] = 1.0
    mask = cv2.GaussianBlur(mask, (31, 1), 5)


    mask = np.repeat(mask, 3, axis=1)

    L1 = buildLaplacianPyramid(row1.astype(np.float32), levels)
    L2 = buildLaplacianPyramid(row2.astype(np.float32), levels)
    GM = buildGaussianPyramid(mask, levels)

    LS = [m * l1 + (1 - m) * l2 for l1, l2, m in zip(L1, L2, GM)]

    blendedRow = imageFromLaplacian(LS)
    blendedRow = np.clip(blendedRow, 0, 255).astype(np.uint8)

    return blendedRow

def end2endAlign(img, shifts):
    sum_y, sum_x = np.sum(shifts, axis=0)
    y_shift = np.abs(sum_y)
    col_shift = None

    if sum_x*sum_y > 0:
        col_shift = np.linspace(y_shift, 0, num=img.shape[1], dtype=np.uint16)
    else:
        col_shift = np.linspace(0, y_shift, num=img.shape[1], dtype=np.uint16)

    aligned = img.copy()
    for x in range(img.shape[1]):
        aligned[:,x] = np.roll(img[:,x], col_shift[x], axis=0)

    return aligned


def crop(img):
    _, thresh = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
    upper, lower = [-1, -1]
    black_pixel_num_threshold = img.shape[1]//100

    for y in range(thresh.shape[0]):
        if len(np.where(thresh[y] == 0)[0]) < black_pixel_num_threshold:
            upper = y
            break
        
    for y in range(thresh.shape[0]-1, 0, -1):
        if len(np.where(thresh[y] == 0)[0]) < black_pixel_num_threshold:
            lower = y
            break

    return img[upper:lower, :]