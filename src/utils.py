import cv2
import numpy as np
import matplotlib.pyplot as plt
import params
import glob

def loadImages():
    imagesPath = glob.glob(params.INPUT_PATH+"*.jpg")
    if not imagesPath:
        raise ValueError("There aren't images .jpg in path "+params.INPUT_PATH)
    rgbImages = []
    for imagePath in imagesPath:
        rgbImg = cv2.imread(imagePath)
        rgbImages.append(rgbImg)
    return rgbImages

def cylindricalProjection(img, FOV=params.FOV):
    height, width, _ = img.shape
    focalLength = width / (2 * np.tan(np.radians(FOV / 2)))

    x, y = np.meshgrid(np.arange(-width // 2, width // 2), np.arange(-height // 2, height // 2))

    cylinder_x = focalLength * np.arctan(x / focalLength)
    cylinder_y = focalLength * y / np.sqrt(x**2 + focalLength**2)

    # Map the cylindrical coordinates to the image size
    cylinder_x = np.round(cylinder_x + width // 2).astype(int)
    cylinder_y = np.round(cylinder_y + height // 2).astype(int)

    cylinder_proj = np.zeros_like(img)

    valid_pixels = (cylinder_x >= 0) & (cylinder_x < width) & (cylinder_y >= 0) & (cylinder_y < height)

    cylinder_proj[cylinder_y[valid_pixels], cylinder_x[valid_pixels]] = img[y[valid_pixels] + height // 2, x[valid_pixels] + width // 2]

    _, thresh = cv2.threshold(cv2.cvtColor(cylinder_proj, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
    x, y, w, h = cv2.boundingRect(thresh)

    return cylinder_proj[y:y+h, x:x+w]

def paddingImages(p1, p2):
    h1, w1 = p1.shape[:2]
    h2, w2 = p2.shape[:2]

    if h1 != h2:
        if h1 < h2:
            pad_height = h2 - h1
            p1 = np.pad(p1, ((0, pad_height), (0, 0), (0, 0)), mode='constant', constant_values=0)
        else:
            pad_height = h1 - h2
            p2 = np.pad(p2, ((0, pad_height), (0, 0), (0, 0)), mode='constant', constant_values=0)
    
    return p1, p2

def plotMatchedPairs(p1, p2, mp, index):
    offset = p1.shape[1] 
    p1, p2 = paddingImages(p1, p2)
    plotImg = np.concatenate((p1, p2), axis=1)
    plt.figure(figsize=(10,10))
    plt.axis('off')
    plt.imshow(plotImg)
    print(len(mp))
    for i in range(len(mp)):
        plt.scatter(x=mp[i][0][1], y=mp[i][0][0], c='r')
        plt.plot([mp[i][0][1], offset+mp[i][1][1]], [mp[i][0][0], mp[i][1][0]], 'y-', lw=1)
        plt.scatter(x=offset+mp[i][1][1], y=mp[i][1][0], c='b')
    
    if params.DEBUG: plt.savefig(params.OUTPUT_PATH+"MatchImg"+str(index) +'.jpg', dpi=300)