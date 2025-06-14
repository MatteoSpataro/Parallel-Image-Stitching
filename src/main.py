import sys
import cv2
import math
import numpy as np
import multiprocessing as mp
import feature
import utils
import stitch
import params 
import time
import cProfile
import pstats
import argparse

def sequentialStitching(ransacThreshold=params.RANSAC_THRESHOLD, 
                      FOV=params.FOV, blendWindow=params.BLEND_WINDOW, 
                      cutX=params.FEATURE_CUT_X_EDGE, cutY=params.FEATURE_CUT_Y_EDGE,
                      dimWindow=params.GAUSSIAN_WINDOW):
    images = utils.loadImages()
    
    cylinderImages = []
    for i in range(len(images)):
        cylinderImage = utils.cylindricalProjection(images[i])
        cylinderImages.append(cylinderImage)

    stitchedImg = cylinderImages[0].copy()

    shifts = [[0, 0]]
    cacheFeatures = [[], []]

    for i in range(1, len(cylinderImages)):
        img1 = cylinderImages[i-1]
        img2 = cylinderImages[i]

        descriptors1, position1 = cacheFeatures
        if len(descriptors1) == 0:
            corner1 = feature.harrisCornerDet(img1, dimWindow = dimWindow)
            descriptors1, position1 = feature.extractDescription(img1, corner1, 
                                        kernel=params.DESCRIPTOR_SIZE, threshold=params.FEATURE_THRESHOLD)

        corner2 = feature.harrisCornerDet(img2, dimWindow = dimWindow)
        descriptors2, position2 = feature.extractDescription(img2, corner2, 
                                    kernel=params.DESCRIPTOR_SIZE, threshold=params.FEATURE_THRESHOLD)

        cacheFeatures = [descriptors2, position2]
        
        matchedPairs = feature.matching(descriptors1, descriptors2, position1, 
                            position2, yRange=params.MATCHING_Y_RANGE)

        if params.DEBUG: utils.plotMatchedPairs(img1, img2, matchedPairs, i)
        shift = stitch.RANSAC(matchedPairs, shifts[-1])
        shifts += [shift]

        stitchedImg = stitch.stitching(stitchedImg, img2, shift)
        if params.DEBUG: cv2.imwrite(params.OUTPUT_PATH+str(i) +'.jpg', stitchedImg)

    aligned = stitch.end2endAlign(stitchedImg, shifts)
    if params.DEBUG: cv2.imwrite(params.OUTPUT_PATH+'aligned_outputSeq.jpg', aligned)
    output = stitch.crop(aligned)
    cv2.imwrite(params.OUTPUT_PATH+"outputSeq"+str(FOV)+"fov_"+str(blendWindow)+"BW_"
                +str(cutX)+"cutX_"+str(cutY)+"cutY_"+str(dimWindow)+"GW.jpg", output)

def parallelStitching(cores = params.CORES, ransacThreshold=params.RANSAC_THRESHOLD, 
                      FOV=params.FOV, blendWindow=params.BLEND_WINDOW, 
                      cutX=params.FEATURE_CUT_X_EDGE, cutY=params.FEATURE_CUT_Y_EDGE,
                      dimWindow=params.GAUSSIAN_WINDOW):
    pool = mp.Pool(cores)

    images = utils.loadImages()
    args = [(img, FOV) for img in images]
    
    cylinderImages = pool.starmap(utils.cylindricalProjection, args) #parallel upload of images
    if params.DEBUG: print("Images found: "+str(len(images)))
    stitchedImg = cylinderImages[0].copy()

    shifts = [[0, 0]]
    cacheFeatures = [[], []]
    
    for i in range(1, len(cylinderImages)):
        img1 = cylinderImages[i-1]
        img2 = cylinderImages[i]

        descriptors1, position1 = cacheFeatures
        if len(descriptors1) == 0:
            corner1 = feature.harrisCornerDet(img1, dimWindow = dimWindow)
            descriptors1, position1 = feature.extractDescription(img1, corner1, 
                                        kernel=params.DESCRIPTOR_SIZE, threshold=params.FEATURE_THRESHOLD)
        
        corner2 = feature.harrisCornerDet(img2, dimWindow = dimWindow)
        if params.DEBUG:
            feature.printImageCorners(img2, corner2, params.OUTPUT_PATH+str(i)+"harris_"
                +str(params.FEATURE_THRESHOLD)+"threshold_"+str(cores)+"cores_"+str(FOV)
                +"fov_"+str(blendWindow)+"BW_"+str(cutX)+"cutX_"+str(cutY)+"cutY_"
                +str(dimWindow)+"GW.jpg", threshold = params.FEATURE_THRESHOLD)
        descriptors2, position2 = feature.extractDescription(img2, corner2, 
                                    kernel=params.DESCRIPTOR_SIZE, threshold=params.FEATURE_THRESHOLD)

        cacheFeatures = [descriptors2, position2]
        #parallel match of pairs
        matchedPairs = feature.matchingPar(descriptors1, descriptors2, position1, 
                                           position2, pool, yRange=params.MATCHING_Y_RANGE)

        if params.DEBUG: 
            utils.plotMatchedPairs(img1, img2, matchedPairs, i)
            cv2.imwrite(params.OUTPUT_PATH+str(i) +'.jpg', stitchedImg)

        shift = stitch.RANSAC(matchedPairs, shifts[-1], ransacThreshold)
        shifts.append(shift)
        stitchedImg = stitch.stitchingPar(stitchedImg, img2, shift, pool, blendWindow)
        if params.DEBUG: 
            cv2.imwrite(params.OUTPUT_PATH+str(i) +'.jpg', stitchedImg)
            print("Computed "+str(i)+" of "+str(len(cylinderImages)))

    pool.close()
    aligned = stitch.end2endAlign(stitchedImg, shifts)
    if params.DEBUG: 
        cv2.imwrite(params.OUTPUT_PATH+"alignedPar_"+str(cores)+"cores_"+str(FOV)
            +"fov_"+str(blendWindow)+"BW_"+str(dimWindow)+"GW.jpg", aligned)

    output = stitch.crop(aligned)
    cv2.imwrite(params.OUTPUT_PATH+"outputPar_"+str(cores)+"cores_"+str(FOV)
        +"fov_"+str(blendWindow)+"BW_"+str(cutX)+"cutX_"+str(cutY)
        +"cutY_"+str(dimWindow)+"GW_"+str(ransacThreshold)+"ransacT.jpg", output)

def printTimes(seqTime, parTime, seqTimeCpu, parTimeCpu):
    print("Process time (ns): "+str(parTimeCpu))
    if parTimeCpu < seqTimeCpu:
        diff = seqTimeCpu - parTimeCpu
        print("Time saved (ns): "+str(diff))
        percentage = diff*100 / seqTimeCpu
        print(str(percentage)+"% faster")
    else:
        diff = parTimeCpu - seqTimeCpu
        print("Time lost (ns): "+str(diff))
        percentage = diff*100 / parTimeCpu
        print(str(percentage)+"% slower")
    print("Total time (s): "+str(parTime))
    diff = seqTime - parTime
    print("Total time saved (s): "+str(diff))
    percentage = diff*100 / seqTime
    print(str(percentage)+"% faster")

def runParallel(cores, seqTimeCpu, seqTime):
    startTime = time.time()
    start = time.process_time_ns()
    parallelStitching(cores)    
    endTime = time.time()
    end = time.process_time_ns()      
    parTime = endTime - startTime
    parTimeCpu = end - start

    printTimes(seqTime, parTime, seqTimeCpu, parTimeCpu)

def testCores():
    print(mp.cpu_count())
    startCpu = time.process_time_ns()
    startTime = time.time()
    sequentialStitching()
    endTime = time.time()
    endCpu = time.process_time_ns()
    seqTimeCpu = endCpu - startCpu 
    seqTime = endTime - startTime
    print("Sequential version:")
    print("Process time (ns) = "+str(seqTimeCpu))
    print("Total time (s) = "+str(seqTime))
    
    for cores in [14, 16]:
        print("Run with "+str(cores)+" cores.")
        runParallel(cores, seqTimeCpu, seqTime)

def seq():
    startCpu = time.process_time_ns()
    startTime = time.time()
    with cProfile.Profile() as profile:
       sequentialStitching()
    endTime = time.time()
    endCpu = time.process_time_ns()
    seqTimeCpu = endCpu - startCpu 
    seqTime = endTime - startTime
    print("Process time (ns): "+str(seqTimeCpu))
    print("Total time (s): "+str(seqTime))
    
    results = pstats.Stats(profile)
    results.sort_stats(pstats.SortKey.TIME)
    results.print_stats(20)
    results.dump_stats("seqStitching.prof")

def par():
    startTime = time.time()
    startCpu = time.process_time_ns()
    with cProfile.Profile() as profile:
        parallelStitching(params.CORES)    
    endTime = time.time()
    endCpu = time.process_time_ns()
    results = pstats.Stats(profile)
    results.sort_stats(pstats.SortKey.TIME)
    results.print_stats(10)
    results.dump_stats("parStitching.prof")
    parTime = endTime - startTime
    parTimeCpu = endCpu - startCpu
    print(str(parTime)+ " - "+ str(parTimeCpu))

def testFov():  
    for fov in [35, 40, 45, 50, 55, 60]:
        print("Run with "+str(fov)+" FOV.")
        parallelStitching(params.CORES, FOV = fov)

def testCutX():  
    for cutX in [20, 30, 40, 50, 100, 150]:
        print("Run with "+str(cutX)+" cut X.")
        parallelStitching(params.CORES, cutX = cutX)

def testCutY():  
    for cutY in [150, 200, 220, 240, 260, 280, 300]:
        print("Run with "+str(cutY)+" cut Y.")
        parallelStitching(params.CORES, cutY = cutY)

def testBlendWindow():  
    for blendWindow in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
        print("Run with "+str(blendWindow)+" blend window.")
        parallelStitching(params.CORES, blendWindow = blendWindow)

def testGaussianWindow():
    for dimWindow in [3, 5, 7, 9, 11]:
        print("Run with "+str(dimWindow)+" Gaussian window.")
        parallelStitching(params.CORES, dimWindow = dimWindow)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
    description="Image stitching."
    )
    parser.add_argument("-testCores", help="Run multiple times the parallel version: with 2, 4, 6, 8 and 12 cores.")
    parser.add_argument("-par", help="Run only the parallel.")
    parser.add_argument("-ransacThreshold", help="Run parallel version varying RANSAC threshold.")
    parser.add_argument("-testFov", help="Run parallel version varying FOV.")
    parser.add_argument("-testBlendW", help="Run parallel version varying the blend window.")
    parser.add_argument("-testGaussianW", help="Run parallel version varying the Gaussian window.")
    parser.add_argument("-testCutX", help="Run parallel version varying feature cut X edge.")
    parser.add_argument("-testCutY", help="Run parallel version varying feature cut Y edge.")
    args = vars(parser.parse_args())
    if args["testCutX"]:
        testCutX()
    elif args["testCutY"]:
        testCutY()
    elif args["testBlendW"]:
        testBlendWindow()
    elif args["testGaussianW"]:
        testGaussianWindow()
    elif args["testFov"]:
        testFov()
    elif args["ransacThreshold"]:
        startTime = time.time()
        startCpu = time.process_time_ns()
        with cProfile.Profile() as profile:
            parallelStitching(params.CORES, ransacThreshold=int(args["ransacThreshold"])) 
        endTime = time.time()
        endCpu = time.process_time_ns()
        results = pstats.Stats(profile)
        results.sort_stats(pstats.SortKey.TIME)
        results.print_stats(10)
        results.dump_stats("parStitching.prof")
        parTime = endTime - startTime
        parTimeCpu = endCpu - startCpu
        print(str(parTime)+ " - "+ str(parTimeCpu))
    elif args["par"]:
        par()
    elif args["testCores"]:
        testCores()
    else:
        seq()
    