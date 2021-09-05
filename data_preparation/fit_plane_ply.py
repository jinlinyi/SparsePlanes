import argparse
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import sys
import os
import open3d as o3d
import copy
from plyfile import PlyData, PlyElement
import json
import zipfile
import pandas as pd
from collections import defaultdict 
from glob import glob
from tqdm import tqdm

from detectron2.utils.colormap import colormap

ROOT_FOLDER = '/Pool1/users/jinlinyi/dataset/matterport3d/v1/extracted/'
OUTPUT_FOLDER = '/Pool1/users/jinlinyi/dataset/mp3d_rpnet_v3/planes_ply'



parser = argparse.ArgumentParser()
parser.register("type", "bool", lambda v: v.lower() == "true")
parser.add_argument('--base', type=int, default=0,
                    help='start scene idx.')
parser.add_argument('--num', type=int, default=1,
                    help='number of scenes to be processed.')
parser.add_argument('--id', type=int, default=0,
                    help='process id.')
args = parser.parse_args()



class ColorPalette:
    def __init__(self, numColors):
        np.random.seed(2)
        self.colorMap = np.array([[255, 0, 0],
                                  [0, 255, 0],
                                  [0, 0, 255],
                                  [80, 128, 255],
                                  [255, 230, 180],
                                  [255, 0, 255],
                                  [0, 255, 255],
                                  [100, 0, 0],
                                  [0, 100, 0],
                                  [255, 255, 0],
                                  [50, 150, 0],
                                  [200, 255, 255],
                                  [255, 200, 255],
                                  [128, 128, 80],
                                  [0, 50, 128],
                                  [0, 100, 100],
                                  [0, 255, 128],
                                  [0, 128, 255],
                                  [255, 0, 128],
                                  [128, 0, 255],
                                  [255, 128, 0],
                                  [128, 255, 0],
        ])

        if numColors > self.colorMap.shape[0]:
            self.colorMap = np.concatenate([self.colorMap, np.random.randint(255, size = (numColors - self.colorMap.shape[0], 3))], axis=0)
            pass

        return

    def getColorMap(self):
        return self.colorMap

    def getColor(self, index):
        if index >= colorMap.shape[0]:
            return np.random.randint(255, size = (3))
        else:
            return self.colorMap[index]
            pass

def writePointCloudFace(filename, points, faces):
    with open(filename, 'w') as f:
        header = """ply
format ascii 1.0
element vertex """
        header += str(len(points))
        header += """
property float x
property float y
property float z
property uchar red                                     { start of vertex color }
property uchar green
property uchar blue
element face """
        header += str(len(faces))
        header += """
property list uchar int vertex_index
end_header
"""
        f.write(header)
        for point in points:
            for value in point[:3]:
                f.write(str(value) + ' ')
                continue
            for value in point[3:]:
                f.write(str(int(value)) + ' ')
                continue
            f.write('\n')
            continue
        for face in faces:
            f.write('3 ' + str(face[0]) + ' ' + str(face[1]) + ' ' + str(face[2]) + '\n')
            continue        
        f.close()
        pass
    return

def loadClassMap():
    # raw_category -> mp40class
    classMap = {}

    with open('../metadata/category_mapping.tsv') as fd:
        rd = pd.read_csv(fd, delimiter='\t', quotechar='"')
        keys = rd['raw_category'].tolist()
        values = rd['mpcat40'].tolist()
    for key, value in zip(keys, values):
        classMap[key] = value
    return classMap

def fitPlane(points):
    if points.shape[0] == points.shape[1]:
        return np.linalg.solve(points, np.ones(points.shape[0]))
    else:
        return np.linalg.lstsq(points, np.ones(points.shape[0]), rcond=None)[0]
    
def mergePlanesNew(points, planes, planePointIndices, planeSegments, segmentNeighbors, numPlanes, planeDiffThreshold = 0.05, planeAngleThreshold = 30, inlierThreshold = 0.9, planeAreaThreshold = 10, orthogonalThreshold = np.cos(np.deg2rad(60)), parallelThreshold = np.cos(np.deg2rad(30)), debug=False):
    fittingErrorThreshold = planeDiffThreshold
    
    planeFittingErrors = []
    for plane, pointIndices in zip(planes, planePointIndices):
        XYZ = points[pointIndices]
        planeNorm = np.linalg.norm(plane)
        if planeNorm == 0:
            planeFittingErrors.append(fittingErrorThreshold)
            continue
        diff = np.abs(np.matmul(XYZ, plane) - np.ones(XYZ.shape[0])) / planeNorm
        planeFittingErrors.append(diff.mean())
        continue
    
    planeList = zip(planes, planePointIndices, planeSegments, planeFittingErrors)
    planeList = sorted(planeList, key=lambda x:x[3])

    ## Merge two planes if they are neighbors and the merged plane has small fitting error
    while len(planeList) > 0:
        hasChange = False
        planeIndex = 0

        if debug:
            for index, planeInfo in enumerate(sorted(planeList, key=lambda x:-len(x[1]))):
                print(index, planeInfo[0] / np.linalg.norm(planeInfo[0]), planeInfo[2], planeInfo[3])
        
        while planeIndex < len(planeList):
            plane, pointIndices, segments, fittingError = planeList[planeIndex]
            if fittingError > fittingErrorThreshold:
                break
            neighborSegments = []
            for segment in segments:
                if segment in segmentNeighbors:
                    neighborSegments += segmentNeighbors[segment]
            neighborSegments += list(segments)
            neighborSegments = set(neighborSegments)
            bestNeighborPlane = (fittingErrorThreshold, -1, None)
            for neighborPlaneIndex, neighborPlane in enumerate(planeList):
                if neighborPlaneIndex <= planeIndex:
                    continue
                if not bool(neighborSegments & neighborPlane[2]):
                    continue
                dotProduct = np.abs(np.dot(neighborPlane[0], plane) / np.maximum(np.linalg.norm(neighborPlane[0]) * np.linalg.norm(plane), 1e-4))
                newPointIndices = np.concatenate([neighborPlane[1], pointIndices], axis=0)
                XYZ = points[newPointIndices]
                if dotProduct > parallelThreshold and len(neighborPlane[1]) > len(pointIndices) * 0.5:
                    newPlane = fitPlane(XYZ)                    
                else:
                    newPlane = plane
                #newPlane = plane
                diff = np.abs(np.matmul(XYZ, newPlane) - np.ones(XYZ.shape[0])) / np.linalg.norm(newPlane)
                newFittingError = diff.mean()
                if debug:
                    print(len(planeList), planeIndex, neighborPlaneIndex, newFittingError, plane / np.linalg.norm(plane), neighborPlane[0] / np.linalg.norm(neighborPlane[0]), dotProduct, orthogonalThreshold)
                if dotProduct < orthogonalThreshold:
                    continue                
                if newFittingError < bestNeighborPlane[0]:
                    newPlaneInfo = [newPlane, newPointIndices, segments.union(neighborPlane[2]), newFittingError]
                    bestNeighborPlane = (newFittingError, neighborPlaneIndex, newPlaneInfo)
                continue
            if bestNeighborPlane[1] != -1:
                newPlaneList = planeList[:planeIndex] + planeList[planeIndex + 1:bestNeighborPlane[1]] + planeList[bestNeighborPlane[1] + 1:]
                newFittingError, newPlaneIndex, newPlane = bestNeighborPlane
                for newPlaneIndex in range(len(newPlaneList)):
                    if (newPlaneIndex == 0 and newPlaneList[newPlaneIndex][3] > newFittingError) \
                       or newPlaneIndex == len(newPlaneList) - 1 \
                       or (newPlaneList[newPlaneIndex][3] < newFittingError and newPlaneList[newPlaneIndex + 1][3] > newFittingError):
                        newPlaneList.insert(newPlaneIndex, newPlane)
                        break                    
                    continue
                if len(newPlaneList) == 0:
                    newPlaneList = [newPlane]
                    pass
                planeList = newPlaneList
                hasChange = True
            else:
                planeIndex += 1
                pass
            continue
        if not hasChange:
            break
        continue

    planeList = sorted(planeList, key=lambda x:-len(x[1]))

    
    minNumPlanes, maxNumPlanes = numPlanes
    if minNumPlanes == 1 and len(planeList) == 0:
        if debug:
            print('at least one plane')
    elif len(planeList) > maxNumPlanes:
        if debug:
            print('too many planes', len(planeList), maxNumPlanes)
        planeList = planeList[:maxNumPlanes]
    groupedPlanes, groupedPlanePointIndices, groupedPlaneSegments, groupedPlaneFittingErrors = zip(*planeList)
    groupNeighbors = []
    for planeIndex, planeSegments in enumerate(groupedPlaneSegments):
        neighborSegments = []
        for segment in planeSegments:
            if segment in segmentNeighbors:            
                neighborSegments += segmentNeighbors[segment]
        neighborSegments += list(planeSegments)        
        neighborSegments = set(neighborSegments)
        neighborPlaneIndices = []
        for neighborPlaneIndex, neighborPlaneSegments in enumerate(groupedPlaneSegments):
            if neighborPlaneIndex == planeIndex:
                continue
            if bool(neighborSegments & neighborPlaneSegments):
                plane = groupedPlanes[planeIndex]
                neighborPlane = groupedPlanes[neighborPlaneIndex]
                if np.linalg.norm(plane) * np.linalg.norm(neighborPlane) < 1e-4:
                    continue
                dotProduct = np.abs(np.dot(plane, neighborPlane) / np.maximum(np.linalg.norm(plane) * np.linalg.norm(neighborPlane), 1e-4))
                if dotProduct < orthogonalThreshold:
                    neighborPlaneIndices.append(neighborPlaneIndex)
        groupNeighbors.append(neighborPlaneIndices)

    if debug and len(groupedPlanes) > 1:
        print('merging result', [len(pointIndices) for pointIndices in groupedPlanePointIndices], groupedPlaneFittingErrors, groupNeighbors)
    # planeList = zip(groupedPlanes, groupedPlanePointIndices, groupNeighbors)
    assert(len(groupedPlanes) == len(groupedPlanePointIndices) and len(groupedPlanes) == len(groupNeighbors))
    planeDict = {
        'groupedPlanes': groupedPlanes,
        'groupedPlanePointIndices': groupedPlanePointIndices, 
        'groupNeighbors': groupNeighbors,
    }
    return planeDict


def fix_JSON(json_message=None):
    result = None
    try:
        result = json.loads(json_message)
    except Exception as e:
        # Find the offending character index:
        json_message = json_message.replace('\\','')
        result = json.loads(json_message)
    return result


def mp3d2habitat(planes):
    rotation = np.array([
        [1,0,0],
        [0,0,-1],
        [0,1,0],
    ])
    rotation = np.linalg.inv(rotation)
    return (rotation@np.array(planes).T).T


def readMesh(house_folder, region_id):
    # Load open3d mesh
    mesh_o3d = o3d.io.read_triangle_mesh(os.path.join(house_folder, region_id+'.ply'))

    # Load map: instance -> [segment id], store in aggregation
    with open(os.path.join(house_folder, region_id+'.semseg.json'), 'r') as f:
        json_string=f.read().replace('\n', '')
    data = fix_JSON(json_string)
    aggregation = np.array(data['segGroups'])

    # Load mesh: pcd, faces
    plydata = PlyData.read(os.path.join(house_folder, region_id+'.ply'))
    vertices = plydata['vertex']
    points = np.stack([vertices['x'], vertices['y'], vertices['z']], axis=1)
    faces = np.array(plydata['face']['vertex_indices'])

    # Load map: vertex -> segment id, store in segmentation
    data = json.load(open(os.path.join(house_folder, region_id+'.vsegs.json'), 'r'))
    segmentation = np.array(data['segIndices']).astype(np.int32)

    # Extract information from aggretation
    # groupSegments -> [segment ids]
    # groupLabels -> category for segment group
    groupSegments = []
    groupLabels = []
    for segmentIndex, item in enumerate(aggregation):
        assert item['id'] == segmentIndex
        groupSegments.append(item['segments'])
        groupLabels.append(item['label'])

    uniqueSegments = np.unique(segmentation).tolist()
    for segments in groupSegments:
        for segmentIndex in segments:
            if segmentIndex in uniqueSegments:
                uniqueSegments.remove(segmentIndex)

    for segment in uniqueSegments:
        groupSegments.append([segment, ])
        groupLabels.append('unannotated')

    numGroups = len(groupSegments)
    numPoints = segmentation.shape[0]    
    numPlanes = 1000

    ## Segment connections for plane merging later
    ## e.g. segmentEdges = [(1100, 2689), (4, 1991), (1372, 2238), (390, 2780), (1706, 2022), (207, 3073), (2552, 2841), (1802, 2817), (1117, 2798), (3142, 3463), (999, 1894), (891, 1657), (13, 1905), (2596, 2999), (1340, 1550), (2675, 3723)]
    ## stores tuples of segment id.
    segmentEdges = []
    for face in faces:
        segment_1 = segmentation[face[0]]
        segment_2 = segmentation[face[1]]
        segment_3 = segmentation[face[2]]
        if segment_1 != segment_2 or segment_1 != segment_3:
            if segment_1 != segment_2 and segment_1 != -1 and segment_2 != -1:
                segmentEdges.append((min(segment_1, segment_2), max(segment_1, segment_2)))
            if segment_1 != segment_3 and segment_1 != -1 and segment_3 != -1:
                segmentEdges.append((min(segment_1, segment_3), max(segment_1, segment_3)))
            if segment_2 != segment_3 and segment_2 != -1 and segment_3 != -1:
                segmentEdges.append((min(segment_2, segment_3), max(segment_2, segment_3)))
    segmentEdges = list(set(segmentEdges))


    numPlanes = 200
    numPlanesPerSegment = 2
    segmentRatio = 0.1
    planeAreaThreshold = 10
    numIterations = 100
    numIterationsPair = 1000
    planeDiffThreshold = 0.05
    fittingErrorThreshold = planeDiffThreshold

    ## Specify the minimum and maximum number of planes for each object
    labelNumPlanes = {'wall': [1, 20], 
                      'floor': [1, 1],
                      'cabinet': [1, 5],
                      'bed': [1, 5],
                      'chair': [1, 4],
                      'sofa': [1, 10],
                      'table': [1, 5],
                      'door': [1, 2],
                      'window': [1, 2],
                    #   'bookshelf': [1, 5],
                      'picture': [1, 1],
                      'counter': [1, 10],
                      'blinds': [0, 0],
                    #   'desk': [1, 10],
                    #   'shelf': [1, 5],
                    #   'shelves': [1, 5],                      
                      'curtain': [0, 0],
                    #   'dresser': [1, 5],
                      'cushion': [0, 0], # 'pillow': [0, 0],
                      'mirror': [0, 0],
                    #   'entrance': [1, 1],
                    #   'floor mat': [1, 1],                      
                      'clothes': [0, 0],
                      'ceiling': [1, 5],
                    #   'book': [0, 1],
                    #   'books': [0, 1],                      
                    #   'refridgerator': [1, 5],
                      'tv_monitor': [1, 1], # 'television': [1, 1], 
                    #   'paper': [0, 1],
                      'towel': [0, 1],
                    #   'shower curtain': [0, 1],
                    #   'box': [1, 5],
                    #   'whiteboard': [1, 5],
                    #   'person': [0, 0],
                    #   'night stand': [1, 5],
                      'toilet': [0, 5],
                      'sink': [0, 5],
                    #   'lamp': [0, 1],
                      'bathtub': [0, 5],
                    #   'bag': [0, 1],
                      'misc': [0, 5], # mp3d
                    #   'otherprop': [0, 5],
                    #   'otherstructure': [0, 5],
                      'furniture': [0, 5], #   'otherfurniture': [0, 5], 
                      'appliances': [0, 5],                     
                      'unannotated': [0, 5],
                      'void': [0, 0],
                      'chest_of_drawers': [0, 5],
                      'stairs': [0, 20],
                      'stool': [0, 2],
                      'shower': [0, 5],
                      'column': [0, 4],
                      'fireplace': [0, 5],
                      'lighting': [0, 2],
                      'beam': [0, 3],
                      'railing': [0, 5],
                      'shelving': [1, 5],
                      'gym_equipment': [0, 5],
                      'seating': [1, 2],
                      'board_panel': [0, 1],
                      'objects': [0, 5],
                      'unlabeled': [0, 5],
                      'plant': [0, 0],

    }
    nonPlanarGroupLabels = ['bicycle', 'bottle', 'water bottle', 'plant']
    nonPlanarGroupLabels = {label: True for label in nonPlanarGroupLabels}
    
    verticalLabels = ['wall', 'door', 'cabinet']
    classMap = loadClassMap()
    allXYZ = points.reshape(-1, 3)

    segmentNeighbors = defaultdict(list)
    for segmentEdge in segmentEdges:
        segmentNeighbors[segmentEdge[0]].append(segmentEdge[1])
        segmentNeighbors[segmentEdge[1]].append(segmentEdge[0]) 

    planeGroups = []

    debug = False    
    debugIndex = -1

    ## A group corresponds to an instance in the ScanNet annotation
    for groupIndex, group in enumerate(groupSegments):
        if debugIndex != -1 and groupIndex != debugIndex:
            continue
        if groupLabels[groupIndex] in nonPlanarGroupLabels:
            groupLabel = groupLabels[groupIndex]
            minNumPlanes, maxNumPlanes = 0, 0
        elif groupLabels[groupIndex] == 'unannotated':
            groupLabel = 'unannotated'
            minNumPlanes, maxNumPlanes = labelNumPlanes[groupLabel]
        elif groupLabels[groupIndex] in classMap:
            groupLabel = classMap[groupLabels[groupIndex]]
            minNumPlanes, maxNumPlanes = labelNumPlanes[groupLabel]            
        else:
            print(groupLabels[groupIndex] + ' not considered')
            minNumPlanes, maxNumPlanes = 0, 0
            groupLabel = ''

        if maxNumPlanes == 0:
            pointMasks = []
            for segmentIndex in group:
                pointMasks.append(segmentation == segmentIndex)
            pointIndices = np.any(np.stack(pointMasks, 0), 0).nonzero()[0]
            groupPlanes = {
                    'groupedPlanes': [np.zeros(3)],
                    'groupedPlanePointIndices': [pointIndices], 
                    'groupNeighbors': [],
                }
            planeGroups.append(groupPlanes)
            continue
        groupPlanes = []
        groupPlanePointIndices = []
        groupPlaneSegments = []


        ## A group contains multiple segments and we run RANSAC for each segment
        for segmentIndex in group:
            segmentMask = segmentation == segmentIndex
            segmentIndices = segmentMask.nonzero()[0]

            XYZ = allXYZ[segmentMask.reshape(-1)]
            if len(XYZ) == 0:
                continue
            numPoints = XYZ.shape[0]

            segmentPlanes = []
            segmentPlanePointIndices = []

            for c in range(2):
                if c == 0:
                    ## First try to fit one plane to see if the entire segment is one plane
                    try:
                        plane = fitPlane(XYZ)
                    except:
                        continue
                    diff = np.abs(np.matmul(XYZ, plane) - np.ones(XYZ.shape[0])) / np.linalg.norm(plane)                        
                    if diff.mean() < fittingErrorThreshold:
                        segmentPlanes.append(plane)
                        segmentPlanePointIndices.append(segmentIndices)
                        break
                else:
                    ## Run ransac                    
                    for planeIndex in range(numPlanesPerSegment):
                        if len(XYZ) < planeAreaThreshold:
                            continue
                        bestPlaneInfo = [None, 0, None]
                        for iteration in range(min(XYZ.shape[0], numIterations)):
                            sampledPoints = XYZ[np.random.choice(np.arange(XYZ.shape[0]), size=(3), replace=False)]
                            try:
                                plane = fitPlane(sampledPoints)
                            except:
                                continue
                            diff = np.abs(np.matmul(XYZ, plane) - np.ones(XYZ.shape[0])) / np.linalg.norm(plane)
                            inlierMask = diff < planeDiffThreshold
                            numInliers = inlierMask.sum()
                            if numInliers > bestPlaneInfo[1]:
                                bestPlaneInfo = [plane, numInliers, inlierMask]

                        if bestPlaneInfo[1] < planeAreaThreshold:
                            break

                        
                        pointIndices = segmentIndices[bestPlaneInfo[2]]
                        bestPlane = fitPlane(XYZ[bestPlaneInfo[2]])
                        
                        segmentPlanes.append(bestPlane)                
                        segmentPlanePointIndices.append(pointIndices)

                        outlierMask = np.logical_not(bestPlaneInfo[2])
                        segmentIndices = segmentIndices[outlierMask]
                        XYZ = XYZ[outlierMask]
                        continue
                continue
            
            if sum([len(indices) for indices in segmentPlanePointIndices]) < numPoints * 0.5:
                print('not enough fitted points')
                if len(segmentIndices) >= planeAreaThreshold:
                    groupPlanes.append(np.zeros(3))
                    groupPlanePointIndices.append(segmentIndices)
                    groupPlaneSegments.append(set([segmentIndex]))
            else:
                groupPlanes += segmentPlanes
                groupPlanePointIndices += segmentPlanePointIndices
                for _ in range(len(segmentPlanes)):
                    groupPlaneSegments.append(set([segmentIndex]))
            continue
        
            
        if len(groupPlanes) > 0:
            ## Merge planes of each instance
            groupPlanes = mergePlanesNew(points, groupPlanes, groupPlanePointIndices, groupPlaneSegments, segmentNeighbors, numPlanes=(minNumPlanes, maxNumPlanes), planeDiffThreshold=planeDiffThreshold, planeAreaThreshold=planeAreaThreshold, debug=debugIndex != -1)

        planeGroups.append(groupPlanes)
    
    
    if debug:
        colorMap = ColorPalette(segmentation.max() + 2).getColorMap()
        colorMap[-1] = 0
        colorMap[-2] = 255
        annotationFolder = 'test/'
    else:
        numPlanes = sum([len(group['groupedPlanes']) for group in planeGroups])
        segmentationColor = (np.arange(numPlanes) + 1) * 100
        colorMap = np.stack([segmentationColor / (256 * 256), segmentationColor / 256 % 256, segmentationColor % 256], axis=1)
        colorMap[-1] = 255
        annotationFolder = os.path.join(OUTPUT_FOLDER, scene_id, 'plane_annotation')
    os.makedirs(annotationFolder, exist_ok=True)


    if debug:
        colors = colorMap[segmentation]
        writePointCloudFace(annotationFolder + '/segments.ply', np.concatenate([points, colors], axis=-1), faces)

        groupedSegmentation = np.full(segmentation.shape, fill_value=-1)
        for segmentIndex in range(len(aggregation)):
            indices = aggregation[segmentIndex]['segments']
            for index in indices:
                groupedSegmentation[segmentation == index] = segmentIndex
                continue
            continue
        groupedSegmentation = groupedSegmentation.astype(np.int32)
        colors = colorMap[groupedSegmentation]
        writePointCloudFace(annotationFolder + '/groups.ply', np.concatenate([points, colors], axis=-1), faces)

    planes = []
    planePointIndices = []
    for index, group in enumerate(planeGroups):
        planes.extend(group['groupedPlanes'])
        planePointIndices.extend(group['groupedPlanePointIndices'])
    
    # Filter out non-planar regions, planeSegmentation consists of all plane id, -1 means non-planar. filtered_plane[plane_id] -> plane params
    filtered_planes = []
    planeSegmentation = np.full(segmentation.shape, fill_value=-1, dtype=np.int32)
    tmp_idx = 0
    for planeIndex, (planePoints, plane_param) in enumerate(zip(planePointIndices, planes)):
        if np.linalg.norm(planes[planeIndex]) < 1e-4:
            pass
        else:
            planeSegmentation[planePoints] = tmp_idx
            filtered_planes.append(plane_param)
            tmp_idx += 1
    assert(len(filtered_planes) == tmp_idx)
    planes = filtered_planes

    if debug:
        groupSegmentation = np.full(segmentation.shape, fill_value=-1, dtype=np.int32)        
        structureSegmentation = np.full(segmentation.shape, fill_value=-1, dtype=np.int32)
        typeSegmentation = np.full(segmentation.shape, fill_value=-1, dtype=np.int32)
        for planeIndex, planePoints in enumerate(planePointIndices):
            if len(planeInfo[planeIndex]) > 1:
                structureSegmentation[planePoints] = planeInfo[planeIndex][1][0]
                typeSegmentation[planePoints] = np.maximum(typeSegmentation[planePoints], planeInfo[planeIndex][1][1] - 2)
                pass
            groupSegmentation[planePoints] = planeInfo[planeIndex][0][0]
            continue

        colors = colorMap[groupSegmentation]    
        writePointCloudFace(annotationFolder + '/group.ply', np.concatenate([points, colors], axis=-1), faces)

        colors = colorMap[structureSegmentation]    
        writePointCloudFace(annotationFolder + '/structure.ply', np.concatenate([points, colors], axis=-1), faces)

        colors = colorMap[typeSegmentation]    
        writePointCloudFace(annotationFolder + '/type.ply', np.concatenate([points, colors], axis=-1), faces)
        pass


    planes = np.array(planes)
    planesD = 1.0 / np.maximum(np.linalg.norm(planes, axis=-1, keepdims=True), 1e-4)
    planes *= pow(planesD, 2)

    # mp3d to ai habitat
    planes = mp3d2habitat(planes)

    ## Remove boundary faces for rendering purpose
    removeIndices = []
    for faceIndex in range(faces.shape[0]):
        face = faces[faceIndex]
        segment_1 = planeSegmentation[face[0]]
        segment_2 = planeSegmentation[face[1]]
        segment_3 = planeSegmentation[face[2]]
        if segment_1 != segment_2 or segment_1 != segment_3:
            removeIndices.append(faceIndex)
            pass
        continue
    faces = np.delete(faces, removeIndices)
    colors = colorMap[planeSegmentation] 
    cmap = colormap() / 255  
    new_color_plane = cmap[planeSegmentation%len(cmap)]  
    new_color_plane[planeSegmentation==-2] = [1,1,1] 
    new_color_plane[planeSegmentation==-1] = [1,1,1]
    mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(new_color_plane)
    o3d.io.write_triangle_mesh(os.path.join(annotationFolder, region_id+".gltf"), mesh_o3d)
    np.save(os.path.join(annotationFolder, region_id+"_planeid.npy"), planeSegmentation)

    if debug:
        exit(1)
        pass
    
    np.save(os.path.join(annotationFolder, region_id+".npy"), planes)


def check_cls(mpcat40_path):
    with open(mpcat40_path) as fd:
        rd = pd.read_csv(fd, delimiter='\t', quotechar='"')
    mpcat = rd['mpcat40'].tolist()

    labelNumPlanes = {'wall': [1, 3], 
                      'floor': [1, 1],
                      'cabinet': [1, 5],
                      'bed': [1, 5],
                      'chair': [1, 2],
                      'sofa': [1, 10],
                      'table': [1, 5],
                      'door': [1, 2],
                      'window': [1, 2],
                    #   'bookshelf': [1, 5],
                      'picture': [1, 1],
                      'counter': [1, 10],
                      'blinds': [0, 0],
                    #   'desk': [1, 10],
                    #   'shelf': [1, 5],
                    #   'shelves': [1, 5],                      
                      'curtain': [0, 0],
                    #   'dresser': [1, 5],
                      'cushion': [0, 0], # 'pillow': [0, 0],
                      'mirror': [0, 0],
                    #   'entrance': [1, 1],
                    #   'floor mat': [1, 1],                      
                      'clothes': [0, 0],
                      'ceiling': [1, 5],
                    #   'book': [0, 1],
                    #   'books': [0, 1],                      
                    #   'refridgerator': [1, 5],
                      'tv_monitor': [1, 1], # 'television': [1, 1], 
                    #   'paper': [0, 1],
                      'towel': [0, 1],
                    #   'shower curtain': [0, 1],
                    #   'box': [1, 5],
                    #   'whiteboard': [1, 5],
                    #   'person': [0, 0],
                    #   'night stand': [1, 5],
                      'toilet': [0, 5],
                      'sink': [0, 5],
                    #   'lamp': [0, 1],
                      'bathtub': [0, 5],
                    #   'bag': [0, 1],
                      'misc': [0, 5], # mp3d
                    #   'otherprop': [0, 5],
                    #   'otherstructure': [0, 5],
                      'furniture': [0, 5], #   'otherfurniture': [0, 5], 
                      'appliances': [0, 5],                     
                      'unannotated': [0, 5],
                      'void': [0, 0],
                      'chest_of_drawers': [0, 5],
                      'stairs': [0, 10],
                      'stool': [0, 2],
                      'shower': [0, 5],
                      'column': [0, 4],
                      'fireplace': [0, 5],
                      'lighting': [0, 5],
                      'beam': [0, 3],
                      'railing': [0, 5],
                      'shelving': [1, 5],
                      'gym_equipment': [0, 5],
                      'seating': [1, 2],
                      'board_panel': [0, 1],
                      'objects': [0, 5],
                      'unlabeled': [0, 5],
                      'plant': [0, 0],

    }
    nonPlanarGroupLabels = ['bicycle', 'bottle', 'water bottle', 'plant']
    nonPlanarGroupLabels = {label: True for label in nonPlanarGroupLabels}
    
    # check labelNumPlanes
    for item in labelNumPlanes.keys():
        if item not in mpcat:
            print(item + " not in mpcat")
    print('----\n')
    for item in mpcat:
        if item not in labelNumPlanes.keys() and item not in nonPlanarGroupLabels:
            print(item)


if __name__=='__main__':
    np.random.seed(2020)
    scene_ids = sorted(os.listdir(ROOT_FOLDER))
    try: 
        scene_ids.remove('sens')
    except:
        pass
    scene_ids = scene_ids[args.base:args.base+args.num]
    if not (len(scene_ids) == args.num):
        print(args)
        exit()

    for scene_id in tqdm(scene_ids, desc=f"process_id {args.id}"):
        print('plane fitting', scene_id)
        house_folder = os.path.join(ROOT_FOLDER, scene_id, 'region_segmentations')
        regions = sorted(list(glob(os.path.join(house_folder, 'region*.ply'))))
        region_ids = [region.split('/')[-1].split('.')[0] for region in regions]
        for region_id in region_ids:
            readMesh(house_folder, region_id)