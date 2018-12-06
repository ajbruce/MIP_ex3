
import nibabel as nib
import numpy as np
from skimage import measure,morphology
import os
import matplotlib.pyplot as plt
from medpy import metric

from skimage.transform import hough_circle, hough_circle_peaks
from skimage.draw import circle
from skimage.filters import sobel

N = 200

def segmentLiver(ctFileName, AortaFileName, outputFileName):
    ct_data = nib.load(ctFileName).get_data()
    aorta_data = nib.load(AortaFileName).get_data()

    body_segmentation = IsolateBody(ct_data)
    # liver_region_nifti = nib.Nifti1Image(body_segmentation, np.eye(4))
    # nib.save(liver_region_nifti, "body_seg.nii.gz")


    roi = IsolateROI(body_segmentation, aorta_data)

    # liver_region_nifti = nib.Nifti1Image(roi, np.eye(4))
    # nib.save(liver_region_nifti, "liver_roi.nii.gz")


    liver_region = multipleSeedsRG(ct_data, roi)

    liver_region_nifti = nib.Nifti1Image(liver_region, np.eye(4))
    nib.save(liver_region_nifti, outputFileName)


def IsolateBody(ct_data):
    new_img = np.ones((ct_data.shape))
    new_img [ct_data < -500] = 0
    new_img [ct_data > 2000] = 0

    new_img = morphology.binary_opening(new_img)

    labels, _ncomponents = measure.label(new_img, return_num=True)
    labels = remove_small_component(labels, 70)

    new_img[labels == 0] = 0
    new_img[labels > 0] = 1

    labels, _ncomponents = measure.label(new_img, return_num=True)
    largest_component = get_largest_component(labels)

    body_segmentation = np.zeros(shape=(new_img.shape))
    body_segmentation[labels == largest_component] = 1

    return body_segmentation


def findSeeds(ct_img_data, roi_img_data):

    roi_img_data[roi_img_data > 0] = ct_img_data[roi_img_data > 0]

    low_threshold = -100
    high_threshold = 200
    idx = (roi_img_data>low_threshold)*(roi_img_data<high_threshold)

    all_seeds = np.argwhere(idx)

    random_indices = np.random.randint(low=0, high=all_seeds.shape[0]-1, size=N)
    segmentation = np.zeros(shape = ct_img_data.shape, dtype=int)
    initial_seeds = []
    for index in random_indices:
        x,y,z = all_seeds[index][0], all_seeds[index][1], all_seeds[index][2]
        segmentation[x,y,z] = 1
        initial_seeds.append((x,y,z))


    liver_region_nifti = nib.Nifti1Image(segmentation, np.eye(4))
    nib.save(liver_region_nifti, "initial_seeds.nii.gz")


    return initial_seeds

def multipleSeedsRG(ct_img_data, roi_img_data):

    initial_seeds = findSeeds(ct_img_data, roi_img_data)
    labeled_voxels = label_voxels(initial_seeds, ct_img_data)
    liver_seg = np.zeros(shape=roi_img_data.shape)
    counter = 0
    segmentation = np.zeros(shape = ct_img_data.shape)
    region_means = [None]*N
    region_sizes = [1]*N
    neighbors = [None]*N

    for i in range(N):
        segmentation[initial_seeds[i]] = 1
        region_means[i] = ct_img_data[initial_seeds[i]]
        neighbors[i] = get_neighbors(segmentation.shape, initial_seeds[i], labeled_voxels)
    
    while (len(neighbors)):

        for i in range(N):
            if(~len(neighbors[i])): continue
            current_neighbor = neighbors[i].pop()
            if(labeled_voxels[current_neighbor] != True):
                labeled_voxels[current_neighbor] = True

                if (abs(ct_img_data[current_neighbor] - (region_means[i] / region_sizes[i])) < 25):
                    segmentation[current_neighbor] = 1
                    region_means[i] += (ct_img_data[current_neighbor])
                    region_sizes[i] += 1
                    neighbors[i] += get_neighbors(segmentation.shape, current_neighbor, labeled_voxels)

    liver_seg[segmentation > 0] = 1
    liver_seg = morphology.binary_dilation(liver_seg, selem=np.ones((3,3,3)))

def evaluateSegmentation(gt_segmentation, es_segmentation):
    # VOD
    vol_intersection = np.zeros(shape=es_segmentation.shape)
    vol_intersection = np.count_nonzero(np.multiply(gt_segmentation, es_segmentation))

    union = np.zeros(shape=es_segmentation.shape)
    union[gt_segmentation > 0] = 1
    union[es_segmentation > 0] = 1

    vol_union = np.count_nonzero(union)

    vod = 100*(1 - vol_intersection/vol_union)

    # DICE
    gt_vol = np.count_nonzero(gt_segmentation)
    es_vol = np.count_nonzero(es_segmentation)

    dice = 2 * vol_union / (gt_vol+es_vol)

    # averagesymmetric surface distance
    assd = metric.binary.assd(es_segmentation, gt_segmentation, voxelspacing=vxlspacing)

    return (vod,dice, assd)

def IsolateROI(ct_data, aorta_data):

    slices_array = np.where(aorta_data > 0)

    min_x = np.min(slices_array[0])
    max_x = np.max(slices_array[0])

    min_slice = np.min(slices_array[2])
    max_slice = np.max(slices_array[2])
    roi = np.zeros((ct_data.shape))
    roi[max_x:,:,min_slice:max_slice] = ct_data[max_x:,:,150:300]

    return roi


def get_neighbors(data_shape, current_voxel, labeled):

    neighbors_indices_list = []

    #x-1,y-1,z-1
    if (current_voxel[0] > 0) and (current_voxel[1] > 0) and (current_voxel[2] > 0)\
            and not labeled[current_voxel[0] - 1, current_voxel[1] - 1, current_voxel[2] - 1]:
        neighbors_indices_list.append((current_voxel[0] - 1, current_voxel[1] - 1, current_voxel[2] - 1))
    #x-1,y-1, z
    if (current_voxel[0] > 0) and (current_voxel[1] > 0) \
            and not labeled[current_voxel[0] - 1, current_voxel[1] - 1, current_voxel[2]]:
        neighbors_indices_list.append((current_voxel[0] - 1, current_voxel[1] - 1, current_voxel[2]))
    #x-1,y,z
    if (current_voxel[0] > 0) and not labeled[current_voxel[0] - 1, current_voxel[1], current_voxel[2]]:
        neighbors_indices_list.append((current_voxel[0] - 1, current_voxel[1], current_voxel[2]))
    #x-1,y,z-1
    if (current_voxel[0] > 0) and (current_voxel[2] > 0) \
            and not labeled[current_voxel[0] - 1, current_voxel[1], current_voxel[2] - 1]:
        neighbors_indices_list.append((current_voxel[0] - 1, current_voxel[1], current_voxel[2] - 1))
    #x-1, y-1, z+1
    if (current_voxel[0] > 0) and (current_voxel[1] > 0) and (current_voxel[2] + 1 < data_shape[2] ) \
            and not labeled[current_voxel[0] - 1, current_voxel[1] - 1, current_voxel[2] + 1]:
        neighbors_indices_list.append((current_voxel[0] - 1, current_voxel[1] - 1, current_voxel[2] + 1))
    #x-1, y, z+1
    if (current_voxel[0] > 0) and (current_voxel[2] + 1 < data_shape[2] ) \
            and not labeled[current_voxel[0] - 1, current_voxel[1], current_voxel[2] + 1]:
        neighbors_indices_list.append((current_voxel[0] - 1, current_voxel[1], current_voxel[2] + 1))
    #x-1, y+1, z
    if (current_voxel[0] > 0) and (current_voxel[1] + 1 < data_shape[1] )  \
            and not labeled[current_voxel[0] - 1, current_voxel[1] + 1, current_voxel[2]]:
        neighbors_indices_list.append((current_voxel[0] - 1, current_voxel[1] + 1, current_voxel[2]))
    #x-1, y+1, z-1
    if (current_voxel[0] > 0) and (current_voxel[1] + 1 < data_shape[1]) and (current_voxel[2] > 0) \
            and not labeled[current_voxel[0] - 1, current_voxel[1] + 1, current_voxel[2] - 1]:
        neighbors_indices_list.append((current_voxel[0] - 1, current_voxel[1] + 1, current_voxel[2] - 1))
    #x-1, y+1, z+1
    if (current_voxel[0] > 0) and (current_voxel[1] + 1 < data_shape[1]) and (current_voxel[2] + 1 < data_shape[2]) \
            and not labeled[current_voxel[0] - 1, current_voxel[1] + 1, current_voxel[2] + 1]:
        neighbors_indices_list.append((current_voxel[0] - 1, current_voxel[1] + 1, current_voxel[2] + 1))

    #x,y,z-1
    if (current_voxel[2] > 0) and not labeled[current_voxel[0], current_voxel[1], current_voxel[2] -1]:
        neighbors_indices_list.append((current_voxel[0], current_voxel[1], current_voxel[2] -1))
    #x,y,z+1
    if (current_voxel[2] + 1 < data_shape[2]) and not labeled[current_voxel[0], current_voxel[1], current_voxel[2] + 1]:
        neighbors_indices_list.append((current_voxel[0], current_voxel[1], current_voxel[2] + 1))

    #x,y-1,z-1
    if (current_voxel[1] > 0) and (current_voxel[2] > 0) \
            and not labeled[current_voxel[0], current_voxel[1] - 1, current_voxel[2] - 1]:
        neighbors_indices_list.append((current_voxel[0], current_voxel[1] - 1, current_voxel[2] - 1))
    #x,y-1,z+1
    if (current_voxel[1] > 0) and (current_voxel[2] +  1 < data_shape[2]) \
            and not labeled[current_voxel[0], current_voxel[1] - 1, current_voxel[2] + 1]:
        neighbors_indices_list.append((current_voxel[0], current_voxel[1] - 1, current_voxel[2] + 1))


    #x,y-1,z
    if (current_voxel[1] > 0) and not labeled[current_voxel[0], current_voxel[1] - 1, current_voxel[2]]:
        neighbors_indices_list.append((current_voxel[0], current_voxel[1] - 1, current_voxel[2]))

    #x,y+1,z-1
    if (current_voxel[2] > 0) and (current_voxel[1] +  1 < data_shape[1]) \
            and not labeled[current_voxel[0], current_voxel[1] + 1, current_voxel[2] - 1]:
        neighbors_indices_list.append((current_voxel[0], current_voxel[1] + 1, current_voxel[2] - 1))

    #x,y+1,z
    if (current_voxel[1] + 1 < data_shape[1]) and not labeled[current_voxel[0], current_voxel[1] + 1, current_voxel[2]]:
        neighbors_indices_list.append((current_voxel[0], current_voxel[1] + 1, current_voxel[2]))

    #x,y+1,z+1
    if (current_voxel[1] + 1 < data_shape[1]) and (current_voxel[2] + 1 < data_shape[2])\
            and not labeled[current_voxel[0], current_voxel[1] + 1, current_voxel[2] + 1]:
        neighbors_indices_list.append((current_voxel[0], current_voxel[1] + 1, current_voxel[2] + 1))
    #x+1, y,z
    if (current_voxel[0] + 1 < data_shape[0]) and not labeled[current_voxel[0] + 1, current_voxel[1], current_voxel[2]]:
        neighbors_indices_list.append((current_voxel[0] + 1, current_voxel[1], current_voxel[2]))
    #x+1, y,z+1
    if (current_voxel[0] + 1 < data_shape[0]) and (current_voxel[2] + 1 < data_shape[2])\
            and not labeled[current_voxel[0] + 1, current_voxel[1], current_voxel[2] + 1]:
        neighbors_indices_list.append((current_voxel[0] + 1, current_voxel[1], current_voxel[2] + 1))

    #x+1, y,z-1
    if (current_voxel[0] + 1 < data_shape[0]) and (current_voxel[2] > 0 )\
            and not labeled[current_voxel[0] + 1, current_voxel[1], current_voxel[2] - 1]:
        neighbors_indices_list.append((current_voxel[0] + 1, current_voxel[1], current_voxel[2] - 1))
    #x+1, y-1,z
    if (current_voxel[0] + 1 < data_shape[0]) and (current_voxel[1] > 0 )\
            and not labeled[current_voxel[0] + 1, current_voxel[1] - 1, current_voxel[2]]:
        neighbors_indices_list.append((current_voxel[0] + 1, current_voxel[1] - 1, current_voxel[2]))

    #x+1, y-1,z-1
    if (current_voxel[0] + 1 < data_shape[0]) and (current_voxel[1] > 0) and (current_voxel[2] > 0 )\
            and not labeled[current_voxel[0] + 1, current_voxel[1] - 1, current_voxel[2] - 1]:
        neighbors_indices_list.append((current_voxel[0] + 1, current_voxel[1] - 1, current_voxel[2] - 1))
    #x+1, y-1,z+1
    if (current_voxel[0] + 1 < data_shape[0]) and (current_voxel[1] > 0) and (current_voxel[2] + 1 < data_shape[2])\
            and not labeled[current_voxel[0] + 1, current_voxel[1] - 1, current_voxel[2] + 1]:
        neighbors_indices_list.append((current_voxel[0] + 1, current_voxel[1] - 1, current_voxel[2] + 1))

    #x+1, y+1,z
    if (current_voxel[0] + 1 < data_shape[0]) and (current_voxel[1] + 1 < data_shape[1])\
            and not labeled[current_voxel[0] + 1, current_voxel[1] + 1, current_voxel[2]]:
        neighbors_indices_list.append((current_voxel[0] + 1, current_voxel[1] + 1, current_voxel[2]))
    #x+1, y+1,z-1
    if (current_voxel[0] + 1 < data_shape[0]) and (current_voxel[1] + 1 < data_shape[1]) and (current_voxel[2] > 0)\
            and not labeled[current_voxel[0] + 1, current_voxel[1] + 1, current_voxel[2] - 1]:
        neighbors_indices_list.append((current_voxel[0] + 1, current_voxel[1] + 1, current_voxel[2] -1))
    #x+1, y+1,z+1
    if (current_voxel[0] + 1 < data_shape[0]) and (current_voxel[1] + 1 < data_shape[1]) and (current_voxel[2] + 1 > data_shape[2])\
            and not labeled[current_voxel[0] + 1, current_voxel[1] + 1, current_voxel[2] + 1]:
        neighbors_indices_list.append((current_voxel[0] + 1, current_voxel[1] + 1, current_voxel[2] + 1))

    return neighbors_indices_list


def label_voxels(initial_seeds, ct_img_data):
    labeled_voxels = np.zeros(shape=ct_img_data.shape, dtype=bool)
    for i in range(N):
        labeled_voxels[initial_seeds[i]] = True
    return labeled_voxels


def get_largest_component(labels):
    component_area = 0
    largest_component = 0
    for label_prop in measure.regionprops(labels):
        current_label_area = label_prop.area
        if (current_label_area > component_area):
           component_area =  current_label_area
           largest_component = label_prop.label
    return largest_component


def remove_small_component(labels, min_area):
    for label_prop in measure.regionprops(labels):
        if(label_prop.area < min_area):
            labels[labels == label_prop.label] = 0

    return labels



if __name__ == '__main__':

    segmentLiver("data/Case1_CT.nii.gz", "data/Case1_Aorta.nii.gz", "output/Case1_Liver.nii.gz")

    (vod, dice, assd) = evaluateSegmentation("data/case1_liver_segmentation.nii.gz", "output/Case1_Liver.nii.gz")
    print("VOD: "+ str(vod))
    print("DICE: "+ str(dice))
    print("ASSD: "+ str(assd))
