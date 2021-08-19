# Private Functions

import cv2
import numpy as np
import sys
import os
import statistics


def _shortest_distance(n, c):
    '''Returns the distance between n and its closest neighbor in c.
    Parameter n is the coordinates of a centroid, c is a list of centroid coordinates.'''
    min_dis = np.inf
    for i in c:
        if (i[0] != n[0] and i[1] != n[1]) and sum((n - i)**2)**.5 < min_dis :
             min_dis = sum((n - i)**2)**.5 
    return min_dis


def _avg_shortest_distance(centroids):
    '''Returns standard deviation and mean of the shortest distances between centroids.
    Parameter centroids is a list of centroid coordinates.'''
    index = 0
    shortest_dis = []
    #iterates through every centroid
    for i in np.arange(0, len(centroids)):
        c = centroids[i]
        #compare given centroid to all other centroids and find distance to closest neighbor
        shortest_dis.append(_shortest_distance(c, centroids))
    return np.std(shortest_dis), np.mean(shortest_dis)


def _threshold_otsu(img):
    '''Thresholds image using Otsu and returns sure foreground of result.
    Parameter img is the array of the original image, returned by cv2.imread and already converted to grayscale.'''
    # Apply blur
    blur = cv2.GaussianBlur(img,(5,5),1)
    # Threshold image using Otsu
    ret, thresh = cv2.threshold(blur,0 , 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    assert(np.amin(thresh) != np.amax(thresh)), 'Unable to identify scales in image.'
    
    # Find average scale size
    binary_map = (thresh > 0).astype(np.uint8)
    output = cv2.connectedComponentsWithStats(binary_map, 4, cv2.CV_32S)
    stats = output[2]
    scale_sizes = np.copy(stats[1:,-1]) # start from 1 instead of 0 to ignore background, use copy so that stats array doesn't get modified
    scale_sizes.sort()
    avg_scale_size = np.average(scale_sizes[len(scale_sizes) // 2:]) # average size among the largest half of the scales
    
    # Determine number of iterations for opening and perform opening
    if avg_scale_size >= 100:
        if avg_scale_size < 400:
            num_iterations = int(avg_scale_size // 100)
        else:
            num_iterations = 4
        kernel = np.ones((3,3),np.uint8)
        opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations=num_iterations)
        # if the opening made all scales disappear, don't do any opening
        if np.amin(opening) == np.amax(opening):
            opening = thresh
    else:
        opening = thresh # if average scale size is already very small, don't do any opening
    
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform, 2, 255, 0)
    
    return sure_fg


def _choose_blocksize_and_iterations(otsu_img):
    '''Selects blocksize and iterations and returns them.
    Parameter otsu_img is the image returned by threshold_otsu.'''
    assert(np.amin(otsu_img) != np.amax(otsu_img)), 'Unable to identify scales in image.'

    # Collect stats from image
    binary_map = (otsu_img > 0).astype(np.uint8)
    connectivity = 4
    output = cv2.connectedComponentsWithStats(binary_map, connectivity, cv2.CV_32S)
    num_labels = output[0]
    labels = output[1]
    stats = output[2]
    centroids = output[3]

    avg_scale_size = np.average(stats[1:,-1])
    # Calculate "uniformity"
    distance_std, avg_shortest_dis = _avg_shortest_distance(centroids)
    
    # Determine blocksize and iterations
    block_size = int(2*np.sqrt(avg_scale_size/(np.pi)) + avg_shortest_dis)
    iterations = 0
    if block_size % 2 == 0: # Blocksize must be an odd number
        block_size = block_size + 1
    if block_size > 85:
        iterations = 1
    else:
        iterations = min(int(block_size/15), 3)
    return block_size, iterations


def _threshold_adaptive(img, blocksize, iterations, noise_thresh=1/7):
    '''Performs adaptive thresholding on img using given blocksize and iterations.
    Parameters:
        -img is the array of the original image, returned by cv2.imread and already converted to grayscale.
        -blocksize (odd integer) to use for adaptive thresholding
        -iterations: number of iterations of opening
        -noise_thresh (float <= 1): noise smaller than this fraction of the average scale area (among the larger half of scales) is removed
    Returns a dictionary containing:
        -sure foreground with noise (for displaying purposes)
        -blurred image
        -noise-removed image with scales labeled
        -scale count'''
    odd_median_blur_value = int(blocksize/2)
    if odd_median_blur_value % 2 == 0:
        odd_median_blur_value = odd_median_blur_value + 1
    cv2.medianBlur(img, odd_median_blur_value)
    if (blocksize > 15):
        blur = cv2.bilateralFilter(img,10,10,200)
    else:
        blur = cv2.GaussianBlur(img,(5,5),0)

    # Perform adaptive thresholding and collect stats from thresholded image
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blocksize, -2)
    binary_map = (thresh > 0).astype(np.uint8)
    connectivity = 4
    output = cv2.connectedComponentsWithStats(binary_map, connectivity, cv2.CV_32S)
    num_labels = output[0]
    labels = output[1]
    stats = output[2]
    centroids = output[3]
    
    # Calculate average scale size among the largest half of the scales
    scale_sizes = np.copy(stats[1:,-1]) # start from 1 instead of 0 to ignore background, use copy so that stats array doesn't get modified
    scale_sizes.sort() # sort the list of scale sizes from smallest to largest
    avg_scale_size = np.average(scale_sizes[len(scale_sizes) // 2:]) # average size among the largest half of the scales
    
    # Fill in holes that are less than 1/5 the average scale size
    thresh2 = cv2.bitwise_not(thresh)
    contour,hier = cv2.findContours(thresh2,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contour:
        if cv2.contourArea(cnt) < avg_scale_size/5:
            thresh2 = cv2.drawContours(thresh2, [cnt], -1, (0,0,0), -1)
    thresh = cv2.bitwise_not(thresh2)

    # Perform opening on resulting image
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = iterations)
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform, 1.5 , 255, 0) 

    # Collect stats from sure_foreground image
    binary_map = (sure_fg > 0).astype(np.uint8)
    connectivity = 4 # can be changed
    output = cv2.connectedComponentsWithStats(binary_map, connectivity, cv2.CV_32S)
    num_labels = output[0]
    labels = output[1]
    stats = output[2]
    centroids = output[3]
    
    # Calculate average scale size among the largest half of the scales
    scale_sizes = np.copy(stats[1:,-1]) # start from 1 instead of 0 to ignore background, use copy so that stats array doesn't get modified
    scale_sizes.sort() # sort the list of scale sizes from smallest to largest
    avg_scale_size = np.average(scale_sizes[len(scale_sizes) // 2:]) # average size among the largest half of the scales
    
    # Remove noise   
    for i in range(num_labels):
        # If a scale is smaller than the fraction noise_thresh of the average among the largest half of the scales
        if stats[i, cv2.CC_STAT_AREA] < (avg_scale_size*noise_thresh):
            noise_left = stats[i, cv2.CC_STAT_LEFT]
            noise_top = stats[i, cv2.CC_STAT_TOP]
            # Go through the bounding box of the noise and find all the pixels that are part of the noise
            for x in range(noise_left, noise_left + stats[i, cv2.CC_STAT_WIDTH]):
                for y in range(noise_top, noise_top + stats[i, cv2.CC_STAT_HEIGHT]):
                    if labels[y][x] == i: # If the pixel is part of the noise
                        labels[y][x] = 0 # Remove the noise by replacing the pixel with the background label
            num_labels -= 1
    # Note: Noise has been removed from labels matrix, but the noise is still in stats/centroids matrices.
    # This is ok because we don't use these outdated stats/centroids matrices again (in later functions, we create updated stats matrices by calling connected components again on the noise-removed image.)
    
    # Label and count
    # source: https://medium.com/analytics-vidhya/images-processing-segmentation-and-objects-counting-in-an-image-with-python-and-opencv-216cd38aca8e
    label_hue = np.uint8(179 * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    labeled_img[label_hue == 0] = 0

    results = {'with_noise':sure_fg, 'blur':blur, 'labeled_img':labeled_img,'count':num_labels - 1} # background doesn't count as a scale
    return results


def _overlay(bw_img, color_mask, w1=0.2, w2=0.95):
    '''Returns image with color_mask overlaid over bw_img.'''
    a = cv2.cvtColor(bw_img, cv2.COLOR_GRAY2RGB)
    a -= np.min(a)
    a = a/np.max(a)*255
    a = np.clip(a, a_min=0, a_max=255).astype(np.uint8)
    b = np.clip(color_mask, a_min=0, a_max=255).astype(np.uint8)
    c = cv2.addWeighted(b,w1,a,w2,0)
    return c


def _calculate_score(img):
    '''Calculates and returns score based on average scale size variation and uniformity of scale distribution.
    THE LOWER THE SCORE, THE BETTER.
    Parameter img is the labeled image, returned by threshold_adaptive'''
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary_map = (img > 0).astype(np.uint8) # black pixels are 0 and everything else is 1
    
    # Collect new data on connected components (rather than using previous stats array) because the noise-removal method in _threshold_adaptive changed the image but not the stats array
    output = cv2.connectedComponentsWithStats(binary_map, 4, cv2.CV_32S)
    num_labels = output[0]
    labels = output[1]
    stats = output[2]
    centroids = output[3]
    
    num_scales = num_labels - 1 # background doesn't count as scale
    if num_scales <= 2: # Checks only work if there's more than 2 scales
        return {'score':100, 'size_var':None, 'dist':None} # Return high value to show that result is not good if less than 3 scales
    
    size_var = _scale_size_variation(num_scales, stats, len(img), len(img[0]))
    dist = _distribution(num_scales, centroids) 
    score = (size_var + dist) / 2 # score is the average of size variation and distribution
    return {'score':score, 'size_var':size_var, 'dist':dist}

    
def _bounding_box_edges(i, stats):
    '''Returns coordinates of edges of the bounding box at index i in stats array'''
    left = stats[i, cv2.CC_STAT_LEFT]
    right = left + stats[i, cv2.CC_STAT_WIDTH] - 1
    top = stats[i, cv2.CC_STAT_TOP]
    bottom = top + stats[i, cv2.CC_STAT_HEIGHT] - 1
    return left, right, top, bottom

        
def _scale_size_variation(num_scales, stats, img_len, img_width):
    '''Returns average variation in scale size, calculated without edge scales (except in the case where there are not enough non-edge scales).
    High variation likely corresponds to poor quality results.
    Result is considered very good if return value is less than 1.
    Only works if there is at least 2 scales.'''
    assert num_scales >= 2, "Image must have at least 2 scales."
    scale_areas = []
    edge_scale_areas = []
    for i in range(1, num_scales + 1):
        left, right, top, bottom = _bounding_box_edges(i, stats)
        # don't consider scales that touch the edge
        if left == 0 or top == 0 or right == img_width - 1 or bottom == img_len - 1:
            edge_scale_areas.append(stats[i, cv2.CC_STAT_AREA].astype(int))
            continue
        scale_areas.append(stats[i, cv2.CC_STAT_AREA].astype(int))
    # if 1 or zero non-edge scales, then use edge scales in calculation (otherwise, do not use edge scales)
    if len(scale_areas) < 2:
        scale_areas = scale_areas + edge_scale_areas
    stddev = statistics.stdev(scale_areas)
    mean = sum(scale_areas) / len(scale_areas)
    # if any of the edge scales are more than five times the average size of the non-edge scales, result is bad
    for edge_scale_area in edge_scale_areas:
        if edge_scale_area > 5*mean:
            return 100 # Return high value because result is bad
    return (stddev / mean)


def _distribution(num_scales, centroids):
    '''Calculates and returns the standard deviation of the distances between the centroids of neighboring scales.
    The lower the return value, the more uniform the distribution of scales.
    Only works if there is at least 3 scales (2 distances to compare)'''
    assert num_scales >= 3, "Image must have at least 3 scales."
    x_values = []
    y_values = []
    for i in range(1, num_scales + 1):
        x_values.append(centroids[i, 0])
        y_values.append(centroids[i, 1])
    x_values.sort()
    y_values.sort()
    x_diff = []
    y_diff = []
    for i in range(1, len(x_values)):
        x_diff.append(x_values[i] - x_values[i-1]) # differences between neighboring x-coordinates
        y_diff.append(y_values[i] - y_values[i-1]) # differences between neighboring y-coordinates
    x_stddev = statistics.stdev(x_diff)
    y_stddev = statistics.stdev(y_diff)
    avg_stddev = (x_stddev + y_stddev) / 2
    return avg_stddev


def _compare_results(score_list, num_to_keep):
    '''Compares the scores from different thresholding attempts and returns a list of the indices corresponding to the best one(s).
    Parameters:
        -score_list is a list of scores for each image from calculate_score
        -num_to_keep is the number of images that we want to keep.'''
    best_indices = []
    for i in range(len(score_list)):
        if len(best_indices) < num_to_keep:
            best_indices.append(i)
        else:
            current_worst_score = max([score_list[i] for i in best_indices])
            if score_list[i] < current_worst_score: # if the score is better than the previous scores
                best_indices.remove(score_list.index(current_worst_score)) # remove the index corresponding to lowest score
                best_indices.append(i)
    return best_indices


def _invert(img):
    '''Inverts a grayscale image.'''
    return cv2.bitwise_not(img)


def _count_scales_helper(img, noise_thresh):
    '''Helper function for count_scales. Calls threshold_otsu, then choose_blocksize_and_iterations, then threshold_adaptive.
    Parameters:
        -img is the array of the original image, returned by cv2.imread and already converted to grayscale.
        -noise_thresh (float <= 1): noise smaller than this fraction of the average scale area (among the larger half of scales) is removed
    Returns:
        -results: a list of resulting images
        -data: a dictionary containing blocksize and iterations.'''
    img_otsu = _threshold_otsu(img)
    blocksize, iterations = _choose_blocksize_and_iterations(img_otsu)
    data = {'blocksize':blocksize, 'iterations':iterations}
    results = _threshold_adaptive(img, blocksize, iterations, noise_thresh)
    return results, data


def _estimate_total_counts(all_counts, best_indices, sub_img):
    '''Used by split_count_select; Estimates total scale count by taking the average count across selected subimages and multiplying by total number of subimages.'''
    return (sum([all_counts[i] for i in best_indices]) * sub_img / len(best_indices))    

def _wrap_title(title, length):
    '''Used to split a string into multiple lines if it is too long.
        Parameter title is the string.
        Parameter length is the maximum number of characters per line.'''
    if len(title) > length:
        lines = [title[i:i+length] for i in range(0, len(title), length)]
        return '\n'.join(lines)
    return title

def _last_part_of_path(img_path):
    '''Takes a string and removes everything to the left of the rightmost '/' character.'''
    slash_index = img_path.rfind('/')
    if slash_index > 0:
        return img_path[slash_index + 1:]
    return img_path
