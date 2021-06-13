# Public Functions:
# 1.count_scales
# 2.count_scales_directory
# 3.split_count_select
# 4.display results

# TODO
# pdf and csv files should have unique filenames each time so previous files are not overwritten
# currently displays warning message about clipping input data - this is because the overlay image is some weird format.
#### I tried fixing this by converting overlay image to np.uint8, and this got rid of the warning message but added weird colored speckles to the overlay image
# For split_count_select:
#### Currently adds created subimages to the folder with original images -> need to fix this
#### Test whether estimated total count is accurate and display the estimated total count somewhere?
#### Fix titles at the top of the two pdfs outputted by split_count_select

# Explanation of split_count_select()
# Splits a large image into subimages of equal size (have to give it a number of subimages to split into).
# For each subimage:
# 1. Performs Otsu threshold and uses results to determine blocksize and iterations.
# 2. Performs adaptive thresholding using selected blocksize and iterations. Removes noise.
# 3. Calculates a score for the result based on scale size variation and uniformity of distribution. The lower the score, the better.
# 4. If the score is too high, repeat steps 1-3 on inverted image and see if the score for the inverted image is lower. Keep the one with lower score.
# Finally, choose the 3 subimages with the lowest (best) scores (printed in a list at the bottom as SELECTED SUBIMAGES)

# Explanation of run_count_on_directory()
# Does the same as split_count_select but does not split images and therefore does not choose best subimages.

from ScaleCount_Private_Functions import _count_scales_helper, _calculate_score, _compare_results, _invert, _overlay, _estimate_total_counts   
import cv2
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import sys
import os
import warnings
warnings.filterwarnings('ignore')
#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
#mpl.rcParams['figure.dpi'] = 120
#from IPython.display import set_matplotlib_formats
from IPython.display import display
#set_matplotlib_formats('retina')
import pandas
import image_slicer
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.axes

def count_scales(img_name, check_invert='auto', noise_thresh=1/7):
    '''Main function; counts scales on a single image.
    Counts scales on original image, checks result, and counts scales on inverted image if needed.
    Finally, compares results from original vs. inverted image and returns better results.
    Parameters:
        -img_name is the filepath of the image to be analyzed.
        -check_invert can have values 'auto', 'orig', or 'invert'.
            'auto': count both original and inverted images; select better result
            'orig': count original image only
            'invert': count inverted image only
        -noise_thresh (float <= 1): noise smaller than this fraction of the average scale area (among the larger half of scales) is removed
    Returns:
        1. A dictionary containing the resulting images from whichever image was better (original or inverted):
            dictionary keys are 'original', 'inverted' (only if inverted used), 'blur', 'with_noise', 'labeled_img', 'count', img_name') 
        2. A dictionary containing 'blocksize', 'iterations', 'score', 'size_var', and 'distribution' from whichever image was better (original or inverted)
    Note: to display results, call display_results function on the first return value of count_scales
    '''
    assert check_invert == 'auto' or check_invert == 'orig' or check_invert == 'invert', "check_invert must be 'auto', 'orig', or 'invert'"
    orig_img = cv2.imread(img_name)
    assert isinstance(orig_img, np.ndarray), 'Invalid image name.'
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    
    if check_invert == 'orig' or check_invert == 'auto':
        # count scales on original image
        orig_results, orig_data = _count_scales_helper(orig_img, noise_thresh)
        orig_results['img_name'] = img_name
        orig_results['original'] = orig_img
        # check how good the results are
        orig_scores = _calculate_score(orig_results['labeled_img'])
        orig_data.update(orig_scores)
        #print('Original image data: ' + str(orig_data))
        # If on original mode, or on auto mode and the result is good, display results and finish
        if check_invert == 'orig' or orig_data['score'] < 3:
            #print("\nOriginal image passed all tests; did not need to check inverted.")
            return orig_results, orig_data
    
    # Try inverted image and compare results to original
    inverted_img = _invert(orig_img)
    # count scales on inverted image
    inverted_results, inverted_data = _count_scales_helper(inverted_img, noise_thresh)
    inverted_results['img_name'] = img_name
    inverted_results['original'] = orig_img
    inverted_results['inverted'] = inverted_img
    # check how good the new results are
    inverted_scores = _calculate_score(inverted_results['labeled_img'])
    inverted_data.update(inverted_scores)
    #print('Inverted image data: ' + str(inverted_data))
    
    # If on invert mode, display results and finish
    if check_invert == 'invert':
        return inverted_results, inverted_data
    
    # Otherwise, compare the results from original and inverted images
    else:
        score_list = [orig_data['score'], inverted_data['score']]
        best_index = _compare_results(score_list, 1)[0]
        if best_index == 0:
            #print("\nTried both original and inverted, but decided to keep original.")
            return orig_results, orig_data
        elif best_index == 1:
            #print("\nDECIDED TO USE INVERTED IMAGE.")
            return inverted_results, inverted_data

def count_scales_directory(dirname):
    '''Calls count_scales function on every image in the given directory and displays results.
    Resulting images are saved to a pdf and counts are saved to a csv.
    Parameter dirname is the name of the directory.
    Returns a list of dictionaries, which are the first return value when calling count_scales on each image.'''
    directory = os.scandir(dirname)
    results_list = []
    for img in directory:
        # REMOVE THIS LINE LATER
        if (img.name == '.ipynb_checkpoints' or img.name == '.DS_Store'):
            continue
        img_filepath = dirname + '/' + img.name
        results, data = count_scales(img_filepath)
        results_list.append(results)
    display_results(results_list, dirname)
    return results_list

def split_count_select(img_path, num_subimages, num_to_keep):
    '''Splits given image into subimages, counts scales in each subimage, and selects best ones to keep. Displays results.
    Results for all subimages (including the selected ones) are displayed in one pdf.
    Results for only the selected subimages are displayed in another pdf.
    Counts for all subimages are saved to a csv.
    Parameters:
        -img_path: filepath for original image
        -num_subimages: number of subimages to split the image into
        -num_to_keep: number of subimages to keep
    Returns:
        -a list of dictionaries, which are the first return value when calling count_scales on each subimage.
        -a list of the indices of the selected subimages
        -estimated total scale count for original image'''
    img = cv2.imread(img_path)
    img_size = img.shape
    tiles = image_slicer.slice(img_path, num_subimages)
    all_scores = []
    all_counts = []
    all_labeled = []
    i = 0
    results_list = []
    for tile in tiles:
        #print('SUBIMAGE #' + str(i) + ':\n')
        results, data = count_scales(tile.filename)
        results_list.append(results)
        all_scores.append(data['score'])
        all_counts.append(results['count'])
        all_labeled.append(results['labeled_img'])
        i += 1
    display_results(results_list)
    best_indices_lst = _compare_results(all_scores, num_to_keep)
    
    estimated_total = _estimate_total_counts(all_counts, best_indices_lst, num_subimages)
    #print('ESTIMATED TOTAL COUNT:', estimated_total, "SCALES")
    #print('\n_______________________________________________')
    #print('\nSELECTED SUBIMAGES: ' + str(best_indices_lst))
    
    with PdfPages(r'selected_subimages.pdf') as export_pdf:
        fig = plt.figure(figsize=(8.5, 11))
        fig.suptitle('Selected Subimages from ' + img_path, fontsize=14, fontweight='bold')
        for i in np.arange(num_to_keep):
            plt.subplot(1,num_to_keep,i+1), plt.imshow(all_labeled[best_indices_lst[i]], 'gray')
            plt.title("Chosen Image: " + str(best_indices_lst[i]), fontsize=5)
        export_pdf.savefig()
        plt.close()
    
    return results_list, best_indices_lst, estimated_total

def display_results(results_list, dirname=None):
    '''Displays original image, inverted image (if applicable), thresholded image with noise,
    noise-removed image with scales labeled, and overlaid image.
    Displayed images are saved to a pdf file, and a table with image names and counts are saved to a csv file.
    Parameter results_list: list of dictionaries containing original, inverted (only if inverted used), blurred, thresholded, labeled_img, img_name.
    results_list doesn't have to be a list, it may be just a single dictionary.
        '''
    if not isinstance(results_list, list):
        results_list = [results_list]
    index = 1
    page_row_count = 0
    total_row_count = 0

    # Display images in a pdf
    with PdfPages(r'scale_count_images.pdf') as export_pdf:
        fig = plt.figure(figsize=(8.5, 11))
        title = 'Scale Counts'
        if dirname:
            title += (' for ' + dirname)
        fig.suptitle(title, fontsize=14, fontweight='bold')
        for results_dict in results_list:
            count = results_dict['count']
            img_name = results_dict['img_name']
            # wrap image name if very long
            if len(img_name) > 20:
                lines = [img_name[i:i+20] for i in range(0, len(img_name), 20)]
                img_name = '\n'.join(lines)
            images = [results_dict['original'], results_dict['with_noise'], results_dict['labeled_img']]
            titles = [img_name, 'With Noise', 'Noise-Removed']
            if 'inverted' in results_dict:
                images.insert(1, results_dict['inverted'])
                titles.insert(1, 'Inverted')
            for x in range(len(images)):
                plt.subplot(5,5,index), plt.imshow(images[x], 'gray')
                plt.title(titles[x], fontsize=7)
                index += 1
            # display overlay image
            overlay_img = _overlay(results_dict['original'], results_dict['labeled_img'])
            plt.subplot(5,5,index), plt.imshow(overlay_img)
            plt.title('Overlay, Count = ' + str(count), fontsize=7)
            index += 1
            if 'inverted' not in results_dict: # if not inverted, skip over last col
                index += 1
            page_row_count += 1
            total_row_count += 1
            # start new page after every 5 rows
            if page_row_count % 5 == 0 and total_row_count < len(results_list):
                for ax in fig.axes:
                    ax.axis("off")
                export_pdf.savefig()
                plt.close()
                fig = plt.figure(figsize=(8.5, 11))
                index = 1
        for ax in fig.axes:
            ax.axis("off")
        export_pdf.savefig()
        plt.close()

    # Create table and save to a csv file
    counts = [d['count'] for d in results_list]
    img_names = [d['img_name'] for d in results_list]
    table=pandas.DataFrame()
    table['Image Names'] = img_names
    table['Count'] = counts
    display(table)
    table.to_csv(r'scale_count_table.csv', index = False)
