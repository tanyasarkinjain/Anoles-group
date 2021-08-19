# TODO
# Remove the part in count_scales_directory that has a comment saying to remove

from ScaleCount_Private_Functions import _count_scales_helper, _calculate_score, _compare_results, _invert, _overlay, _estimate_total_counts, _wrap_title, _last_part_of_path   
import cv2
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import sys
import os
#import warnings
#warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from IPython.display import display
import pandas
import image_slicer
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.axes

def count_scales(img_name, check_invert='auto', noise_thresh=1/7):
    '''Main function; counts scales on a single image.
    Counts scales on original image, checks result, and counts scales on inverted image if needed.
    Finally, compares results from original vs. inverted image and returns better results.
    Parameters:
        -img_name (string) is the filepath of the image to be analyzed.
        -check_invert (string, optional) can have values 'auto', 'orig', or 'invert'.
            'auto': count both original and inverted images; select better result
            'orig': count original image only
            'invert': count inverted image only
        -noise_thresh (float, optional): noise smaller than this fraction of the average scale area (among the larger half of scales) is removed
    Returns:
        1. A dictionary containing the resulting images from whichever image was better (original or inverted):
            dictionary keys are 'original', 'inverted' (only if inverted used), 'blur', 'with_noise', 'labeled_img', 'count', 'img_name', 'subimage_from_split') 
        2. A dictionary containing 'blocksize', 'iterations', 'score', 'size_var', and 'distribution' from whichever image was better (original or inverted)
    Note: To display results, call display_results function and use the first return value of count_scales as the first parameter
    '''
    assert check_invert == 'auto' or check_invert == 'orig' or check_invert == 'invert', "check_invert must be 'auto', 'orig', or 'invert'"
    assert isinstance(noise_thresh, float) or isinstance(noise_thresh, int), 'Noise_thresh must be a float, usually a fraction less than 1.'
    orig_img = cv2.imread(img_name)
    assert isinstance(orig_img, np.ndarray), 'Invalid image name.'
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    
    if check_invert == 'orig' or check_invert == 'auto':
        # count scales on original image
        orig_results, orig_data = _count_scales_helper(orig_img, noise_thresh)
        orig_results['img_name'] = img_name
        orig_results['original'] = orig_img
        orig_results['subimage_from_split'] = False
        # check quality of results
        orig_scores = _calculate_score(orig_results['labeled_img'])
        orig_data.update(orig_scores)
        # If on original mode, or on auto mode with passing score, display results and finish
        if check_invert == 'orig' or orig_data['score'] < 3:
            return orig_results, orig_data
    
    # Try inverted image
    inverted_img = _invert(orig_img)
    # count scales on inverted image
    inverted_results, inverted_data = _count_scales_helper(inverted_img, noise_thresh)
    inverted_results['img_name'] = img_name
    inverted_results['original'] = orig_img
    inverted_results['inverted'] = inverted_img
    inverted_results['subimage_from_split'] = False
    # check quality of results
    inverted_scores = _calculate_score(inverted_results['labeled_img'])
    inverted_data.update(inverted_scores)
    
    # If on invert mode, display results and finish
    if check_invert == 'invert':
        return inverted_results, inverted_data
    
    # Otherwise, compare the results from original and inverted images
    else:
        score_list = [orig_data['score'], inverted_data['score']]
        best_index = _compare_results(score_list, 1)[0]
        if best_index == 0:
            # Keep original
            return orig_results, orig_data
        elif best_index == 1:
            # Keep inverted
            return inverted_results, inverted_data

def count_scales_directory(dirname):
    '''Calls count_scales function on every image in the given directory.
    Parameter dirname (string) is the name of the directory.
    Parameter output_name (string) is the desired name for the output directory.
    Returns a list of dictionaries (each dictionary is the first return value from calling count_scales on each image)'''
    assert os.path.isdir(dirname), 'Invalid directory name.'
    directory = os.scandir(dirname)
    results_list = []
    for img in directory:
        # REMOVE THIS PART LATER
        if (img.name == '.ipynb_checkpoints' or img.name == '.DS_Store'):
            continue
        #########################
        img_filepath = dirname + '/' + img.name
        results, data = count_scales(img_filepath)
        results_list.append(results)
    return results_list

def split_count_select(img_path, num_subimages=0, num_to_keep=0):
    '''Splits given image into subimages, counts scales in each subimage, and selects best ones to keep.
    All subimages are placed in a new directory called "Subimages_From_Splitting_[original image name]"
    Parameters:
        -img_path (string): filepath for original image
        -num_subimages (int, optional): number of subimages to split the image into
            Note: according to documentation, image_slicer will split the image into a
            number of images greater than or equal to num_subimages, so long as the image is split into equal parts
            https://readthedocs.org/projects/image-slicer/downloads/pdf/latest/
        -num_to_keep (int, optional): number of subimages to keep
    Returns:
        1. List of dictionaries (each dictionary is the first return value from calling count_scales on each subimage.
        2. List of the indices of the selected subimages
        3. Estimated total scale count for original image based on selected subimages
    Note: To display results, call display_results function and use the return values from split_count_select as parameters'''
    assert isinstance(num_subimages, int), 'num_subimages must be an integer.'
    assert isinstance(num_to_keep, int), 'num_to_keep must be an integer.'
    assert(num_to_keep <= num_subimages), 'num_to_keep cannot be greater than num_subimages'
    img = cv2.imread(img_path)
    assert isinstance(img, np.ndarray), 'Invalid image name.'

    # Determine num_subimages and num_to_keep
    if num_subimages == 0:
        results, data = count_scales(img_path)
        count = results['count']
        if count < 200:
            num_subimages = 12
            num_to_keep = 3
        elif count < 500:
            num_subimages = count//20
            num_to_keep = num_subimages//3
        else:
            num_subimages = 30
            num_to_keep = 10
            
    img_size = img.shape
    # Split image into subimages
    tiles = image_slicer.slice(img_path, num_subimages, save=False)
    img_name_cropped = _last_part_of_path(img_path)
    subimage_dirname = 'Subimages_From_Splitting_' + img_name_cropped
    # If directory name is already taken, add a number to the end
    i = 1
    while os.path.isdir(subimage_dirname):
        if i == 1:
            subimage_dirname = subimage_dirname + '(1)'
        else:
            subimage_dirname = subimage_dirname[:-2] + str(i) + ')'
        i += 1
    os.mkdir(subimage_dirname)
    image_slicer.save_tiles(tiles, directory=subimage_dirname, prefix=img_name_cropped)
    all_scores = []
    all_counts = []
    all_labeled = []
    i = 0
    results_list = []
    # Count scales in each subimage
    for tile in tiles:
        results, data = count_scales(tile.filename)
        results['subimage_from_split'] = True
        results_list.append(results)
        all_scores.append(data['score'])
        all_counts.append(results['count'])
        all_labeled.append(results['labeled_img'])
        i += 1
    # Select the best subimages
    best_indices_lst = _compare_results(all_scores, num_to_keep)
    # Estimate total number of scales
    estimated_total = _estimate_total_counts(all_counts, best_indices_lst, num_subimages)
    return results_list, best_indices_lst, estimated_total

def display_results(results_list, output_name="ScaleCount_results_display", best_indices_lst=None, estimated_total=None):
    '''Outputs a new directory called output_name, containing the following files:
        1. A pdf called Results_Images_[output_name].pdf (or if displaying for split_count_select, pdf is called Subimage_Counts_[output_name].pdf.
           The pdf displays, for every image or subimage, the following:
           original image, inverted image (if applicable), thresholded image with noise, noise-removed image with scales labeled, and overlaid image with scale count.
        2. A csv file containing a table with all the image (or subimage) names and corresponding scale counts.
        3. (Only if displaying results from split_count_select) A pdf called Selected_Subimages_[output_name].pdf.
            The pdf displays all of the selected subimages with scales labeled and the estimated total count.
    Parameters:
        1. results_list (list): list of dictionaries, from the first return value of count_scales, count_scales_directory, or split_count_select
         (results_list doesn't have to be a list, it may be just a single dictionary.)
        2. output_name (string, optional): desired name for the output directory
        3. best_indices_lst (list, only required if displaying for split_count_select): second return value from split_count_select
        4. estimated_total (float, only required if displaying for split_count_select): third return value from split_count_select'''
    assert(isinstance(output_name, str)), 'Output name must be a string.'
    #add assert statement checking that output_name doesn't have any invalid characters such as slashes
    if not isinstance(results_list, list):
        results_list = [results_list]
    # Check if displaying results for split_count_select
    split = True if (results_list[0]['subimage_from_split']) else False
    if split:
        assert(best_indices_lst != None), 'Missing parameter best_indices_lst.'
        assert(estimated_total != None), 'Missing parameter estimated_total.'
    i = 1
    # If output_name already taken, add a number to the end
    while os.path.isdir(output_name):
        if i == 1:
            output_name = output_name + '(1)'
        else:
            output_name = output_name[:-2] + str(i) + ')'
        i += 1
    os.mkdir(output_name)
    pdf_prefix = r'/Subimage_Counts_' if split else r'/Result_Images_'
    # Display images in a pdf
    with PdfPages(output_name + pdf_prefix + output_name + '.pdf') as export_pdf:
        fig = plt.figure(figsize=(8.5, 11))
        title_prefix = 'Subimage Counts for ' if split else 'Scale Counts for '
        title = title_prefix + output_name
        title = _wrap_title(title, 60)
        if split:
            title += '\n Original Image split into ' + str(len(results_list)) + ' Subimages'
        fig.suptitle(title, fontsize=14, fontweight='bold')
        index = 1
        page_row_count = 0
        total_row_count = 0
        for results_dict in results_list:
            count = results_dict['count']
            img_name = _last_part_of_path(results_dict['img_name'])
            img_name = _wrap_title(img_name, 20)
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
            # close page after every 5 rows or if finished
            if page_row_count % 5 == 0 or total_row_count == len(results_list):
                for ax in fig.axes:
                    ax.axis("off")
                export_pdf.savefig()
                plt.close()
                # start new page if not finished yet
                if total_row_count < len(results_list):
                    fig = plt.figure(figsize=(8.5, 11))
                    index = 1

    # Create table and save to a csv file
    counts = [d['count'] for d in results_list]
    img_names = [_last_part_of_path(d['img_name']) for d in results_list]
    table=pandas.DataFrame()
    first_col = 'Subimage Name' if split else 'Image Name'
    table[first_col] = img_names
    table['Scale Count'] = counts
    if split:
        table['Subimage Selected?'] = ['Yes' if i in best_indices_lst else 'No' for i in range(len(results_list))]
        table['Estimated Total Scale Count'] = [estimated_total] + ['-']*(len(results_list)-1)
    display(table)
    table_prefix = r'/Subimage_Counts_Table_' if split else r'/ScaleCounts_Table_'
    table.to_csv(output_name + table_prefix + output_name + '.csv', index = False)

    # If displaying results from split_count_select, create an additional pdf with selected subimages and estimated total count
    if split:
        with PdfPages(output_name + r'/Selected_Subimages_' + output_name + '.pdf') as export_pdf:
            num_selected = len(best_indices_lst)
            fig = plt.figure(figsize=(8.5, 11))
            title = 'Selected Subimages for ' + output_name
            title = _wrap_title(title, 60)
            title += '\n' + str(num_selected) + ' Subimages Selected from ' + str(len(results_list)) + ' Total Subimages' 
            title += '\nEstimated Total Count: ' + str(estimated_total) + ' Scales'
            fig.suptitle(title, fontsize=14, fontweight='bold')
            plot_index = 1
            for i in np.arange(num_selected):
                img_data = results_list[best_indices_lst[i]]
                plt.subplot(5,5,plot_index), plt.imshow(img_data['labeled_img'], 'gray')
                img_title = _last_part_of_path(img_data['img_name'])
                img_title = _wrap_title(img_title, 20) + '\n Count: ' + str(img_data['count']) + ' Scales'
                plt.title(img_title, fontsize=7)
                plot_index += 1
                # If current page is full or if finished, close page
                if plot_index > 25 or i == num_selected - 1:
                    for ax in fig.axes:
                        ax.axis("off")
                    export_pdf.savefig()
                    plt.close()
                    # start new page if not finished yet
                    if i < num_selected - 1:
                        fig = plt.figure(figsize=(8.5, 11))
                        plot_index = 1

