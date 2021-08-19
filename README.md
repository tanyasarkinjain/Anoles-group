# ScaleCount

ScaleCount allows users to count scales easily. The script has methods that can estimate the scales in a large area where 
scales are overall uniform but in some places difficult to count by hand. It can also count the total scales in smaller good quality images. There are two different methods that can handle for these cases. This should save time and tedium in laboratory settings, providing even results. 
The main code is in [ScaleCount_Public_Functions.py](https://github.com/tanyasarkinjain/ScaleCount/blob/master/ScaleCount_Public_Functions.py)

## Image Requirements
Works best for: 
- scales and colonies that are not overlapping and instead are distinct
- images taken **without** flash
- scales and colonies that contrast their background/medium and are paler than the background
- scales with overall uniform color
- skin/background that has uniform color in image

*It is okay if there is uneven lighting in someplaces (as long as no flash)

- Example Image Sets: [Anolis-cristatellus-Imgs](https://github.com/tanyasarkinjain/Anoles-group/tree/master/Anolis_cristatellus_images)    [Anolis-Guanica-County-Imgs](https://github.com/tanyasarkinjain/Anoles-group/tree/master/Anolis_cristatellus_images)

## Other Requirements
- numpy
- matplotlib
- opencv2
- IPython.display
- pandas
- image_slicer
- statistics
- datascience

## Method Frameworks

### count_scales(): 
`count_scales(img_name, check_invert='auto', noise_thresh=1/7)`

Ideal for smaller images that have very clearly defined scales. Image should be good quality and mostly countable by hand.

1. Performs Otsu threshold and uses results to determine blocksize and iterations.
2. Performs adaptive thresholding using selected blocksize and iterations. Removes noise.
3. Calculates a score for the result based on scale size variation and uniformity of distribution.
The lower the score, the better.
4. If the score is too high, repeat steps 1-3 on inverted image and see if the score for the inverted
image is lower. Keep the one with lower score.

<img width="1200" alt="Screenshot 2021-04-19 at 4 48 56 PM" src="https://user-images.githubusercontent.com/67300971/115317476-7cd88980-a130-11eb-86ef-a73bf6bf27ec.png">

### count_scales_directory():
`count_scales_directory(dirname)`

Runs count_scales on each image in the directory. 

### split_count_select: Green boxes indicate steps unique to split_count_select()
`split_count_select(img_path, num_subimages=0, num_to_keep=0)`

Ideal for images that are large with scales/spots that are unclear in some regions.

For each subimage:
1. Runs count_scales on each subimage.
2. Finally, choose the subimages with the best scores.
3. Estimates total count using the selected subimages.


<img width="1191" alt="Screenshot 2021-04-19 at 4 49 13 PM" src="https://user-images.githubusercontent.com/67300971/115316963-49492f80-a12f-11eb-8b73-61ebe6f44eb8.png">

### display_results():
`display_results(results_list, output_name="ScaleCount_results_display", best_indices_lst=None, estimated_total=None)`

Displays pdf showing labeled and counted images. 

___________________________________________________________________________________________________________________

### Examples (using count_scales):

<img width="1215" alt="Screenshot 2021-04-16 at 12 48 31 AM" src="https://user-images.githubusercontent.com/67300971/115346260-cfcc3400-a164-11eb-88c6-505d951f1f74.png">

