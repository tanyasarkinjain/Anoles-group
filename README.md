# Anoles-group

Anoles-group (probably changing the name) allows users to count scales easily. The script has methods that can estimate the scales in a large area where 
scales are overall uniform but in some places difficult to count by hand. It can also count the total scales in smaller good quality images. There are two different methodsthat can handle for these cases. This should save time and tedium in laboratory settings, providing even results. 
The main code is in [Anoles_counting_scales.ipynb](https://github.com/tanyasarkinjain/Anoles-group/blob/master/Anoles_counting_scales.ipynb)

## Image Requirements
- Works best for scales and colonies that are not overlapping
- Works best for images taken **without** flash
- Works best for scale and colonies that contrast their background/medium and are paler than the background
- Works best for scales with overall uniform color
- It is okay if there is uneven lighting in someplaces (as long as no flash)
- Example Image Sets: [Anolis-cristatellus-Imgs](https://github.com/tanyasarkinjain/Anoles-group/tree/master/Anolis_cristatellus_images)    [Anolis-Guanica-County-Imgs](https://github.com/tanyasarkinjain/Anoles-group/tree/master/Anolis_cristatellus_images)   [Imgs-Result-From-Split](https://github.com/tanyasarkinjain/Anoles-group/tree/master/unsplit_images)

## Methods

### Framework for split_count_select: Green boxes indicate steps unique to split_count_select()
`split_count_select(img_directory_path, num_subimages, num_to_keep)`

Ideal for images that are large with scales/spots that are unclear in some regions.
Splits a large image into subimages of equal size (have to give it a number of subimages to split into) and will select the best subimages from which to estimate 
total scale/colony count.

For each subimage:
1. Performs Otsu threshold and uses results to determine blocksize and iterations.
2. Performs adaptive thresholding using selected blocksize and iterations. Removes noise.
3. Calculates a score for the result based on scale size variation and uniformity of distribution.
The lower the score, the better.
4. If the score is too high, repeat steps 1-3 on inverted image and see if the score for the inverted
image is lower. Keep the one with lower score.
5. Finally, choose the 3 subimages with the lowest (best) scores (printed in a list at the bottom as SELECTED SUBIMAGES)


<img width="1191" alt="Screenshot 2021-04-19 at 4 49 13 PM" src="https://user-images.githubusercontent.com/67300971/115316963-49492f80-a12f-11eb-8b73-61ebe6f44eb8.png">

### Framework for run_count_on_directory(): 
`run_count_on_directory(img_directory_path)`

Ideal for smaller images that have very clearly defined scales. Image should be good quality and mostly countable by hand.

For each image:
Does the same as split_count_select but does not split images and therefore does not choose best segments.

<img width="1200" alt="Screenshot 2021-04-19 at 4 48 56 PM" src="https://user-images.githubusercontent.com/67300971/115317476-7cd88980-a130-11eb-86ef-a73bf6bf27ec.png">
_____________________________________________________________________________________________________________________________________________

### Examples (using run_count_on_directory):

<img width="1215" alt="Screenshot 2021-04-16 at 12 49 11 AM" src="https://user-images.githubusercontent.com/67300971/115317636-d771e580-a130-11eb-8610-e8988420bc41.png">

<img width="1215" alt="Screenshot 2021-04-16 at 12 48 31 AM" src="https://user-images.githubusercontent.com/67300971/115346260-cfcc3400-a164-11eb-88c6-505d951f1f74.png">

