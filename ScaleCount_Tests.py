# SAMPLE TESTS

from ScaleCount_Public_Functions import count_scales, count_scales_directory, split_count_select, display_results

# SINGLE IMAGES:
#results, data = count_scales('Arctos_Database_images/UCM_HERP_12429_Anolis_rodriguezii_ventral.jpeg')
#display_results(results)

# WHOLE DIRECTORY:
count_scales_directory('Guanica_County_images')
#count_scales_directory('Fish_scales')
#count_scales_directory('sample_bacteria_colonies')
#count_scales_directory('Anolis_cristatellus_images')

# SPLIT COUNT SELECT:
#split_count_select('Arctos_Database_images/UCM_HERP_12429_Anolis_rodriguezii_ventral.jpeg', 10, 3)

