# SAMPLE TESTS

from ScaleCount_Public_Functions import count_scales, count_scales_directory, split_count_select, display_results

# SINGLE IMAGES:
#results, data = count_scales('Arctos_Database_images/UCM_8774_Sceloporus_dugesii_ventral_02.png')
#display_results(results, 'Sceloporus')

# WHOLE DIRECTORY:
count_scales_directory('Guanica_County_images', 'Guanica_Results')
#count_scales_directory('Fish_scales', 'Fish_Results')
#count_scales_directory('sample_bacteria_colonies', 'Bacteria_results')
#count_scales_directory('Anolis_cristatellus_images', 'Anolis_results')
#count_scales_directory('Sea_Bass_Kaggle', 'Sea_bass_results')

# SPLIT COUNT SELECT:
#split_count_select('Arctos_Database_images/UCM_8774_Sceloporus_dugesii_ventral_02.png', 10, 3)

