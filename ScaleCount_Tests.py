# SAMPLE TESTS

from ScaleCount_Public_Functions import count_scales, count_scales_directory, split_count_select, display_results

# SINGLE IMAGES:
#results, data = count_scales('Arctos_Database_images/UCM_8774_Sceloporus_dugesii_ventral_02.png')
#display_results(results, 'ARCTOS_SCELOPORUS_RESULTS')


# WHOLE DIRECTORY:
#results = count_scales_directory('Guanica_County_images')
#display_results(results, 'GUANICA_RESULTS')

#results = count_scales_directory('Fish_scales') #this is the folder of multicolored scales we thought was fish but actually sceloporus lizards
#display_results(results, 'SCELOPORUS_RESULTS')

#results = count_scales_directory('sample_bacteria_colonies')
#display_results(results, 'BACTERIA_RESULTS')

#results = count_scales_directory('Anolis_cristatellus_images')
#display_results(results, 'ANOLIS_RESULTS')

#results = count_scales_directory('Sea_Bass_Kaggle')
#display_results(results, 'SEA_BASS_RESULTS')


# SPLIT COUNT SELECT:
results, best_indices, estimated_total = split_count_select('Arctos_Database_Images/UCM_HERP_12429_Anolis_rodriguezii_ventral.jpeg', 20, 10)
display_results(results, 'ARCTOS RODRIGUEZII RESULTS', best_indices, estimated_total)
