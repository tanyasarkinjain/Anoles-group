# SAMPLE TESTS
import Scale_Counter
from Scale_Counter.ScaleCount_Public_Functions import *

# SINGLE IMAGE:
results, data = count_scales('UCM_HERP_12429_Anolis_rodriguezii_ventral.jpeg')
display_results(results, 'SAMPLE_RESULTS_WHOLE_IMAGE')

# WHOLE DIRECTORY:
results = count_scales_directory('Sample_Image_Directory')
display_results(results, 'SAMPLE_RESULTS_DIRECTORY')


# SPLIT COUNT SELECT:
results, best_indices, estimated_total = split_count_select('UCM_HERP_12429_Anolis_rodriguezii_ventral.jpeg')
display_results(results, 'SAMPLE_RESULTS_SPLIT_COUNT',best_indices_lst=best_indices, estimated_total=estimated_total)
