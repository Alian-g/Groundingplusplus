from mgit.experiments import ExperimentMGIT
import os
# ... path setting ...
root_dir = "/data/ATCTrack"
# setup experiment (validation subset)
experiment = ExperimentMGIT(
  root_dir='/data3/dataset/MGIT/', # MGIT's root directory
  save_dir= os.path.join(root_dir,'result'), # the path to save the experiment result
  subset='test', # 'train' | 'val' | 'test'
  repetition=1,
  version='tiny' # temporarily, the toolkit only support tiny version of MGIT
)
tracker_name = "atctrack_large"
experiment.convert_results(
  # the original results path will be "root_dir/tracker_name/original_results_folder", and the converted results will be saved to "root_dir/result"
  '/data/ATCTrack', # the path of tracker e.g. JointNLT results
  "atctrack_large", # tracker name
  "videocube_test_tiny"# original result folder name
  )
experiment.report(["atctrack_large"],"normal")