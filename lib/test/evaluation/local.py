from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/data3/dataset/got10k_lmdb'
    settings.got10k_path = '/data3/dataset/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_extension_subset_path = '/data3/dataset/lasot_ext'
    settings.lasot_lmdb_path = '/data3/dataset/lasot_lmdb'
    settings.lasot_path = '/data3/dataset/lasot'
    settings.lasotlang_path = '/data3/dataset/lasot'
    settings.network_path = '/data/ATCTrack/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/data3/dataset/nfs'
    settings.otb_path = '/data3/dataset/OTB2015'
    settings.otblang_path = '/data3/dataset/OTB_sentences'
    settings.prj_dir = '/data/ATCTrack'
    settings.result_plot_path = '/data/ATCTrack/test/result_plots'
    settings.results_path = '/data/ATCTrack/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/data/ATCTrack'
    settings.segmentation_path = '/data/ATCTrack/test/segmentation_results'
    settings.tc128_path = '/data3/dataset/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/data3/dataset/tnl2k/test'
    settings.tpl_path = ''
    settings.trackingnet_path = '/data3/dataset/trackingnet'
    settings.uav_path = '/data3/dataset/UAV123'
    settings.vot_path = '/data3/dataset/VOT2019'
    settings.youtubevos_dir = ''

    return settings

