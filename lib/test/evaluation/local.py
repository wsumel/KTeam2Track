from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/home/wsl/wsl_workspace/Tracking_dataset/GOT-10k_lmdb'
    settings.got10k_path = '/home/wsl/wsl_workspace/Tracking_dataset/GOT-10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_extension_subset_path = '/home/wsl/wsl_workspace/Tracking_dataset/LaSOT/LaSOT_extension_subset'
    settings.lasot_lmdb_path = '/disk3/wsl_tmp/Workspace210/MDTrack/data/lasot_lmdb'
    settings.lasot_path = '/home/wsl/wsl_workspace/Tracking_dataset/LaSOT/LaSOT/LaSOTBenchmark'
    settings.lasotlang_path = '/disk3/wsl_tmp/Workspace210/MDTrack/data/lasot'
    settings.network_path = '/disk3/wsl_tmp/Workspace210/MDTrack/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/disk3/wsl_tmp/Workspace210/MDTrack/data/nfs'
    settings.otb_path = '/disk3/wsl_tmp/Workspace210/MDTrack/data/OTB2015'
    settings.otblang_path = '/disk3/wsl_tmp/Workspace210/MDTrack/data/otb_lang'
    settings.prj_dir = '/disk3/wsl_tmp/Workspace210/MDTrack'
    settings.result_plot_path = '/disk3/wsl_tmp/Workspace210/MDTrack/test/result_plots'
    settings.results_path = '/disk3/wsl_tmp/Workspace210/MDTrack/1104'    # Where to store tracking results
    settings.save_dir = '/disk3/wsl_tmp/Workspace210/MDTrack/1104'
    settings.segmentation_path = '/disk3/wsl_tmp/Workspace210/MDTrack/test/segmentation_results'
    settings.tc128_path = '/disk3/wsl_tmp/Workspace210/MDTrack/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/disk3/wsl_tmp/Workspace210/MDTrack/data/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/disk3/wsl_tmp/Workspace210/MDTrack/data/trackingnet'
    settings.uav_path = '/disk3/wsl_tmp/Workspace210/MDTrack/data/UAV123'
    settings.vot_path = '/disk3/wsl_tmp/Workspace210/MDTrack/data/VOT2019'
    settings.rgbt234_path = '/disk3/dataset/RGBT234/RGBT234/'


    settings.youtubevos_dir = ''

    return settings

