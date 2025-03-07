from opensfm import io
from opensfm import reconstruction


def run_dataset(data):
    """ Compute the SfM reconstruction. """

    import os
    import yaml
    config_path = os.path.join(data.data_path,"config_full_opensfm.yaml")
    with open(config_path, 'w') as file:
        yaml.dump(data.config, file)

    tracks_manager = data.load_tracks_manager()
    report, reconstructions = reconstruction.incremental_reconstruction(
        data, tracks_manager
    )
    
    
    data.save_reconstruction(reconstructions)
    data.save_report(io.json_dumps(report), "reconstruction.json")
