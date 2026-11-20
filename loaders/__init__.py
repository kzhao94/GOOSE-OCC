import os

from .pipelines import __all__
from .goose_occ_mono import GOOSEOccMono
from .goose_temporal_occ_mono import GOOSETemporalOccMono


if os.environ.get('VISIBILITYOCC_GOOSE_ONLY') != '1':
    from .nuscenes_dataset import CustomNuScenesDataset
    from .nuscenes_occ_dataset import NuSceneOcc
    from .nuscenes_occ_dataset_gs import NuSceneOccGS
    from .nuscenes_occ_dataset_mono import NuSceneOccMono
    from .semantic_kitti_occ_mono import SemanticKITTIOccMono
    from .semantic_kitti360_occ_mono import SemanticKITTI360OccMono
    from .rellis3d_wildocc_mono import RELLIS3DWildOccMono

if os.environ.get('VISIBILITYOCC_GOOSE_ONLY') == '1':
    __all__ = ['GOOSEOccMono', 'GOOSETemporalOccMono']
else:
    __all__ = [
        'CustomNuScenesDataset', 'NuSceneOcc', 'NuSceneOccGS',
        'NuSceneOccMono', 'SemanticKITTIOccMono',
        'SemanticKITTI360OccMono', 'RELLIS3DWildOccMono',
        'GOOSEOccMono', 'GOOSETemporalOccMono',
    ]
