import os

from .loading_goose_mono import LoadGOOSEImageAndCalib, LoadGOOSE2DSupervision, LoadGOOSEOccGT

if os.environ.get('VISIBILITYOCC_GOOSE_ONLY') == '1':
    __all__ = [
        'LoadGOOSEImageAndCalib', 'LoadGOOSE2DSupervision',
        'LoadGOOSEOccGT',
    ]
else:
    from .loading import LoadMultiViewImageFromMultiSweeps, LoadOccGTFromFile
    from .transforms import PadMultiViewImage, NormalizeMultiviewImage, PhotoMetricDistortionMultiViewImage
    from .loading_kitti_mono import LoadKittiImageAndCalib, LoadKittiOccGT
    from .loading_nuscenes_mono import LoadFrontViewImage
    from .loading_kitti360_mono import LoadKITTI360ImageAndCalib, LoadKITTI360OccGT, LoadKITTI3602DPseudoLabels, RandomTransformImageMonoKITTI360
    from .loading_rellis3d_mono import LoadRELLIS3DImageAndCalib, LoadRELLIS3DOccGT, LoadRELLIS3D2DPseudoLabels, RandomTransformImageMonoRELLIS3D

    __all__ = [
        'LoadMultiViewImageFromMultiSweeps', 'PadMultiViewImage',
        'NormalizeMultiviewImage', 'PhotoMetricDistortionMultiViewImage',
        'LoadOccGTFromFile', 'LoadKittiImageAndCalib', 'LoadKittiOccGT',
        'LoadFrontViewImage',
        'LoadKITTI360ImageAndCalib', 'LoadKITTI360OccGT',
        'LoadKITTI3602DPseudoLabels', 'RandomTransformImageMonoKITTI360',
        'LoadRELLIS3DImageAndCalib', 'LoadRELLIS3DOccGT',
        'LoadRELLIS3D2DPseudoLabels', 'RandomTransformImageMonoRELLIS3D',
        'LoadGOOSEImageAndCalib', 'LoadGOOSE2DSupervision',
        'LoadGOOSEOccGT',
    ]
