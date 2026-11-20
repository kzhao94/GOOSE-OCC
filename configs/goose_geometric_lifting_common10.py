"""Fixed configuration for the image-semantic + depth lifting baseline."""

import os


dataset_root = os.environ.get(
    'GOOSE_OCC_ROOT', '/path/to/GOOSE-OCC')
train_infos = os.path.join(dataset_root, 'infos', 'goose_infos_train.pkl')
val_infos = os.path.join(dataset_root, 'infos', 'goose_infos_val.pkl')
train_temporal_infos = os.path.join(
    dataset_root, 'temporal', 'temporal_infos_train.pkl')
val_temporal_infos = os.path.join(
    dataset_root, 'temporal', 'temporal_infos_val.pkl')
taxonomy_path = os.path.join(
    dataset_root, 'infos', 'goose_occ_taxonomy.json')

output_dir = 'work_dir/goose_geometric_lifting_common10'
image_size = (256, 704)
pc_range = (0.0, -16.0, -4.0, 50.0, 16.0, 4.0)
occ_size = (200, 128, 32)
voxel_size = 0.25
depth_scale = 256.0
num_classes = 8
ignore_id = 255
min_history = 10

model = 'lraspp_mobilenet_v3_large'
pretrained_backbone = True
epochs = 12
batch_size = 24
workers = 4
learning_rate = 1e-4
weight_decay = 1e-4
seed = 3407

# The segmenter is trained only on train RGB and official train 2D semantics.
# No validation labels or OCC-derived data are used for training/model selection.
training_inputs = ('train_rgb', 'train_official_2d_semantics')
prediction_inputs = ('val_rgb', 'val_metric3d_depth', 'camera_calibration')
prohibited_training_inputs = (
    'occupancy_gt', 'lidar', 'observation_masks', 'sweeps',
    'occupancy_fusion_timestamps')
