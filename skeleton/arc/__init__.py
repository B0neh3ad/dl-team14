from .arc import ARCSolver
from .augmentation import (
    DataAugmentation,
    swap_one_train_and_test_sample,
    swap_task_colors,
    permute_train_samples,
    apply_geometric_augmentations,
    apply_all_data_augmentations,
)
from .utils import (
    render_grid,
    make_rich_lines,
)