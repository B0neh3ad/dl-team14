import random
import numpy as np
from itertools import product, islice, permutations, chain

import tqdm

class DataAugmentation():
    def __init__(self, flip, n_rot90):
        self.flip = flip
        self.n_rot90 = n_rot90

    def augment_task(self, task):
        """
        Augment the task by flipping and rotating the grids.
        :param task (dict): The task containing training and test samples.
        :return (dict): The augmented task with flipped and rotated grids.
        """
        augmented_task = dict()
        for partition, samples in task.items():
            augmented_task[partition] = [{name:self.augment_grid(grid) for name,grid in sample.items()} for sample in samples]
        return augmented_task

    def augment_grid(self, grid):
        """
        Augment the grid by flipping and rotating.
        :param grid (List[List[int]]): The grid to augment.
        :return (List[List[int]]): The augmented grid.
        """
        grid = np.array(grid)
        if self.flip:
            grid = np.flip(grid, axis=1)
        grid = np.rot90(grid, k=self.n_rot90)
        return grid.tolist()

    def revert_augmentation(self, grid):
        """
        Revert the augmentation by flipping and rotating back.
        :param grid (List[List[int]]): The augmented grid to revert.
        :return (List[List[int]]): The reverted grid.
        """
        grid = np.array(grid)
        grid = np.rot90(grid, k=-self.n_rot90)
        if self.flip:
            grid = np.flip(grid, axis=1)
        return grid.tolist()

def swap_one_train_and_test_sample(task):
    """
    Swap one training sample with one test sample in the task.
    :param task (dict): The task containing training and test samples.
    :return (list): A list of augmented tasks with swapped samples.
    """
    augmented_tasks = [task]
    for train_idx, train_sample in enumerate(task['train']):
        for test_idx, test_sample in enumerate(task['test']):
            augmented_task = dict()
            augmented_task['train'] = task['train'][:train_idx] + [test_sample] + task['train'][train_idx+1:]
            augmented_task['test'] = task['test'][:test_idx] + [train_sample] + task['test'][test_idx+1:]
            augmented_tasks.append(augmented_task)
    return augmented_tasks

def swap_task_colors(task, change_background_probability=0.1):
    """
    Swap colors in the task grids.
    :param task (dict): The task containing training and test samples.
    :param change_background_probability (float): Probability of changing the background color.
    :return (dict): The task with swapped colors.
    """
    colors = list(range(10))
    if random.random() < change_background_probability:
        new_colors = list(range(10))
        random.shuffle(new_colors)
    else:
        new_colors = list(range(1, 10))
        random.shuffle(new_colors)
        new_colors = [0] + new_colors

    color_map = {x: y for x, y in zip(colors, new_colors)}
    vectorized_mapping = np.vectorize(color_map.get)

    new_task = dict()
    for key in task.keys():
        new_task[key] = [{name:vectorized_mapping(grid) for name, grid in sample.items()} for sample in task[key]]
    return new_task

def permute_train_samples(task, max_permutations=6):
    """
    Permute the training samples in the task.
    :param task (dict): The task containing training and test samples.
    :param max_permutations (int): The maximum number of permutations to apply.
    :return (list): A list of augmented tasks with permuted training samples.
    """
    augmented_tasks = []
    for _ in range(max_permutations):
        train_order = np.arange(len(task['train']))
        np.random.shuffle(train_order)
        augmented_task = dict()
        augmented_task['train'] = [task['train'][idx] for idx in train_order]
        augmented_task['test'] = task['test']
        augmented_tasks.append(augmented_task)
    return augmented_tasks

def apply_geometric_augmentations(task, n_augmentations=8):
    """
    Apply geometric augmentations to the task.
    :param task (dict): The task containing training and test samples.
    :param n_augmentations (int): The number of augmentations to apply.
    :return (list): A list of augmented tasks with geometric transformations.
    """
    augmented_tasks = []
    data_augmentation_params = product([False, True], [0, 1, 2, 3])
    if n_augmentations < 8:
        data_augmentation_params = list(data_augmentation_params)
        indices = np.random.choice(np.arange(len(data_augmentation_params)), n_augmentations, replace=False)
        data_augmentation_params = [data_augmentation_params[idx] for idx in indices]
    for flip, n_rot90 in data_augmentation_params:
        data_augmentation = DataAugmentation(flip, n_rot90)
        augmented_task = data_augmentation.augment_task(task)
        augmented_tasks.append(augmented_task)
    return augmented_tasks

def apply_all_data_augmentations(tasks, args):
    """
    Apply all data augmentations to the tasks.
    :param tasks (list): A list of tasks to augment.
    :param args (argparse.Namespace): The arguments containing augmentation settings.
    :return (list): A list of augmented tasks.
    """
    print('Applying all data augmentations, initial number of tasks is', len(tasks))
    augmented_tasks = tasks
    if args.geometric_transforms:
        augmented_tasks = list(chain(*[apply_geometric_augmentations(task, args.geometric_transforms) for task in tqdm(augmented_tasks, desc='geometric augmentations')]))
        print(f'After applying geometric augmentations there are {len(augmented_tasks)} tasks')
    if args.swap_train_and_test:
        augmented_tasks = list(chain(*[swap_one_train_and_test_sample(task) for task in tqdm(augmented_tasks, desc='swap train and test')]))
        print(f'After swapping train and test samples there are {len(augmented_tasks)} tasks')
    if args.max_train_permutations:
        augmented_tasks = list(chain(*[permute_train_samples(task, max_permutations=args.max_train_permutations) for task in tqdm(augmented_tasks, desc='permute train samples')]))
        print(f'After permuting train samples there are {len(augmented_tasks)} tasks')
    if args.color_swaps:
        if args.preserve_original_colors:
            augmented_tasks.extend([swap_task_colors(task) for task in tqdm(augmented_tasks*args.color_swaps, desc='swap colors')])
        else:
            augmented_tasks = [swap_task_colors(task) for task in tqdm(augmented_tasks*args.color_swaps, desc='swap colors')]
        print(f'After swapping colors there are {len(augmented_tasks)} tasks')
    return augmented_tasks


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib import colors

    def plot_grid(grid):
        grid = np.array(grid)
        cmap = colors.ListedColormap(
            ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
            '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
        norm = colors.Normalize(vmin=0, vmax=9)
        plt.imshow(grid, cmap=cmap, norm=norm)
        plt.grid(True,which='both',color='lightgrey', linewidth=0.5)
        plt.xticks(np.arange(-0.5, grid.shape[1]), [])
        plt.yticks(np.arange(-0.5, grid.shape[0]), [])
        plt.xlim(-0.5, grid.shape[1]-0.5)

        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                plt.text(j, i, grid[i, j], ha='center', va='center')

    def plot_task(task):
        all_samples = task['train'] + task['test']
        for plot_idx, sample in enumerate(all_samples):
            plt.subplot(1, len(all_samples), plot_idx+1)
            plot_grid(sample['input'])
            if plot_idx < len(task['train']):
                plt.title(f'train {plot_idx}')
            else:
                plt.title(f'test {plot_idx-len(task["train"])}')
        plt.suptitle('Inputs for task')
        plt.show()
        for plot_idx, sample in enumerate(all_samples):
            plt.subplot(1, len(all_samples), plot_idx+1)
            plot_grid(sample['output'])
            if plot_idx < len(task['train']):
                plt.title(f'train {plot_idx}')
            else:
                plt.title(f'test {plot_idx-len(task["train"])}')
        plt.suptitle('Outputs for task')
        plt.show()

    sample_grid = np.eye(3, dtype=int).tolist()
    for flip in [True, False]:
        for n_rot90 in range(4):
            data_augmentation = DataAugmentation(flip, n_rot90)
            assert sample_grid == data_augmentation.revert_augmentation(data_augmentation.augment_grid(sample_grid))