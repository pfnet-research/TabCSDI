import pickle
import yaml
import os
import re
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset


def process_func(path: str, aug_rate=1, missing_ratio=0.1):
    data = pd.read_csv(path, header=None).iloc[:, 1:]
    data.replace("?", np.nan, inplace=True)
    data_aug = pd.concat([data] * aug_rate)

    observed_values = data_aug.values.astype("float32")
    observed_masks = ~np.isnan(observed_values)

    masks = observed_masks.copy()
    # for each column, mask {missing_ratio} % of observed values.
    for col in range(observed_values.shape[1]):  # col #
        obs_indices = np.where(masks[:, col])[0]
        miss_indices = np.random.choice(
            obs_indices, (int)(len(obs_indices) * missing_ratio), replace=False
        )
        masks[miss_indices, col] = False
    # gt_mask: 0 for missing elements and manully maksed elements
    gt_masks = masks.reshape(observed_masks.shape)

    observed_values = np.nan_to_num(observed_values)
    observed_masks = observed_masks.astype(int)
    gt_masks = gt_masks.astype(int)

    return observed_values, observed_masks, gt_masks


class tabular_dataset(Dataset):
    # eval_length should be equal to attributes number.
    def __init__(
        self, eval_length=10, use_index_list=None, aug_rate=1, missing_ratio=0.1, seed=0
    ):
        self.eval_length = eval_length
        np.random.seed(seed)

        dataset_path = "./data_breast/breast-cancer-wisconsin.data"
        processed_data_path = (
            f"./data_breast/missing_ratio-{missing_ratio}_seed-{seed}.pk"
        )
        processed_data_path_norm = (
            f"./data_breast/missing_ratio-{missing_ratio}_seed-{seed}_max-min_norm.pk"
        )

        if not os.path.isfile(processed_data_path):
            self.observed_values, self.observed_masks, self.gt_masks = process_func(
                dataset_path, aug_rate=aug_rate, missing_ratio=missing_ratio
            )

            with open(processed_data_path, "wb") as f:
                pickle.dump(
                    [self.observed_values, self.observed_masks, self.gt_masks], f
                )
            print("--------Dataset created--------")

        elif os.path.isfile(processed_data_path_norm):
            with open(processed_data_path_norm, "rb") as f:
                self.observed_values, self.observed_masks, self.gt_masks = pickle.load(
                    f
                )
            print("--------Normalized dataset loaded--------")

        if use_index_list is None:
            self.use_index_list = np.arange(len(self.observed_values))
        else:
            self.use_index_list = use_index_list

    def __getitem__(self, org_index):
        index = self.use_index_list[org_index]
        s = {
            "observed_data": self.observed_values[index],
            "observed_mask": self.observed_masks[index],
            "gt_mask": self.gt_masks[index],
            "timepoints": np.arange(self.eval_length),
        }
        return s

    def __len__(self):
        return len(self.use_index_list)


def get_dataloader(seed=1, nfold=5, batch_size=16, missing_ratio=0.1):
    dataset = tabular_dataset(missing_ratio=missing_ratio, seed=seed)
    print(f"Dataset size:{len(dataset)} entries")

    indlist = np.arange(len(dataset))

    np.random.seed(seed + 1)
    np.random.shuffle(indlist)

    tmp_ratio = 1 / nfold
    start = (int)((nfold - 1) * len(dataset) * tmp_ratio)
    end = (int)(nfold * len(dataset) * tmp_ratio)

    test_index = indlist[start:end]
    remain_index = np.delete(indlist, np.arange(start, end))

    np.random.shuffle(remain_index)
    num_train = (int)(len(remain_index) * 1)
    train_index = remain_index[:num_train]
    valid_index = remain_index[num_train:]

    # Here we perform max-min normalization.
    processed_data_path_norm = (
        f"./data_breast/missing_ratio-{missing_ratio}_seed-{seed}_max-min_norm.pk"
    )
    if not os.path.isfile(processed_data_path_norm):
        print(
            "--------------Dataset has not been normalized yet. Perform data normalization and store the mean value of each column.--------------"
        )
        # data transformation after train-test split.
        col_num = dataset.observed_values.shape[1]
        max_arr = np.zeros(col_num)
        min_arr = np.zeros(col_num)
        mean_arr = np.zeros(col_num)
        for k in range(col_num):
            # Using observed_mask to avoid counting missing values.
            obs_ind = dataset.observed_masks[train_index, k].astype(bool)
            temp = dataset.observed_values[train_index, k]
            max_arr[k] = max(temp[obs_ind])
            min_arr[k] = min(temp[obs_ind])
        print(f"--------------Max-value for each column {max_arr}--------------")
        print(f"--------------Min-value for each column {min_arr}--------------")

        dataset.observed_values = (
            (dataset.observed_values - 0 + 1) / (max_arr - 0 + 1)
        ) * dataset.observed_masks

        with open(processed_data_path_norm, "wb") as f:
            pickle.dump(
                [dataset.observed_values, dataset.observed_masks, dataset.gt_masks], f
            )

    # Create datasets and corresponding data loaders objects.
    train_dataset = tabular_dataset(
        use_index_list=train_index, missing_ratio=missing_ratio, seed=seed
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=1)

    valid_dataset = tabular_dataset(
        use_index_list=valid_index, missing_ratio=missing_ratio, seed=seed
    )
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=0)

    test_dataset = tabular_dataset(
        use_index_list=test_index, missing_ratio=missing_ratio, seed=seed
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=0)

    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(valid_dataset)}")
    print(f"Testing dataset size: {len(test_dataset)}")

    return train_loader, valid_loader, test_loader
