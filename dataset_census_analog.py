import pickle
import yaml
import os
import math
import re
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset


def process_func(path: str, cat_list, missing_ratio=0.1):
    def sortby(x):
        if isinstance(x, str):
            return ord(x[1])
        else:
            return x

    def check_col(x):
        if "_" in x:
            return False
        else:
            return True

    data = pd.read_csv(path, header=None)

    data.replace(0, 1, inplace=True)
    data.replace(" ?", np.nan, inplace=True)

    observed_masks = ~pd.isnull(data)
    observed_masks = observed_masks.values

    data.replace(np.nan, 0, inplace=True)

    last_bits_num = 0
    num_bits_list = []
    new_df = data.copy()
    new_df.columns = new_df.columns.astype(str)
    tot_map_dict = []
    last_bits_num = 0
    for col in cat_list:
        cat_num = data.iloc[:, col].nunique()
        bits_num = int(math.log2(cat_num)) + 1
        num_bits_list.append(bits_num)
        map_target = [i for i in range(1, cat_num + 1)]

        unique_obj = list(data.iloc[:, col].unique())
        # exclude 0
        unique_obj = [i for i in unique_obj if i != 0]

        unique_obj.sort(key=sortby)
        map_dict = {
            unique_obj[i]: bin(map_target[i])[2:].zfill(bits_num)
            for i in range(len(unique_obj))
        }

        # create key-value pair for missing values
        map_dict[0] = "0" * bits_num

        tot_map_dict.append(map_dict)
        data.iloc[:, col] = data.iloc[:, col].map(map_dict)

        unique_obj = list(data.iloc[:, col].unique())

        # new_df.drop(col+last_bits_num, inplace=True, axis=1)
        for i in range(bits_num):
            new_df.insert(
                col + i + last_bits_num, f"{col}_{i}", data.iloc[:, col].str[i]
            )
            new_df.iloc[:, col + i + last_bits_num] = new_df.iloc[
                :, col + i + last_bits_num
            ].astype(int)
            new_df.iloc[:, col + i + last_bits_num] = (
                2 * new_df.iloc[:, col + i + last_bits_num] - 1
            )
        last_bits_num += bits_num

    # remove original categorical columns
    new_df.drop(list(map(str, cat_list)), axis=1, inplace=True)

    new_observed_values = new_df.values
    masks = observed_masks.copy()
    # for each column, mask `missing_ratio` % of observed values.
    for col in range(masks.shape[1]):  # col #
        obs_indices = np.where(masks[:, col])[0]
        miss_indices = np.random.choice(
            obs_indices, (int)(len(obs_indices) * missing_ratio), replace=False
        )
        masks[miss_indices, col] = False
    # gt_mask: 0 for missing elements and manully maksed elements
    gt_masks = masks.reshape(observed_masks.shape)
    # We now need to transform these masks to the new one, suitable for mixed data types.
    cum_num_bits = 0
    new_observed_masks = observed_masks.copy()
    new_gt_masks = gt_masks.copy()

    for index, col in enumerate(cat_list):
        add_col_num = num_bits_list[index]
        insert_col_obs = observed_masks[:, col]
        insert_col_gt = gt_masks[:, col]

        for i in range(add_col_num - 1):
            new_observed_masks = np.insert(
                new_observed_masks, cum_num_bits + col, insert_col_obs, axis=1
            )
            new_gt_masks = np.insert(
                new_gt_masks, cum_num_bits + col, insert_col_gt, axis=1
            )
        cum_num_bits += add_col_num - 1

    # get columns for continous variables
    cont_cols = []
    for index, col_name in enumerate(new_df.columns):
        if check_col(col_name):
            cont_cols.append(index)

    saved_cat_cols = {}
    for index, col in enumerate(cat_list):
        indices = [
            i for i, s in enumerate(new_df.columns) if s.startswith(str(col) + "_")
        ]
        saved_cat_cols[str(index)] = indices

    with open("./data_census_analog/transform.pk", "wb") as f:
        pickle.dump([tot_map_dict, cont_cols, saved_cat_cols], f)

    # NaN is replaced by zero
    new_observed_values = np.nan_to_num(new_observed_values)
    new_observed_values = new_observed_values.astype(np.float)

    # observed_masks: 0 for missing elements
    observed_masks = observed_masks.astype(int)  # "float32"
    gt_masks = gt_masks.astype(int)

    return new_observed_values, new_observed_masks, new_gt_masks, cont_cols


class tabular_dataset(Dataset):
    # eval_length should be equal to attributes number.
    def __init__(self, eval_length=39, use_index_list=None, missing_ratio=0.1, seed=0):
        self.eval_length = eval_length
        np.random.seed(seed)

        dataset_path = "./data_census_analog/adult_trim.data"
        processed_data_path = (
            f"./data_census_analog/missing_ratio-{missing_ratio}_seed-{seed}.pk"
        )
        processed_data_path_norm = f"./data_census_analog/missing_ratio-{missing_ratio}_seed-{seed}_max-min_norm.pk"

        cat_list = [1, 3, 5, 6, 7, 8, 9, 13, 14]
        if not os.path.isfile(processed_data_path):
            (
                self.observed_values,
                self.observed_masks,
                self.gt_masks,
                self.cont_cols,
            ) = process_func(
                dataset_path, cat_list=cat_list, missing_ratio=missing_ratio
            )

            with open(processed_data_path, "wb") as f:
                pickle.dump(
                    [
                        self.observed_values,
                        self.observed_masks,
                        self.gt_masks,
                        self.cont_cols,
                    ],
                    f,
                )
            print("--------Dataset created--------")

        elif os.path.isfile(processed_data_path_norm):  # load datasetfile
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

    # 5-fold test
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
    processed_data_path_norm = f"./data_census_analog/missing_ratio-{missing_ratio}_seed-{seed}_max-min_norm.pk"
    if not os.path.isfile(processed_data_path_norm):
        print(
            "--------------Dataset has not been normalized yet. Perform data normalization and store the mean value of each column.--------------"
        )
        # Data transformation after train-test split.
        col_num = len(dataset.cont_cols)
        max_arr = np.zeros(col_num)
        min_arr = np.zeros(col_num)
        mean_arr = np.zeros(col_num)
        for index, k in enumerate(dataset.cont_cols):
            # Using observed_mask to avoid counting missing values (now represented as 0)
            obs_ind = dataset.observed_masks[train_index, k].astype(bool)
            temp = dataset.observed_values[train_index, k]
            max_arr[index] = max(temp[obs_ind])
            min_arr[index] = min(temp[obs_ind])
        print(
            f"--------------Max-value for cont-variable column {max_arr}--------------"
        )
        print(
            f"--------------Min-value for cont-variable column {min_arr}--------------"
        )

        for index, k in enumerate(dataset.cont_cols):
            dataset.observed_values[:, k] = (
                (dataset.observed_values[:, k] - (min_arr[index] - 1))
                / (max_arr[index] - min_arr[index] + 1)
            ) * dataset.observed_masks[:, k]

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
