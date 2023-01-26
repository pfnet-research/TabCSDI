import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import pickle


def train(
    model,
    config,
    train_loader,
    valid_loader=None,
    valid_epoch_interval=300,
    foldername="",
):
    # Control random seed in the current script.
    torch.manual_seed(0)
    np.random.seed(0)
    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)
    if foldername != "":
        output_path = foldername + "/model.pth"

    p0 = int(0.25 * config["epochs"])
    p1 = int(0.5 * config["epochs"])
    p2 = int(0.75 * config["epochs"])
    p3 = int(0.9 * config["epochs"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p0, p1, p2, p3], gamma=0.1
    )
    # history = {'train_loss':[], 'val_rmse':[]}
    best_valid_loss = 1e10
    for epoch_no in range(config["epochs"]):
        avg_loss = 0
        model.train()
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()
                # The forward method returns loss.
                loss = model(train_batch)
                loss.backward()
                avg_loss += loss.item()
                optimizer.step()
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )
            lr_scheduler.step()

        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            print("Start validation")
            model.eval()
            avg_loss_valid = 0
            # some initial settings
            val_nsample = 15
            val_scaler = 1
            mse_total = 0
            mae_total = 0
            evalpoints_total = 0

            with torch.no_grad():
                with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as it:
                    for batch_no, valid_batch in enumerate(it, start=1):
                        output = model.evaluate(valid_batch, val_nsample)
                        # `eval_points` is `target_mask`. `observed_time` is `observed_tp`(10)
                        # `c_target` is `observed_data`
                        (
                            samples,
                            c_target,
                            eval_points,
                            observed_points,
                            observed_time,
                        ) = output
                        samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                        c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                        eval_points = eval_points.permute(0, 2, 1)
                        observed_points = observed_points.permute(0, 2, 1)

                        # take the median from samples.
                        samples_median = samples.median(dim=1)
                        mse_current = (
                            ((samples_median.values - c_target) * eval_points) ** 2
                        ) * (val_scaler**2)
                        mae_current = (
                            torch.abs((samples_median.values - c_target) * eval_points)
                        ) * val_scaler

                        mse_total += torch.sum(mse_current, dim=0)
                        evalpoints_total += torch.sum(eval_points, dim=0)

                        it.set_postfix(
                            ordered_dict={
                                "rmse_total": torch.mean(
                                    torch.sqrt(torch.div(mse_total, evalpoints_total))
                                ).item(),
                                "batch_no": batch_no,
                            },
                            refresh=True,
                        )

    if foldername != "":
        torch.save(model.state_dict(), output_path)

    # Use folloing code for saving training history.
    # with open(foldername+'/saved_history.pkl', 'wb') as f:
    #     pickle.dump(history, f)


def evaluate(model, test_loader, nsample=100, scaler=1, mean_scaler=0, foldername=""):
    # Control random seed in the current script.
    torch.manual_seed(0)
    np.random.seed(0)

    with torch.no_grad():
        model.eval()
        mse_total = 0
        mae_total = 0
        evalpoints_total = 0

        all_target = []
        all_observed_point = []
        all_observed_time = []
        all_evalpoint = []
        all_generated_samples = []
        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, test_batch in enumerate(it, start=1):
                output = model.evaluate(test_batch, nsample)

                samples, c_target, eval_points, observed_points, observed_time = output
                samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                eval_points = eval_points.permute(0, 2, 1)
                observed_points = observed_points.permute(0, 2, 1)

                # take the median from samples.
                samples_median = samples.median(dim=1)

                all_target.append(c_target)
                all_evalpoint.append(eval_points)
                all_observed_point.append(observed_points)
                all_observed_time.append(observed_time)
                all_generated_samples.append(samples)

                mse_current = (
                    ((samples_median.values - c_target) * eval_points) ** 2
                ) * (scaler**2)
                mae_current = (
                    torch.abs((samples_median.values - c_target) * eval_points)
                ) * scaler
                mse_total += torch.sum(mse_current, dim=0)
                mae_total += torch.sum(mae_current, dim=0)
                evalpoints_total += torch.sum(eval_points, dim=0)
                it.set_postfix(
                    ordered_dict={
                        "rmse_total": torch.mean(
                            torch.sqrt(torch.div(mse_total, evalpoints_total))
                        ).item(),
                        "batch_no": batch_no,
                    },
                    refresh=True,
                )

            # Use folloing code for saving generated results.
            # with open(
            #     foldername + "/generated_outputs_nsample" + str(nsample) + ".pk", "wb"
            # ) as f:
            #     all_target = torch.cat(all_target, dim=0)
            #     all_evalpoint = torch.cat(all_evalpoint, dim=0)
            #     all_observed_point = torch.cat(all_observed_point, dim=0)
            #     all_observed_time = torch.cat(all_observed_time, dim=0)
            #     all_generated_samples = torch.cat(all_generated_samples, dim=0)

            #     pickle.dump(
            #         [
            #             # shape: [len(test_dataset), nsample, L, K]]
            #             all_generated_samples,
            #             all_target,
            #             all_evalpoint,
            #             all_observed_point,
            #             all_observed_time,
            #             scaler,
            #             mean_scaler,
            #         ],
            #         f,
            #     )

            with open(foldername + "/result_nsample" + str(nsample) + ".pk", "wb") as f:
                pickle.dump(
                    [
                        torch.mean(
                            torch.sqrt(torch.div(mse_total, evalpoints_total))
                        ).item(),
                    ],
                    f,
                )
            print(
                "RMSE:",
                torch.mean(torch.sqrt(torch.div(mse_total, evalpoints_total))).item(),
            )


def evaluate_analog(
    exe_name, model, test_loader, nsample=100, scaler=1, mean_scaler=0, foldername=""
):

    if exe_name == "census":
        with open("./data_census_analog/transform.pk", "rb") as f:
            _, cont_cols, saved_cat_dict = pickle.load(f)

    torch.manual_seed(0)
    np.random.seed(0)
    with torch.no_grad():
        model.eval()
        mse_total = 0
        mae_total = 0
        err_total = np.zeros([len(saved_cat_dict)])
        err_total_eval_nums = np.zeros([len(saved_cat_dict)])
        evalpoints_total = 0

        all_target = []
        all_observed_point = []
        all_observed_time = []
        all_evalpoint = []
        all_generated_samples = []

        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, test_batch in enumerate(it, start=1):
                output = model.evaluate(test_batch, nsample)
                samples, c_target, eval_points, observed_points, observed_time = output
                samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                eval_points = eval_points.permute(0, 2, 1)
                observed_points = observed_points.permute(0, 2, 1)
                samples_median = samples.median(dim=1)  # (B, L, K)

                # Threashold
                for i in range(samples_median.values.size(dim=1)):
                    if i in cont_cols:
                        continue
                    index1 = samples_median.values[:, i, :] >= 0
                    samples_median.values[:, i, :][index1] = 1
                    index2 = samples_median.values[:, i, :] < 0
                    samples_median.values[:, i, :][index2] = -1

                all_target.append(c_target)
                all_evalpoint.append(eval_points)
                all_observed_point.append(observed_points)
                all_observed_time.append(observed_time)
                all_generated_samples.append(samples)

                # for continous variables
                mse_current = (
                    (
                        (
                            samples_median.values[:, cont_cols, :]
                            - c_target[:, cont_cols, :]
                        )
                        * eval_points[:, cont_cols, :]
                    )
                    ** 2
                ) * (scaler**2)
                mae_current = (
                    torch.abs(
                        (
                            samples_median.values[:, cont_cols, :]
                            - c_target[:, cont_cols, :]
                        )
                        * eval_points[:, cont_cols, :]
                    )
                ) * scaler

                # for categorical variables
                for i in range(len(saved_cat_dict)):
                    cate_cols = saved_cat_dict[str(i)]
                    matched_nums = (
                        (
                            (
                                samples_median.values[:, cate_cols, :]
                                == c_target[:, cate_cols, :]
                            )
                            * eval_points[:, cate_cols, :]
                        )
                        .all(1)
                        .sum()
                    )
                    eval_nums = eval_points[:, cate_cols, :].sum().sum() / len(
                        cate_cols
                    )
                    err_total[i] += eval_nums - matched_nums
                    err_total_eval_nums[i] += eval_nums

                mse_total += torch.sum(mse_current, dim=0)
                mae_total += torch.sum(mae_current, dim=0)
                evalpoints_total += torch.sum(eval_points[:, cont_cols, 0], dim=0)

                it.set_postfix(
                    ordered_dict={
                        "rmse_total": torch.mean(
                            torch.sqrt(torch.div(mse_total, evalpoints_total))
                        ).item(),
                        "batch_no": batch_no,
                    },
                    refresh=True,
                )

            with open(foldername + "/result_nsample" + str(nsample) + ".pk", "wb") as f:
                pickle.dump(
                    [
                        torch.mean(torch.sqrt(mse_total / evalpoints_total)).item(),
                        mae_total / evalpoints_total,
                        err_total / err_total_eval_nums,
                    ],
                    f,
                )
                print(
                    "RMSE:",
                    torch.mean(
                        torch.sqrt(torch.div(mse_total, evalpoints_total))
                    ).item(),
                )
                print("ERR_CATE:", err_total / err_total_eval_nums)


def evaluate_onehot(
    exe_name, model, test_loader, nsample=100, scaler=1, mean_scaler=0, foldername=""
):

    if exe_name == "census":
        with open("./data_census_onehot/transformed_columns.pk", "rb") as f:
            cont_cols, saved_cat_dict = pickle.load(f)
        print(cont_cols, saved_cat_dict)
        with open("./data_census_onehot/encoder.pk", "rb") as f:
            encoder = pickle.load(f)
    print(cont_cols, saved_cat_dict)
    torch.manual_seed(0)
    np.random.seed(0)
    with torch.no_grad():
        model.eval()
        mse_total = 0
        mae_total = 0
        err_total = np.zeros([len(saved_cat_dict)])
        err_total_eval_nums = np.zeros([len(saved_cat_dict)])
        evalpoints_total = 0

        all_target = []
        all_observed_point = []
        all_observed_time = []
        all_evalpoint = []
        all_generated_samples = []

        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, test_batch in enumerate(it, start=1):

                output = model.evaluate(test_batch, nsample)

                samples, c_target, eval_points, observed_points, observed_time = output
                samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                eval_points = eval_points.permute(0, 2, 1)
                observed_points = observed_points.permute(0, 2, 1)
                samples_median = samples.median(dim=1)  # (B, L, K)

                for i in range(c_target.shape[1]):
                    if i in cont_cols:
                        continue
                    index1 = c_target[:, i, :] >= 0
                    c_target[:, i, :][index1] = 1
                    index2 = c_target[:, i, :] < 0
                    c_target[:, i, :][index2] = 0

                all_target.append(c_target)
                all_evalpoint.append(eval_points)
                all_observed_point.append(observed_points)
                all_observed_time.append(observed_time)
                all_generated_samples.append(samples)

                # for continous variables
                mse_current = (
                    (
                        (samples_median.values[:, cont_cols] - c_target[:, cont_cols])
                        * eval_points[:, cont_cols, :]
                    )
                    ** 2
                ) * (scaler**2)
                mae_current = (
                    torch.abs(
                        (samples_median.values[:, cont_cols] - c_target[:, cont_cols])
                        * eval_points[:, cont_cols, :]
                    )
                ) * scaler

                # for categorical variables
                for index, i in enumerate(list(saved_cat_dict.keys())):
                    cate_cols = saved_cat_dict[str(i)]
                    matched_nums = (
                        (
                            torch.argmax(samples_median.values[:, cate_cols], dim=1)
                            == torch.argmax(c_target[:, cate_cols], dim=1)
                        )
                        * eval_points[:, cate_cols[0], :]
                    ).sum()
                    eval_nums = eval_points[:, cate_cols, :].sum().sum() / len(
                        cate_cols
                    )
                    eval_nums = eval_nums.cuda(matched_nums.device)
                    err_total[index] += eval_nums - matched_nums
                    err_total_eval_nums[index] += eval_nums
                mse_total += torch.sum(mse_current, dim=0)
                mae_total += torch.sum(mae_current, dim=0)
                evalpoints_total += torch.sum(eval_points[:, cont_cols, 0], dim=0)
                it.set_postfix(
                    ordered_dict={
                        "rmse_total": torch.mean(
                            torch.sqrt(torch.div(mse_total, evalpoints_total))
                        ),
                        "batch_no": batch_no,
                    },
                    refresh=True,
                )
                print(
                    np.mean(err_total / err_total_eval_nums),
                    err_total / err_total_eval_nums,
                )

            with open(foldername + "/result_nsample" + str(nsample) + ".pk", "wb") as f:
                pickle.dump(
                    [
                        torch.mean(torch.sqrt(torch.div(mse_total, evalpoints_total))),
                        err_total / err_total_eval_nums,
                    ],
                    f,
                )
                print(
                    "RMSE:",
                    torch.mean(torch.sqrt(torch.div(mse_total, evalpoints_total))),
                )
                print("ERR_CATE:", err_total / err_total_eval_nums)


def evaluate_ft(
    exe_name, model, test_loader, nsample=100, scaler=1, mean_scaler=0, foldername=""
):

    if exe_name == "census":
        with open("./data_census_ft/transformed_columns.pk", "rb") as f:
            cont_list, num_cate_list = pickle.load(f)
        with open("./data_census_ft/encoder.pk", "rb") as f:
            encoder = pickle.load(f)

    print(cont_list, num_cate_list)
    torch.manual_seed(0)
    np.random.seed(0)
    with torch.no_grad():
        model.eval()
        mse_total = 0
        mae_total = 0
        err_total = np.zeros([len(num_cate_list)])
        err_total_eval_nums = np.zeros([len(num_cate_list)])
        evalpoints_total = 0

        all_target = []
        all_observed_point = []
        all_observed_time = []
        all_evalpoint = []
        all_generated_samples = []

        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, test_batch in enumerate(it, start=1):

                output = model.evaluate(test_batch, nsample)
                samples, c_target, eval_points, observed_points, observed_time = output
                samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                eval_points = eval_points.permute(0, 2, 1)
                observed_points = observed_points.permute(0, 2, 1)

                # take the median from samples.
                samples_median = samples.median(dim=1)  # (B, L, K)

                samples_median = model.tokenizer.recover(
                    samples_median.values, len(cont_list)
                )
                c_target = model.tokenizer.recover(c_target, len(cont_list))

                all_target.append(c_target)
                all_evalpoint.append(eval_points)
                all_observed_point.append(observed_points)
                all_observed_time.append(observed_time)
                all_generated_samples.append(samples)

                # for continous variables
                mse_current = (
                    (
                        (samples_median[:, cont_list] - c_target[:, cont_list])
                        * eval_points[:, cont_list, 0]
                    )
                    ** 2
                ) * (scaler**2)
                mae_current = (
                    torch.abs(
                        (samples_median[:, cont_list] - c_target[:, cont_list])
                        * eval_points[:, cont_list, 0]
                    )
                ) * scaler

                # for categorical variables
                for i in range(len(num_cate_list)):
                    matched_nums = (
                        samples_median[:, len(cont_list) + i]
                        == c_target[:, len(cont_list) + i]
                        * eval_points[:, len(cont_list) + i, 0]
                    ).sum()
                    eval_nums = eval_points[:, len(cont_list) + i, 0].sum()
                    err_total[i] += eval_nums - matched_nums
                    err_total_eval_nums[i] += eval_nums

                mse_total += torch.sum(mse_current, dim=0)
                mae_total += torch.sum(mae_current, dim=0)
                evalpoints_total += torch.sum(eval_points[:, cont_list, 0], dim=0)

                it.set_postfix(
                    ordered_dict={
                        "rmse_total": torch.mean(
                            torch.sqrt(torch.div(mse_total, evalpoints_total))
                        ).item(),
                        "batch_no": batch_no,
                    },
                    refresh=True,
                )

            with open(foldername + "/result_nsample" + str(nsample) + ".pk", "wb") as f:
                pickle.dump(
                    [
                        torch.mean(
                            torch.sqrt(torch.div(mse_total, evalpoints_total))
                        ).item(),
                        err_total / err_total_eval_nums,
                    ],
                    f,
                )
                print(
                    "RMSE:",
                    torch.mean(
                        torch.sqrt(torch.div(mse_total, evalpoints_total))
                    ).item(),
                )
                print("ERR_CATE:", err_total / err_total_eval_nums)
