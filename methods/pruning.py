import os
import csv

import torch
import torch_pruning as tp
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from utils.model_utils import get_model_info_extended ,path_to_save
from utils.base_model import ResNet
from data.prepare_data import get_cifar10_datasets
from utils.base_train import train


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":

    train_set, val_set, test_set, class_names, full_labels = get_cifar10_datasets(
        seed=42, from_disk=True
    )

    example_inputs = torch.randn(1, 3, 32, 32).to(device)
    log_rows = []
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

    for i in range(0, 19):
        amount = i * 0.05
        model_pruned = ResNet(num_classes=10).to(device)
        model_pruned.load_state_dict(torch.load("../results/resnet_full/best_model.pth", map_location=device))

        # Build dependency graph
        DG = tp.DependencyGraph().build_dependency(model_pruned, example_inputs=example_inputs)

        pruner = tp.pruner.MagnitudePruner(
            model_pruned,
            example_inputs,
            importance=tp.importance.MagnitudeImportance(),
            pruning_ratio=amount,
            ignored_layers=[],
            root_module_types=[torch.nn.Conv2d],
        )

        pruner.step()

        print(f"currently pruning {int(amount * 100)}% of the weights:\n")
        info = get_model_info_extended(model_pruned, test_loader, train_time_sec=None)
        model_size_MB = info["model_size_MB"]
        trainable_parameters = info["trainable_parameters"]
        test_accuracy_before = info["test_accuracy"]
        infer_time = info["inference_time_ms_per_image"]

        if i == 0:
            epochs = 1
        elif i <= 10:
            epochs = 3
        else:
            epochs = 10


        def get_lr_for_pruning_amount(amount):
            thresholds = [
                (0.00, 1e-5),
                (0.30, 5e-5),
                (0.55, 1e-4),
                (0.65, 1e-3),
                (1.00, 1e-2),
            ]
            for threshold, lr in thresholds:
                if amount <= threshold:
                    return lr
            return thresholds[-1][1]  # fallback in case amount >= 1.0


        lr = get_lr_for_pruning_amount(amount)

        model_pruned_retrained, avg_time,_,_ = train(model=model_pruned,
                                                    train_set=train_set,
                                                    val_set=val_set,
                                                    test_set=test_set,
                                                    run_name = f"pruning/{int(100 * amount)}_percent",
                                                    epochs=epochs,
                                                    batch_size=256,
                                                    optimizer_name="SGD",
                                                    lr=lr,
                                                    momentum=0.9,
                                                    weight_decay=1e-2,
                                                    loss_name='CrossEntropy',
                                                    label_smoothing=0.0,
                                                    l1_lambda=0.0,
                                                    use_scheduler=True,
                                                    amp_mode="none"
                                                    )

        info_after = get_model_info_extended(model_pruned_retrained, test_loader, train_time_sec=avg_time)
        test_accuracy_after = info_after["test_accuracy"]

        log_rows.append([
            int(amount * 100),
            f"{model_size_MB:.2f}",
            f"{trainable_parameters / 1e6:.2f}",
            f"{infer_time:.2f}",
            f"{test_accuracy_before:.2f}",
            f"{test_accuracy_after:.2f}",
            f"{lr:.5f}",
            f"{epochs}",
            f"{avg_time:.2f}",
        ])

    headers = ["Pruning %", "Size (MB)", "Parameters (M)", "Inference Time (ms/image)",
               "Accuracy Before Fine-Tuning (%)", "Accuracy After Fine-Tuning (%)", "Learning Rate", "Epochs",
               "Epoch Time (s)"]

    csv_path = path_to_save("pruning", "pruning_results_summary_table.csv")

    # Save the header + log rows to CSV
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(log_rows)

    print(f" Saved pruning summary to: {csv_path}")
    print(f"\n")

    # === TABLE FIGURE ===

    fig, ax = plt.subplots(figsize=(10, 6))  # slightly smaller
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=log_rows, colLabels=headers, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.auto_set_column_width(col=list(range(len(headers))))

    # Paths
    script_path = os.path.abspath(__file__)
    parent_dir = os.path.dirname(os.path.dirname(script_path))
    results_dir = os.path.join(parent_dir, "results")
    output_dir = os.path.join(results_dir, "pruning")
    os.makedirs(output_dir, exist_ok=True)

    plot_path1 = path_to_save("pruning", "pruning_results_summary_table.png")
    plt.title("Pruning Results Summary Table", fontsize=12, weight='bold')
    plt.savefig(plot_path1, bbox_inches="tight")
    plt.show()

    # === CONVERT DATA ===
    pruning_percentages = [row[0] for row in log_rows]
    model_sizes = [float(row[1]) for row in log_rows]
    trainable_params = [row[2] for row in log_rows]
    inference_times = [float(row[3]) for row in log_rows]
    test_acc_before = [float(row[4]) for row in log_rows]
    test_acc_after = [float(row[5]) for row in log_rows]

    # === PLOT 2: Accuracy vs Pruning % ===
    plt.figure(figsize=(8, 5))
    plt.plot(pruning_percentages, test_acc_before, marker='o', label='Before Fine-Tuning')
    plt.plot(pruning_percentages, test_acc_after, marker='o', label='After Fine-Tuning')
    plt.title('Test Accuracy vs Pruning %')
    plt.xlabel('Pruning %')
    plt.ylabel('Test Accuracy')
    plt.legend()
    plt.grid(True)

    plot_path_acc = path_to_save("pruning", "accuracy_vs_pruning.png")
    plt.savefig(plot_path_acc, bbox_inches="tight")
    plt.show()

    # === PLOT 3: Trainable Params vs Pruning % ===
    trainable_params = [float(row[2]) for row in log_rows]
    plt.figure(figsize=(8, 5))
    plt.plot(pruning_percentages, trainable_params, marker='o', color='green')
    plt.title('Trainable Parameters vs Pruning %')
    plt.xlabel('Pruning %')
    plt.ylabel('Trainable Parameters')
    plt.yscale('log')
    plt.grid(True)

    plot_path_params = path_to_save("pruning", "params_vs_pruning.png")
    plt.savefig(plot_path_params, bbox_inches="tight")
    plt.show()
