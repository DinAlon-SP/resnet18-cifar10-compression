import os
import csv

import torch
import torch_pruning as tp
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from methods.static_quantization_cifar10 import static_GPU_quantization
from data.prepare_data import get_cifar10_datasets
from utils.model_utils import get_model_info_extended, path_to_save
from utils.base_model import ResNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    _, _, test_set, _, _ = get_cifar10_datasets(seed=42, from_disk=True)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)
    example_inputs = torch.randn(1, 3, 32, 32).to(device)

    quantization_log_rows = []

    for i in range(10, 16):  # 50% to 75%
        amount = i * 0.05
        pruning_percent = int(amount * 100)

        model_fp32_pruned = ResNet(num_classes=10).to(device)

        # Prune it
        DG = tp.DependencyGraph().build_dependency(model_fp32_pruned, example_inputs=example_inputs)
        pruner = tp.pruner.MagnitudePruner(
            model_fp32_pruned,
            example_inputs,
            importance=tp.importance.MagnitudeImportance(),
            pruning_ratio=amount,
            ignored_layers=[],
            root_module_types=[torch.nn.Conv2d],
        )
        pruner.step()

        # Load the matching pruned weights
        model_fp32_pruned.load_state_dict(
            torch.load(f"../results/pruning/{pruning_percent}_percent/best_model.pth", map_location=device)
        )
        model_fp32_pruned.eval()

        # === Evaluate FP32
        info_fp32 = get_model_info_extended(model_fp32_pruned, test_loader)
        quantization_log_rows.append([
            f"{pruning_percent}%", "FP32",
            f"{info_fp32['model_size_MB']:.2f}",
            f"{info_fp32['trainable_parameters'] / 1e6:.2f}",
            f"{info_fp32['inference_time_ms_per_image']:.2f}",
            f"{info_fp32['test_accuracy']:.2f}"
        ])

        # === FP16 Quantization
        acc_fp16, time_fp16, size_fp16, fmt_fp16 = static_GPU_quantization(model_fp32_pruned, test_set, "fp16")
        quantization_log_rows.append([
            f"{pruning_percent}%", fmt_fp16.upper(),
            f"{size_fp16:.2f}",
            f"{info_fp32['trainable_parameters'] / 1e6:.2f}",
            f"{time_fp16:.2f}",
            f"{acc_fp16:.2f}"
        ])

        # === INT8 Quantization
        acc_int8, time_int8, size_int8, fmt_int8 = static_GPU_quantization(model_fp32_pruned, test_set, "int8")
        quantization_log_rows.append([
            f"{pruning_percent}%", fmt_int8.upper(),
            f"{size_int8:.2f}",
            f"{info_fp32['trainable_parameters'] / 1e6:.2f}",
            f"{time_int8:.2f}",
            f"{acc_int8:.2f}"
        ])

    # === Format + Save Table ===
    quant_headers = ["Pruning %", "Format", "Model Size (MB)", "Trainable Parameters (M)",
                     "Inference Time (ms/image)", "Test Accuracy (%)"]

    print(f"\n Combined Quantization Summary:")
    print(f"{quant_headers[0]:<11} | {quant_headers[1]:<8} | {quant_headers[2]:<17} | "
          f"{quant_headers[3]:<25} | {quant_headers[4]:<25} | {quant_headers[5]}")
    print("-" * 120)
    for row in quantization_log_rows:
        print(f"{row[0]:<11} | {row[1]:<8} | {row[2]:<17} | {row[3]:<25} | {row[4]:<25} | {row[5]}")

    # === Save as CSV ===
    quant_csv_path = path_to_save("pruning_with_quantization", "resnet18_pruning_quantization_summary.csv")
    os.makedirs(os.path.dirname(quant_csv_path), exist_ok=True)
    with open(quant_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(quant_headers)
        writer.writerows(quantization_log_rows)
    print(f"\n Saved table to: {quant_csv_path}")

    # === Save as PNG ===
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=quantization_log_rows, colLabels=quant_headers, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.auto_set_column_width(col=list(range(len(quant_headers))))

    quant_plot_path = path_to_save("pruning_with_quantization", "resnet18_pruning_quantization_summary_table.png")
    plt.title("ResNet-18 Pruning + Quantization Summary", fontsize=12, weight='bold')
    plt.savefig(quant_plot_path, bbox_inches="tight")
    plt.show()