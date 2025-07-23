import csv
import os

import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from data.prepare_data import get_cifar10_datasets

from methods.static_quantization_cifar10 import static_GPU_quantization
from utils.model_utils import get_model_info_extended, path_to_save

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    quantization_log_rows = []
    model_fp32 = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True).to(device)

    _, _, test_set, _, _ = get_cifar10_datasets(seed=42, from_disk=True)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)
    info_fp32 = get_model_info_extended(model_fp32, test_loader)

    # Run TensorRT FP32
    acc_trt_fp32, time_trt_fp32, size_trt_fp32, fmt_trt_fp32 = static_GPU_quantization(
        model_fp32, test_set, quantization_format="fp32")
    quantization_log_rows.append([
        fmt_trt_fp32.upper() + " (TRT)",
        f"{size_trt_fp32:.2f}",
        f"{info_fp32['trainable_parameters'] / 1e3:.2f}",  # same parameters
        f"{time_trt_fp32:.2f}",
        f"{acc_trt_fp32:.2f}"
    ])

    # Run quantized FP16
    acc_fp16, time_fp16, size_fp16, fmt_fp16 = static_GPU_quantization(model_fp32, test_set, quantization_format="fp16")
    quantization_log_rows.append([
        fmt_fp16.upper(),
        f"{size_fp16:.2f}",
        f"{info_fp32['trainable_parameters'] / 1e3:.2f}",  # same parameters
        f"{time_fp16:.2f}",
        f"{acc_fp16:.2f}"
    ])

    # Run quantized INT8
    acc_int8, time_int8, size_int8, fmt_int8 = static_GPU_quantization(model_fp32, test_set, quantization_format="int8")
    quantization_log_rows.append([
        fmt_int8.upper(),
        f"{size_int8:.2f}",
        f"{info_fp32['trainable_parameters'] / 1e3:.2f}",  # same parameters
        f"{time_int8:.2f}",
        f"{acc_int8:.2f}"
    ])

    quant_headers = ["Format", "Model Size (MB)", "Trainable Parameters (K)", "Inference Time (ms/image)",
                     "Test Accuracy (%)"]

    # === Print Table to Console ===
    print(f"\n Quantization Summary:")
    print(
        f"{quant_headers[0]:<10} | {quant_headers[1]:<17} | {quant_headers[2]:<25} | {quant_headers[3]:<25} | {quant_headers[4]}")
    print("-" * 100)
    for row in quantization_log_rows:
        print(f"{row[0]:<10} | {row[1]:<17} | {row[2]:<25} | {row[3]:<25} | {row[4]}")

    # === Save as CSV ===
    quant_csv_path = path_to_save("resnet20/quantization", "ResNet-20_CIFAR-10_quantization_summary_table.csv")
    os.makedirs(os.path.dirname(quant_csv_path), exist_ok=True)
    with open(quant_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(quant_headers)
        writer.writerows(quantization_log_rows)
    print(f"✅ Saved quantization summary to: {quant_csv_path}")

    # === Plot Table as Image ===
    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=quantization_log_rows, colLabels=quant_headers, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.auto_set_column_width(col=list(range(len(quant_headers))))

    quant_plot_path = path_to_save("resnet20/quantization", "ResNet-20_CIFAR-10_quantization_summary_table.png")
    plt.title("ResNet-20 CIFAR-10 Quantization Results Summary Table", fontsize=12, weight='bold')
    plt.savefig(quant_plot_path, bbox_inches="tight")
    plt.show()
