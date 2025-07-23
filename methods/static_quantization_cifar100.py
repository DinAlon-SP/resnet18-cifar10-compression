import os
import csv

import timm
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from methods.static_quantization_cifar10 import static_GPU_quantization
from utils.model_utils import  path_to_save, get_model_info_extended
import detectors

if __name__ == "__main__":
    # ---  Load CIFAR-100 test set ---
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
    ])

    test_set = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = timm.create_model("resnet18_cifar100", pretrained=True).to(device)
    model.eval()

    info_fp32 = get_model_info_extended(model, test_loader)
    quantization_log_rows = []

    # Run TensorRT FP32
    acc_trt_fp32, time_trt_fp32, size_trt_fp32, fmt_trt_fp32 = static_GPU_quantization(model, test_set,
                                                                                       quantization_format="fp32",
                                                                                       num_classes=100,samples_per_class=1)
    quantization_log_rows.append([
        fmt_trt_fp32.upper() + " (TRT)",
        f"{info_fp32['model_size_MB']:.2f}",
        f"{info_fp32['trainable_parameters'] / 1e6:.2f}",
        f"{time_trt_fp32:.2f}",
        f"{acc_trt_fp32:.2f}"
    ])

    # --- Quantize to FP16 ---
    acc_fp16, time_fp16, size_fp16, fmt_fp16 = static_GPU_quantization(model, test_set,
                                                                       quantization_format="fp16",
                                                                       num_classes=100,samples_per_class=1)
    quantization_log_rows.append([
        fmt_fp16.upper(),
        f"{size_fp16:.2f}",
        f"{info_fp32['trainable_parameters'] / 1e6:.2f}",
        f"{time_fp16:.2f}",
        f"{acc_fp16:.2f}"
    ])

    # ---  Quantize to INT8 ---
    acc_int8, time_int8, size_int8, fmt_int8 = static_GPU_quantization(model, test_set,
                                                                       quantization_format="int8",
                                                                       num_classes=100,samples_per_class=1)

    quantization_log_rows.append([
        fmt_int8.upper(),
        f"{size_int8:.2f}",
        f"{info_fp32['trainable_parameters'] / 1e6:.2f}",
        f"{time_int8:.2f}",
        f"{acc_int8:.2f}"
    ])

    # --- 6. Print table ---
    quant_headers = ["Format", "Model Size (MB)", "Trainable Parameters (M)", "Inference Time (ms/image)", "Test Accuracy (%)"]
    print(f"\n Quantization Summary:")
    print(f"{quant_headers[0]:<10} | {quant_headers[1]:<17} | {quant_headers[2]:<25} | {quant_headers[3]:<25} | {quant_headers[4]}")
    print("-" * 100)
    for row in quantization_log_rows:
        print(f"{row[0]:<10} | {row[1]:<17} | {row[2]:<25} | {row[3]:<25} | {row[4]}")

    # === Save as CSV ===
    quant_csv_path = path_to_save("quantization", "CIFAR-100_quantization_summary_table.csv")
    os.makedirs(os.path.dirname(quant_csv_path), exist_ok=True)
    with open(quant_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(quant_headers)
        writer.writerows(quantization_log_rows)
    print(f" Saved quantization summary to: {quant_csv_path}")

    # === Plot Table as Image ===
    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=quantization_log_rows, colLabels=quant_headers, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.auto_set_column_width(col=list(range(len(quant_headers))))

    quant_plot_path = path_to_save("quantization", "CIFAR-100_quantization_summary_table.png")
    plt.title("CIFAR-100 Quantization Results Summary Table", fontsize=12, weight='bold')
    plt.savefig(quant_plot_path, bbox_inches="tight")
    plt.show()

