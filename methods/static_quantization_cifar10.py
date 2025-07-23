import csv
import os
from collections import defaultdict

import numpy as np
import cupy as cp
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader
import tensorrt as trt

from data.prepare_data import get_cifar10_datasets
from utils.base_model import ResNet
from utils.model_utils import get_model_info_extended, path_to_save, measure_inference_time


def static_GPU_quantization(model_ext, test_set_ext, quantization_format="fp16", samples_per_class=50,
                            num_classes=10):
    # Dummy calibrator for INT8
    test_loader = DataLoader(test_set_ext, batch_size=1, shuffle=False)
    if quantization_format == "int8":
        class DummyCalibrator(trt.IInt8EntropyCalibrator2):
            def __init__(self, calibration_data):
                super(DummyCalibrator, self).__init__()
                self.data = calibration_data
                self.data_index = 0
                self.device_memory = cp.cuda.Memory(self.data[0].nbytes)  # Raw memory block
                self.device_input = cp.cuda.MemoryPointer(self.device_memory, 0)

            def get_batch_size(self):
                return 1

            def get_batch(self, names):
                if self.data_index >= len(self.data):
                    return None
                batch = self.data[self.data_index]
                cp.cuda.runtime.memcpy(self.device_input.ptr,
                                       batch.ctypes.data,
                                       batch.nbytes,
                                       cp.cuda.runtime.memcpyHostToDevice)
                self.data_index += 1
                return [int(self.device_input.ptr)]

            def read_calibration_cache(self):
                return None

            def write_calibration_cache(self, cache):
                pass

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    # === 1. Export model to ONNX ===
    model_ext.to("cpu").eval()
    dummy_input = torch.randn(1, 3, 32, 32)
    onnx_path = "resnet18.onnx"
    torch.onnx.export(
        model_ext,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=13
    )

    # === 2. Build TensorRT Engine ===
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            print(" Failed parsing ONNX model")
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            return

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    if quantization_format == "fp32":
        engine_path = "resnet18_fp32.engine"  # NEW: no flags
    elif quantization_format == "fp16":
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        engine_path = "resnet18_fp16.engine"
    elif quantization_format == "int8":
        config.set_flag(trt.BuilderFlag.INT8)

        # total 10×50 = 500 samples
        class_counts = defaultdict(int)
        calibration_samples = []

        for img, label in test_loader:
            label = label.item()
            if class_counts[label] < samples_per_class:
                calibration_samples.append(img.unsqueeze(0).numpy().astype(np.float32))
                class_counts[label] += 1

            if sum(class_counts.values()) >= samples_per_class * 10:
                break

        calibrator = DummyCalibrator(calibration_samples)
        config.int8_calibrator = calibrator
        engine_path = "resnet18_int8.engine"
    else:
        raise ValueError("Unsupported format. Use 'fp16' or 'int8'.")


    serialized_engine = builder.build_serialized_network(network, config)

    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    # === 3. Inference using TensorRT engine ===
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    context = engine.create_execution_context()

    input_shape = (1, 3, 32, 32)
    output_shape = (1, num_classes)

    input_size = np.prod(input_shape) * np.dtype(np.float32).itemsize
    output_size = np.prod(output_shape) * np.dtype(np.float32).itemsize

    d_input = cp.cuda.alloc(input_size)
    d_output = cp.cuda.alloc(output_size)
    bindings = [int(d_input.ptr), int(d_output.ptr)]

    correct = 0
    total = 0

    for img, label in test_loader:
        np_input = img.numpy().astype(np.float32)
        label = label.item()

        cp.cuda.runtime.memcpy(d_input.ptr, np_input.ctypes.data, input_size, cp.cuda.runtime.memcpyHostToDevice)

        context.execute_v2(bindings)

        output_np = np.empty(output_shape, dtype=np.float32)
        cp.cuda.runtime.memcpy(output_np.ctypes.data, d_output.ptr, output_size, cp.cuda.runtime.memcpyDeviceToHost)

        pred = np.argmax(output_np)
        correct += int(pred == label)
        total += 1

    accuracy = correct / total * 100
    infer_time = measure_inference_time(test_loader, engine_type='tensorrt',
                                        context=context, bindings=bindings,
                                        d_input=d_input, d_output=d_output,
                                        input_size=input_size, output_size=output_size)
    model_size = os.path.getsize(engine_path) / (1024 * 1024)

    return accuracy, infer_time, model_size, quantization_format



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    quantization_log_rows = []

    # Evaluate full model (FP32) as baseline
    model_fp32 = ResNet(num_classes=10).to(device)
    model_fp32.load_state_dict(torch.load("../results/resnet_full/best_model.pth", map_location=device))
    _, _, test_set, _, _ = get_cifar10_datasets(seed=42, from_disk=True)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)
    info_fp32 = get_model_info_extended(model_fp32, test_loader)

    # Run TensorRT FP32
    acc_trt_fp32, time_trt_fp32, size_trt_fp32, fmt_trt_fp32 = static_GPU_quantization(model_fp32, test_set,
                                                                                       quantization_format="fp32")
    quantization_log_rows.append([
        fmt_trt_fp32.upper() + " (TRT)",
        f"{info_fp32['model_size_MB']:.2f}",
        f"{info_fp32['trainable_parameters'] / 1e6:.2f}",
        f"{time_trt_fp32:.2f}",
        f"{acc_trt_fp32:.2f}"
    ])

    # Run quantized FP16
    acc_fp16, time_fp16, size_fp16, fmt_fp16 = static_GPU_quantization(model_fp32, test_set, quantization_format="fp16")
    quantization_log_rows.append([
        fmt_fp16.upper(),
        f"{size_fp16:.2f}",
        f"{info_fp32['trainable_parameters'] / 1e6:.2f}",
        f"{time_fp16:.2f}",
        f"{acc_fp16:.2f}"
    ])

    # Run quantized INT8
    acc_int8, time_int8, size_int8, fmt_int8 = static_GPU_quantization(model_fp32, test_set, quantization_format="int8")
    quantization_log_rows.append([
        fmt_int8.upper(),
        f"{size_int8:.2f}",
        f"{info_fp32['trainable_parameters'] / 1e6:.2f}",
        f"{time_int8:.2f}",
        f"{acc_int8:.2f}"
    ])


    quant_headers = ["Format", "Model Size (MB)", "Trainable Parameters (M)", "Inference Time (ms/image)",
                     "Test Accuracy (%)"]

    # === Print Table to Console ===
    print(f"\n Quantization Summary:")
    print(
        f"{quant_headers[0]:<10} | {quant_headers[1]:<17} | {quant_headers[2]:<25} | {quant_headers[3]:<25} | {quant_headers[4]}")
    print("-" * 100)
    for row in quantization_log_rows:
        print(f"{row[0]:<10} | {row[1]:<17} | {row[2]:<25} | {row[3]:<25} | {row[4]}")

    # === Save as CSV ===
    quant_csv_path = path_to_save("quantization", "CIFAR-10_quantization_summary_table.csv")
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

    quant_plot_path = path_to_save("quantization", "CIFAR-10_quantization_summary_table.png")
    plt.title("CIFAR-10 Quantization Results Summary Table", fontsize=12, weight='bold')
    plt.savefig(quant_plot_path, bbox_inches="tight")
    plt.show()
