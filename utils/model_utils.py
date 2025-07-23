import os
import time

import torch
import cupy as cp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def get_model_info_extended(model, test_loader, train_time_sec=None):
    device = next(model.parameters()).device  # get model device

    # 1. Precision type
    dtype = next(model.parameters()).dtype
    dtype_map = {
        torch.float32: "FP32 (float32)",
        torch.float64: "FP64 (float64)",
        torch.float16: "FP16 (float16)",
        torch.bfloat16: "BF16 (bfloat16)",
        torch.int8: "INT8",
        torch.int4 if hasattr(torch, "int4") else None: "INT4",
        torch.float8_e4m3fn if hasattr(torch, "float8_e4m3fn") else None: "FP8 (e4m3fn)",
        torch.float8_e5m2 if hasattr(torch, "float8_e5m2") else None: "FP8 (e5m2)",
    }
    precision = dtype_map.get(dtype, str(dtype))

    # 2. Model size and parameters
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 3. Accuracy on test set (GPU-only)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    accuracy = 100 * correct / total

    # 4. Pure GPU Inference Time
    model.eval()
    avg_infer_time_ms = measure_inference_time(test_loader, engine_type='pytorch', model=model)

    # 5. Format train time
    if train_time_sec is not None:
        train_time_str = f"{train_time_sec:.2f} sec" if train_time_sec < 60 else f"{train_time_sec / 60:.2f} min"
    else:
        train_time_str = "N/A"

    #  Print summary
    print(f"Precision Type:     {precision}")
    print(f"Model Size:         {size_all_mb:.2f} MB")
    print(f"Trainable Params:   {num_trainable_params:,}")
    print(f"Test Accuracy:      {accuracy:.2f}%")
    print(f"Inference Time:     {avg_infer_time_ms:.4f} ms/image (GPU)")
    print(f"Training Time:      {train_time_str}")

    return {
        "precision": precision,
        "model_size_MB": round(size_all_mb, 2),
        "trainable_parameters": num_trainable_params,
        "test_accuracy": round(accuracy, 2),
        "inference_time_ms_per_image": round(avg_infer_time_ms, 2),
        "train_time_sec": train_time_sec
    }
# evaluate the model on the loader set(train / validation / test)
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs.to(device))
            _, predicted = torch.max(outputs, 1)
            correct += (predicted.cpu() == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total

def path_to_save(folder,run_name):
    script_path = os.path.abspath(__file__)
    parent_dir = os.path.dirname(os.path.dirname(script_path))
    results_dir = os.path.join(parent_dir, "results")
    output_dir = os.path.join(results_dir, folder)
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, run_name)
    return path

def measure_inference_time(test_loader, engine_type='pytorch',
                           model=None,
                           context=None, bindings=None,
                           d_input=None, d_output=None,
                           input_size=None, output_size=None):

    assert engine_type in ['pytorch', 'tensorrt'], "engine_type must be 'pytorch' or 'tensorrt'"
    total_gpu_time = 0.0
    total_wall_time = 0.0
    count = 0

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()

    if engine_type == 'pytorch':
        assert model is not None, "Must provide PyTorch model"
        model.eval()
        with torch.no_grad():
            for img, _ in test_loader:
                img = img.to('cuda')
                if img.dim() == 3:
                    img = img.unsqueeze(0)

                torch.cuda.synchronize()
                t0 = time.time()
                starter.record()
                _ = model(img)
                ender.record()
                torch.cuda.synchronize()
                t1 = time.time()

                total_gpu_time += starter.elapsed_time(ender)
                total_wall_time += (t1 - t0) * 1000
                count += 1

    elif engine_type == 'tensorrt':
        assert all(x is not None for x in [context, bindings, d_input, d_output, input_size, output_size]), \
            "Must provide all TensorRT arguments"
        for img, _ in test_loader:
            np_input = img.numpy().astype(cp.float32)
            cp.cuda.runtime.memcpy(d_input.ptr, np_input.ctypes.data, input_size, cp.cuda.runtime.memcpyHostToDevice)

            torch.cuda.synchronize()
            t0 = time.time()
            starter.record()
            context.execute_v2(bindings)
            ender.record()
            torch.cuda.synchronize()
            t1 = time.time()

            total_gpu_time += starter.elapsed_time(ender)
            total_wall_time += (t1 - t0) * 1000
            count += 1

            output_buffer = cp.empty(output_size // 4, dtype=cp.float32)
            cp.cuda.runtime.memcpy(output_buffer.data.ptr, d_output.ptr, output_size,
                                   cp.cuda.runtime.memcpyDeviceToHost)

    avg_gpu_time = total_gpu_time / count  # currently not being used. if needed can be returned.
    avg_wall_time = total_wall_time / count
    return avg_wall_time
