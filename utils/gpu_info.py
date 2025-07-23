import torch

def can_run_project():
    if not torch.cuda.is_available():
        return False, ["CUDA not available (no GPU)"]

    device = torch.device("cuda")
    major, minor = torch.cuda.get_device_capability(device)

    reasons = []

    if major < 6:
        reasons.append("FP16 not supported (requires compute capability ≥ 6.0)")

    if major < 7:
        reasons.append("INT8 and Tensor Cores not supported (requires compute capability ≥ 7.0)")

    if not (major >= 6):
        reasons.append("AMP FP16 requires compute capability ≥ 6.0")

    return len(reasons) == 0, reasons

if __name__ == "__main__":
    ok, reasons = can_run_project()

    if ok:
        print("You can run the Project.")
    else:
        print("You can't run the Project because:")
        for reason in reasons:
            print(f" - {reason}")
