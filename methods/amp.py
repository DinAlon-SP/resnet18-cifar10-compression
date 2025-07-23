import torch

from utils.base_model import ResNet
from data.prepare_data import get_cifar10_datasets
from utils.base_train import train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":

    train_set, val_set, test_set, class_names, full_labels = get_cifar10_datasets(
        seed=42, from_disk=True
    )

    full_model = ResNet(num_classes=10).to(device)

    train(
        model=full_model,
        train_set=train_set,
        val_set=val_set,
        test_set=test_set,
        run_name="resnet_full_amp",
        epochs=100,
        batch_size=256,
        optimizer_name="SGD",
        lr=1e-2,
        momentum=0.9,
        weight_decay=1e-2,
        loss_name='CrossEntropy',
        label_smoothing=0.0,
        l1_lambda=0.0,
        use_scheduler=True,
        amp_mode="full"
    )
