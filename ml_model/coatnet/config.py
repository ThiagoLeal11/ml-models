from __future__ import annotations
from dataclasses import dataclass


@dataclass
class InputImgConfig:
    size: int | tuple[int, int] = 224
    channels: int = 3
    classes: int | None = None


@dataclass
class ModelConfig:
    input: InputImgConfig
    num_blocks: list[int]  # L
    channels: list[int]  # D


@dataclass
class TrainingConfig:
    pretrained: bool
    model_number: int
    center_crop: bool
    randaugment: tuple[int, int]
    mixup_alpha: float
    loss_type: str
    label_smoothing: float
    train_epochs: int
    train_batch_size: int
    optimizer_type: str
    peak_learning_rate: float
    min_learning_rate: float
    warm_up: int
    lr_decay_schedule: str
    weight_decay_rate: float
    gradient_clip: float
    ema_decay_rate: float | None
    stochastic_depth_rate: float = 0.2

    def set_stochastic_depth_rate(self, stochastic_depth_rate) -> 'TrainingConfig':
        self.stochastic_depth_rate = stochastic_depth_rate
        return self


IMAGENET_1K = InputImgConfig(
    size=224,
    channels=3,
    classes=1000,
)

COATNET_0 = ModelConfig(
    input=IMAGENET_1K,
    num_blocks=[2, 2, 3, 5, 2],
    channels=[64, 96, 192, 384, 768],
)

PRETRAINING_IMAGENET_1K = TrainingConfig(
    pretrained=False,
    model_number=0,
    center_crop=True,
    randaugment=(2, 15),
    mixup_alpha=0.8,
    loss_type='softmax',
    label_smoothing=0.1,
    train_epochs=300,
    train_batch_size=4096,
    optimizer_type='adamw',
    peak_learning_rate=0.001,
    min_learning_rate=0.000_01,
    warm_up=10_000,
    lr_decay_schedule='cosine',
    weight_decay_rate=0.05,
    gradient_clip=1.0,
    ema_decay_rate=None,
)
