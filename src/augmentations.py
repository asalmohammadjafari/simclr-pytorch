from torchvision import transforms


class SimCLRTransform:
    def __init__(self, size=32):
        color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
        self.base_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(size=size, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2023, 0.1994, 0.2010),
                ),
            ]
        )

    def __call__(self, x):
        return self.base_transform(x), self.base_transform(x)
