from dataclasses import dataclass


STEM_POLICIES = ('a4', 'a8', 'reference', 'float')


@dataclass(frozen=True)
class QuantPolicy:
    stem: str = 'a4'
    token_channel_wise: bool = True
    head_channel_wise: bool = True
    fold_layer_norm: bool = True

    def __post_init__(self):
        if self.stem not in STEM_POLICIES:
            raise ValueError(f'Unsupported stem policy {self.stem!r}')
