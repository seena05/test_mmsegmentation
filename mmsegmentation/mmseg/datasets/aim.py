from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class AIMDataset(CustomDataset):

    CLASSES = None
    PALETTE = None

    def __init__(self, **kwargs):
        super(AIMDataset, self).__init__(
            img_suffix='.png', seg_map_suffix='.png', **kwargs)
