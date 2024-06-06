
from .SSCNet import SSCNet


def make_model(modelname, num_classes):
    if modelname == 'sscnet':
        return SSCNet(num_classes)
    else:
        print(f"Unknown model '{}', using 'sscnet' instead.")
        return SSCNet(num_classes)


__all__ = ["make_model"]
