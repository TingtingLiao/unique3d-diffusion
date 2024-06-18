__modules__ = {}

def register(name):
    def decorator(cls):
        __modules__[name] = cls
        return cls

    return decorator


def find(name):
    return __modules__[name]

from .trainings import base, image2mvimage_trainer, image2image_trainer
