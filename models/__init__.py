# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .detr.detr import build
from .quant_detr.quant_detr import build_quant

def build_model(args):
    return build(args)

def build_quant_model(args):
    return build_quant(args)
