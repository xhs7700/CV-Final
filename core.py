val_set = 'bottle_val'
train_set = 'bottle_train'

dataset_dir = {
    'val': f'./{val_set}',
    'train': f'./{train_set}'
}

dataset_cfg = {
    'val': {
        'json_file': f'{dataset_dir["val"]}/annotations/instances_default.json',
        'image_root': f'{dataset_dir["val"]}/images',
        'name': val_set,
    },
    'train': {
        'json_file': f'{dataset_dir["train"]}/annotations/instances_default.json',
        'image_root': f'{dataset_dir["train"]}/images',
        'name': train_set,
    }
}

model_cfg = {
    'FasterRCNN': "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
    # 'FastRCNN': "COCO-Detection/fast_rcnn_R_50_FPN_1x.yaml",
    # 'RCNN': "COCO-Detection/rpn_R_50_FPN_1x.yaml",
    'RetinaNet': "COCO-Detection/retinanet_R_50_FPN_3x.yaml",
}

output_cfg = {
    'FasterRCNN': "./output_FasterRCNN",
    # 'FastRCNN': "./output_FastRCNN",
    # 'RCNN': "./output_RCNN",
    'RetinaNet': "output_RetinaNet",
}


if __name__ == '__main__':
    pass
