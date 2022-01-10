import os

import cv2
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import load_coco_json
from detectron2.engine import DefaultPredictor
from detectron2.model_zoo import model_zoo
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from tqdm import tqdm

from core import dataset_cfg, model_cfg, output_cfg


def register_dataset(json_file, image_root, name):
    DatasetCatalog.register(name, lambda: load_coco_json(json_file, image_root, name))
    MetadataCatalog.get(name).set(
        json_file=json_file,
        image_root=image_root,
        evaluator_type="coco",
        thing_classes=['bottle'],
    )
    inner_metadata = MetadataCatalog.get(name)
    inner_dataset_dicts = load_coco_json(json_file, image_root, name)
    return inner_dataset_dicts, inner_metadata


if __name__ == '__main__':
    model = 'RetinaNet'
    
    setup_logger()
    assert model in model_cfg
    
    dataset_dicts, metadata = register_dataset(**dataset_cfg['val'])
    
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_cfg[model]))
    cfg.MODEL.WEIGHTS = os.path.join(output_cfg[model], "model_final.pth")
    cfg.MODEL.RETINANET.NUM_CLASSES = 1
    predictor = DefaultPredictor(cfg)
    
    for d in tqdm(dataset_dicts):
        _, file_name = os.path.split(d['file_name'])
        img = cv2.imread(d['file_name'])
        out = predictor(img)
        v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1.0)
        img_out = v.draw_instance_predictions(out['instances'].to('cpu'), score_threshold=0.5)
        cv2.imwrite(f'./output_img/{file_name}', img_out.get_image()[:, :, ::-1])
