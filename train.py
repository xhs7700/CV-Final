import os

from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.model_zoo import model_zoo
from detectron2.utils.logger import setup_logger

from core import dataset_cfg, val_set, train_set, model_cfg, output_cfg

if __name__ == '__main__':
    model = 'RetinaNet'
    is_train = True
    
    setup_logger()
    assert model in model_cfg
    
    register_coco_instances(**dataset_cfg['val'], metadata={})
    register_coco_instances(**dataset_cfg['train'], metadata={})
    
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_cfg[model]))
    cfg.DATASETS.TRAIN = (train_set,)
    cfg.DATASETS.TEST = (val_set,)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00015
    cfg.SOLVER.MAX_ITER = 1000
    cfg.SOLVER.STEPS = []
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 8
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.RETINANET.NUM_CLASSES = 1
    cfg.OUTPUT_DIR = output_cfg[model]
    
    if is_train:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_cfg[model])
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    else:
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    if is_train:
        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()
    else:
        predictor = DefaultPredictor(cfg)
        evaluator = COCOEvaluator(val_set, output_dir=cfg.OUTPUT_DIR)
        val_loader = build_detection_test_loader(cfg, val_set)
        print(inference_on_dataset(predictor.model, val_loader, evaluator))
