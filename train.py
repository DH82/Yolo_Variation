from ultralytics import YOLO
import argparse

def main(opt):
    model = YOLO(opt.model)
    model.train(
        data=opt.data,
        epochs=opt.epochs,
        patience=opt.patience,
        batch=opt.batch,
        imgsz=opt.imgsz,
        save=opt.save,
        save_period=opt.save_period,
        cache=opt.cache,
        device=opt.device,
        workers=opt.workers,
        project=opt.project,
        name=opt.name,
        exist_ok=opt.exist_ok,
        pretrained=opt.pretrained,
        optimizer=opt.optimizer,
        verbose=opt.verbose,
        seed=opt.seed,
        deterministic=opt.deterministic,
        single_cls=opt.single_cls,
        rect=opt.rect,
        cos_lr=opt.cos_lr,
        close_mosaic=opt.close_mosaic,
        resume=opt.resume,
        amp=opt.amp,
        dropout=opt.dropout
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='ultralytics/cfg/models/v8/yolov8.yaml', required=True, help='YOLO config YAML or pretrained weights path')
    parser.add_argument('--data', type=str, default='/app/Base/Fold0/data.yaml', help='Path to dataset YAML')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size per device')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--optimizer', type=str, default='SGD', help='Optimizer to use (SGD, Adam, AdamW)')
    parser.add_argument('--device', type=str, default='', help='CUDA device, e.g. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='Number of dataloader workers')
    parser.add_argument('--amp', type=bool, default=True, help='Use AMP (Automatic Mixed Precision)')
    parser.add_argument('--patience', type=int, default=50, help='Early stopping patience')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--cos_lr', type=bool, default=False, help='Use cosine learning rate schedule')
    parser.add_argument('--single_cls', type=bool, default=False, help='Train multi-class as single-class')
    parser.add_argument('--resume', type=bool, default=False, help='Resume training from last checkpoint')
    parser.add_argument('--pretrained', type=bool, default=False, help='Use pretrained model')
    parser.add_argument('--rect', type=bool, default=False, help='Rectangular training')
    parser.add_argument('--project', type=str, default='runs/train', help='Project folder to save results')
    parser.add_argument('--name', type=str, default='exp', help='Run name')
    parser.add_argument('--exist_ok', type=bool, default=False, help='Overwrite existing project/name')
    parser.add_argument('--save', type=bool, default=True, help='Save checkpoints and results')
    parser.add_argument('--save_period', type=int, default=-1, help='Checkpoint save interval (-1 for epoch end only)')
    parser.add_argument('--cache', type=bool, default=False, help='Use dataset caching (ram/disk)')
    parser.add_argument('--verbose', type=bool, default=False, help='Verbose output')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--deterministic', type=bool, default=False, help='Deterministic mode for reproducibility')
    parser.add_argument('--close_mosaic', type=int, default=0, help='Disable mosaic augmentation after N epochs')

    opt = parser.parse_args()
    main(opt)
