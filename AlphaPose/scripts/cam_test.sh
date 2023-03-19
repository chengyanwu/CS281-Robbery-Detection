# set -x
# cd "/home/homesecurity/CS281-Robbery-Detection/AlphaPose"
# cd "$(dirname "$0")"
pwd
CONFIG=configs/coco/resnet/256x192_res152_lr1e-3_1x-duc.yaml
CKPT=pretrained_models/fast_421_res152_256x192.pth
OUTDIR=${4:-"./examples/res"}

python scripts/demo_inference.py \
    --cfg ${CONFIG} \
    --checkpoint ${CKPT} \
    --outdir ${OUTDIR} \
    --detector "yolo"  --save_img --save_video \
    --webcam 1

