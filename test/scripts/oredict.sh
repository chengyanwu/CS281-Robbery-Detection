# set -x
# cd "/home/homesecurity/CS281-Robbery-Detection/AlphaPose"
# cd "$(dirname "$0")"
pwd
CONFIG=configs/halpe_136/resnet/256x192_res50_lr1e-3_2x-dcn-combined.yaml
CKPT=pretrained_models/halpe136_fast50_dcn_combined_256x192_10handweight.pth
OUTDIR=${4:-"./examples/res"}



python scripts/prediction_generation.py \
    --cfg ${CONFIG} \
    --checkpoint ${CKPT} \
    --outdir ${OUTDIR} \
    --detector "yolo"  --save_img --save_video \
    --video examples/limin.mp4 \
    --profile


