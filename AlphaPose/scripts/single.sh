# set -x
cd "/home/homesecurity/CS281-Robbery-Detection/AlphaPose"
CONFIG=configs/halpe_68_noface/resnet/256x192_res50_lr1e-3_2x-dcn-combined.yaml
CKPT=pretrained_models/noface_fast50_dcn_combined_256x192.pth
OUTDIR=${4:-"./examples/res"}
# examples/theft_data2/non-steal/*


python scripts/demo_inference.py \
--cfg ${CONFIG} \
--checkpoint ${CKPT} \
--outdir "${OUTDIR}" \
--detector "yolo"  --save_img --save_video \
--debug \
--video "examples/theft_data2/steal/steal56.mp4"
