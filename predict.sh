
AP_CONFIG=/home/homesecurity/CS281-Robbery-Detection/AlphaPose/configs/halpe_68_noface/resnet/256x192_res50_lr1e-3_2x-dcn-combined.yaml
AcT_CONFIG=/home/homesecurity/CS281-Robbery-Detection/AcT/utils/config.yaml

AP_CKPT=/home/homesecurity/CS281-Robbery-Detection/AlphaPose/pretrained_models/noface_fast50_dcn_combined_256x192.pth
AcT_CKPT=/home/homesecurity/CS281-Robbery-Detection/AcT/bin/AcT_micro_3_9.h5

# INPUT=/home/homesecurity/CS281-Robbery-Detection/AlphaPose/examples/res/non-steal3/alphapose-results.json
INPUT=/home/homesecurity/CS281-Robbery-Detection/AlphaPose/examples/theft_data2/non-steal/non-steal2.mp4
OUTDIR=${4:-"./examples/res"}

python predict.py \
    --AP_cfg ${AP_CONFIG} --AP_CKPT ${AP_CKPT} --AP_detector "yolo"\
    --AcT_cfg ${AcT_CONFIG} --AcT_CKPT ${AcT_CKPT}\
    --video "$INPUT" \
    --qsize 1024

