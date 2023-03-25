
AP_CONFIG=/home/homesecurity/CS281-Robbery-Detection/AlphaPose/configs/halpe_68_noface/resnet/256x192_res50_lr1e-3_2x-dcn-combined.yaml
AcT_CONFIG=/home/homesecurity/CS281-Robbery-Detection/config.yaml

AP_CKPT=/home/homesecurity/CS281-Robbery-Detection/AlphaPose/pretrained_models/noface_fast50_dcn_combined_256x192.pth
AcT_CKPT=/home/homesecurity/CS281-Robbery-Detection/AcT/bin/AcT_large_1_2.h5

INPUT=/home/homesecurity/CS281-Robbery-Detection/AlphaPose/examples/theft_data2/steal/steal56.mp4
OUTDIR=${4:-"./examples/res"}

python3 predict.py \
    --AP_cfg ${AP_CONFIG} --AP_CKPT ${AP_CKPT} --AP_detector "yolo"\
    --AcT_cfg ${AcT_CONFIG} --AcT_CKPT ${AcT_CKPT}\
    --outdir ${OUTDIR} \
    --video "$INPUT" \
    --profile --save_video \
    --qsize 1024
    

