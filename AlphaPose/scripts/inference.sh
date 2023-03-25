# set -x
cd "/home/homesecurity/CS281-Robbery-Detection/AlphaPose"
CONFIG=configs/halpe_68_noface/resnet/256x192_res50_lr1e-3_2x-dcn-combined.yaml
CKPT=pretrained_models/noface_fast50_dcn_combined_256x192.pth
OUTDIR=${4:-"./examples/res"}
# examples/theft_data2/non-steal/*

for file in examples/theft_data2/steal/*
do
  echo "$file"
  filename=$(basename -- "$file")
  extension="${filename##*.}"
  filename="${filename%.*}"
  OUTDIR="./examples/res/${filename}"
  if timeout 300s python scripts/demo_inference.py \
    --cfg ${CONFIG} \
    --checkpoint ${CKPT} \
    --outdir ${OUTDIR} \
    --detector "yolo"  \
    --debug \
    --video "$file"
  then
    echo "Process $file completed"
    continue
  else
    echo "Skipping file $file due to error"
  fi
done