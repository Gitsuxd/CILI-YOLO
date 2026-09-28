#!/bin/bash
set -e

cd ~/workspace/yolov5-my

CFG="/home/suxd/workspace/yolov5-my/models/yolov5n.yaml"    # 最终模型
HYP="/home/suxd/workspace/yolov5-my/data/hyps/hyp.scratch-low.yaml"
KFOLD_ROOT="/home/suxd/workspace/dataset/ciligs/dataaug/kfold"

for FOLD in 0 1 2; do
    echo ""
    echo "=========================================="
    echo "  Fold ${FOLD}"
    echo "=========================================="

    python train.py \
        --cfg ${CFG} \
        --data ${KFOLD_ROOT}/fold${FOLD}/data.yaml \
        --hyp ${HYP} \
        --epochs 100 \
        --patience 30 \
        --batch-size 16 \
        --imgsz 640 \
        --device 0 \
        --name cili_3fold_full_f${FOLD}
done

echo ""
echo "3 折训练完成"