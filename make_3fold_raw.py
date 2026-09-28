# -*- coding: utf-8 -*-
"""
3 折交叉验证数据划分（第一步）
- 合并 split/train + split/val + split/test（原始图，不含增强）
- 3 折分层划分（按类别比例 + 小目标占比）
- 每折内部再切 15% 做 val（用于早停）
- 输出到 kfold_raw/fold{0,1,2}/

用法：
    python make_3fold_raw.py
"""

import shutil
from pathlib import Path
from collections import Counter
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

# ============ 路径配置 ============
SRC_ROOT = Path('/home/suxd/workspace/dataset/ciligs/dataaug')
OUT_ROOT = SRC_ROOT / 'kfold_raw'

# ============ 参数 ============
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
K = 3                # 折数
VAL_RATIO = 0.15     # 每折内部 val 占 train+val 的比例
SEED = 2024          # 划分随机种子


def load_yolo_labels(label_path):
    """读取 YOLO 标签，返回 [(cls, x, y, w, h), ...]"""
    boxes = []
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:5])
            if w > 0 and h > 0:
                boxes.append((cls, x, y, w, h))
    return boxes


def get_stratum(label_path, all_classes):
    """
    分层键：各类别实例数分箱 + 小目标数量分箱
    例如: '1_3plus_s2plus' 表示 cls0=1个, cls1=3+个, 小目标=2+个
    """
    boxes = load_yolo_labels(label_path)
    if not boxes:
        return 'empty'

    counts = Counter(b[0] for b in boxes)
    parts = []
    for cls in sorted(all_classes):
        n = counts.get(cls, 0)
        if n == 0:
            parts.append('0')
        elif n == 1:
            parts.append('1')
        elif n == 2:
            parts.append('2')
        else:
            parts.append('3plus')

    # 小目标数量分箱（w*h < 0.04）
    n_small = sum(1 for b in boxes if b[3] * b[4] < 0.04)
    if n_small == 0:
        parts.append('s0')
    elif n_small <= 2:
        parts.append('s1')
    else:
        parts.append('s2plus')

    return '_'.join(parts)


def merge_rare_strata(strata, min_count=5):
    """合并稀有层，避免 StratifiedShuffleSplit 报错"""
    counts = Counter(strata)
    rare = {s for s, c in counts.items() if c < min_count}
    if not rare:
        return strata
    print(f'[分层] 稀有层 {len(rare)} 个将被合并')
    return ['rare' if s in rare else s for s in strata]


def copy_files(pairs, dst_dir):
    """复制图像 + 标签到目标目录"""
    img_out = dst_dir / 'images'
    lbl_out = dst_dir / 'labels'
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    for img, lbl in pairs:
        shutil.copy2(img, img_out / img.name)
        shutil.copy2(lbl, lbl_out / lbl.name)


def main():
    print('=' * 60)
    print('  3 折交叉验证数据划分')
    print('=' * 60)

    # ---- 1. 合并原始数据 ----
    all_pairs = []
    for sub in ['train', 'val', 'test']:
        img_dir = SRC_ROOT / 'split' / sub / 'images'
        lbl_dir = SRC_ROOT / 'split' / sub / 'labels'
        if not img_dir.exists():
            print(f'[警告] {img_dir} 不存在，跳过')
            continue
        n_before = len(all_pairs)
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in IMAGE_EXTS:
                continue
            lbl = lbl_dir / f'{img.stem}.txt'
            if lbl.exists():
                all_pairs.append((img, lbl))
        print(f'  {sub}: {len(all_pairs) - n_before} 张')

    print(f'\n[合并] 总共 {len(all_pairs)} 张原始图像')

    # ---- 2. 检测类别 ----
    class_counts = Counter()
    for _, lbl in all_pairs:
        for b in load_yolo_labels(lbl):
            class_counts[b[0]] += 1

    if not class_counts:
        raise RuntimeError('未找到任何有效标注')

    all_classes = sorted(class_counts.keys())
    minority_cls = min(class_counts, key=class_counts.get)
    print(f'[类别] {dict(class_counts)}')
    print(f'[类别] 少数类: class {minority_cls}')

    # ---- 3. 生成分层键 ----
    strata = [get_stratum(lbl, all_classes) for _, lbl in all_pairs]
    strata = merge_rare_strata(strata, min_count=5)
    print(f'[分层] 各层样本数: {dict(Counter(strata))}')

    # ---- 4. K 折划分 ----
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    sss = StratifiedShuffleSplit(n_splits=K, test_size=1.0 / K, random_state=SEED)
    idx = np.arange(len(all_pairs))

    for fold, (tv_idx, te_idx) in enumerate(sss.split(idx, strata)):
        fold_dir = OUT_ROOT / f'fold{fold}'
        print(f'\n{"=" * 60}')
        print(f'  Fold {fold}')
        print(f'{"=" * 60}')

        tv_pairs = [all_pairs[i] for i in tv_idx]
        te_pairs = [all_pairs[i] for i in te_idx]
        tv_strata = [strata[i] for i in tv_idx]

        # 从 train+val 里再切 val
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=VAL_RATIO, random_state=SEED)
        tr_idx, va_idx = next(sss2.split(np.arange(len(tv_pairs)), tv_strata))
        tr_pairs = [tv_pairs[i] for i in tr_idx]
        va_pairs = [tv_pairs[i] for i in va_idx]

        print(f'  train: {len(tr_pairs)}  val: {len(va_pairs)}  test: {len(te_pairs)}')

        # 统计每折的类别分布
        for sub_name, pairs in [('train', tr_pairs), ('val', va_pairs), ('test', te_pairs)]:
            c = Counter()
            for _, lbl in pairs:
                for b in load_yolo_labels(lbl):
                    c[b[0]] += 1
            print(f'    {sub_name}: 类别分布 {dict(c)}')

        # 复制文件
        copy_files(tr_pairs, fold_dir / 'train')
        copy_files(va_pairs, fold_dir / 'val')
        copy_files(te_pairs, fold_dir / 'test')

    print(f'\n{"=" * 60}')
    print(f'  完成！输出目录: {OUT_ROOT}')
    print(f'{"=" * 60}')
    print(f'  kfold_raw/')
    print(f'  ├── fold0/{{train,val,test}}')
    print(f'  ├── fold1/{{train,val,test}}')
    print(f'  └── fold2/{{train,val,test}}')
    print(f'\n  下一步：运行 augment_3fold.py 对每折 train 做增强')


if __name__ == '__main__':
    main()