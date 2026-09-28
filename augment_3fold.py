# -*- coding: utf-8 -*-
"""
第二步：对 3 折的 train 做增强
- 读 kfold_raw/fold{i}/train
- 输出 kfold/fold{i}/train_aug（4 常规 + 条件 CP）
- val/test 直接复制（不增强）
- 生成 kfold/fold{i}/data.yaml

用法：
    python augment_3fold.py
"""

import shutil
from pathlib import Path
from collections import Counter
from tqdm import tqdm
import random
import numpy as np
from PIL import Image, ImageOps, ImageEnhance

# ============ 路径配置 ============
SRC_ROOT = Path('/home/suxd/workspace/dataset/ciligs/dataaug')
RAW_ROOT = SRC_ROOT / 'kfold_raw'
OUT_ROOT = SRC_ROOT / 'kfold'

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
CLASS_NAMES = ['CLGSCS', 'CLGSWCS']

# ============ 增强配置 ============
AUGMENTATIONS = ['hflip', 'brightness', 'contrast', 'color_jitter']
CP_TRIGGER = {0: 2, 1: 1}       # 少数类稀疏度 k -> CP 次数
MIN_PASTE_SIZE = 24             # 粘贴后短边最小像素
MAX_PASTE_RATIO = 0.4           # 粘贴框不超过图像短边比例
COVER_THR = 0.05                # 覆盖度阈值
SEED = 2024


# ============ YOLO 标签读写 ============
def load_yolo_labels(label_path):
    boxes = []
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:5])
            if w > 0 and h > 0:
                boxes.append([cls, x, y, w, h])
    return boxes


def save_yolo_labels(boxes, path):
    with open(path, 'w', encoding='utf-8') as f:
        for cls, x, y, w, h in boxes:
            x1 = max(0.0, min(1.0, x - w / 2))
            y1 = max(0.0, min(1.0, y - h / 2))
            x2 = max(0.0, min(1.0, x + w / 2))
            y2 = max(0.0, min(1.0, y + h / 2))
            nw, nh = x2 - x1, y2 - y1
            if nw < 1e-3 or nh < 1e-3:
                continue
            f.write(f'{cls} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} '
                    f'{nw:.6f} {nh:.6f}\n')


# ============ 4 种常规增强 ============
def aug_hflip(img, boxes):
    img = ImageOps.mirror(img)
    boxes = [[c, 1.0 - x, y, w, h] for c, x, y, w, h in boxes]
    return img, boxes


def aug_brightness(img, boxes):
    return ImageEnhance.Brightness(img).enhance(random.uniform(0.6, 1.4)), boxes


def aug_contrast(img, boxes):
    return ImageEnhance.Contrast(img).enhance(random.uniform(0.6, 1.4)), boxes


def aug_color_jitter(img, boxes):
    hsv = img.convert('HSV')
    h, s, v = hsv.split()
    s = s.point(lambda x: min(255, int(x * random.uniform(0.7, 1.3))))
    v = v.point(lambda x: min(255, int(x * random.uniform(0.7, 1.3))))
    return Image.merge('HSV', (h, s, v)).convert('RGB'), boxes


AUG_FNS = {
    'hflip': aug_hflip,
    'brightness': aug_brightness,
    'contrast': aug_contrast,
    'color_jitter': aug_color_jitter,
}


# ============ Copy-Paste ============
def build_minority_pool(image_dir, label_dir, minority_cls,
                        min_patch_size=8, max_patches=300):
    pool = []
    for img_path in sorted(Path(image_dir).iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        lbl_path = Path(label_dir) / f'{img_path.stem}.txt'
        if not lbl_path.exists():
            continue
        boxes = load_yolo_labels(lbl_path)
        minor = [b for b in boxes if b[0] == minority_cls]
        if not minor:
            continue
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception:
            continue
        W, H = img.size
        img_np = np.array(img)
        for cls, x, y, w, h in minor:
            x1 = int(max(0, (x - w / 2) * W))
            y1 = int(max(0, (y - h / 2) * H))
            x2 = int(min(W, (x + w / 2) * W))
            y2 = int(min(H, (y + h / 2) * H))
            if x2 - x1 < min_patch_size or y2 - y1 < min_patch_size:
                continue
            pool.append((img_np[y1:y2, x1:x2].copy(), cls))
            if len(pool) >= max_patches:
                break
        if len(pool) >= max_patches:
            break
    return pool


def aug_copy_paste(img, boxes, pool, n_range=(1, 2), seed=None):
    if not pool:
        return img, boxes
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    img_np = np.array(img).copy()
    H, W = img_np.shape[:2]
    n_paste = random.randint(*n_range)
    new_boxes = list(boxes)

    for _ in range(n_paste):
        patch, cls = random.choice(pool)
        ph, pw = patch.shape[:2]
        min_side = min(pw, ph)
        max_allowed = int(min(W, H) * MAX_PASTE_RATIO)

        if min_side < MIN_PASTE_SIZE:
            if max_allowed <= MIN_PASTE_SIZE:
                continue
            target_side = random.randint(MIN_PASTE_SIZE,
                                         min(MIN_PASTE_SIZE + 24, max_allowed))
            scale = target_side / min_side
        else:
            scale = random.uniform(0.85, 1.15)

        nw = min(int(pw * scale), max_allowed)
        nh = min(int(ph * scale), max_allowed)
        if nw < MIN_PASTE_SIZE or nh < MIN_PASTE_SIZE or nw >= W or nh >= H:
            continue

        patch_resized = np.array(Image.fromarray(patch).resize((nw, nh), Image.BILINEAR))
        paste_area = nw * nh

        ok = False
        px = py = 0
        for _try in range(20):
            px = random.randint(0, W - nw)
            py = random.randint(0, H - nh)
            max_cover = 0.0
            for b in new_boxes:
                _, bx, by, bw, bh = b
                bx1, by1 = (bx - bw / 2) * W, (by - bh / 2) * H
                bx2, by2 = (bx + bw / 2) * W, (by + bh / 2) * H
                ix1, iy1 = max(px, bx1), max(py, by1)
                ix2, iy2 = min(px + nw, bx2), min(py + nh, by2)
                if ix2 > ix1 and iy2 > iy1:
                    inter = (ix2 - ix1) * (iy2 - iy1)
                    max_cover = max(max_cover, inter / paste_area)
            if max_cover <= COVER_THR:
                ok = True
                break
        if not ok:
            continue

        img_np[py:py + nh, px:px + nw] = patch_resized
        new_boxes.append([cls, (px + nw / 2) / W, (py + nh / 2) / H,
                          nw / W, nh / H])

    return Image.fromarray(img_np), new_boxes


# ============ 增强单折的 train ============
def augment_fold_train(fold_idx):
    src = RAW_ROOT / f'fold{fold_idx}' / 'train'
    out = OUT_ROOT / f'fold{fold_idx}' / 'train_aug'
    out_img = out / 'images'
    out_lbl = out / 'labels'
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    img_dir = src / 'images'
    lbl_dir = src / 'labels'

    # 检测少数类
    class_counts = Counter()
    for lbl in lbl_dir.glob('*.txt'):
        for b in load_yolo_labels(lbl):
            class_counts[b[0]] += 1
    minority_cls = min(class_counts, key=class_counts.get)
    print(f'  [类别] {dict(class_counts)}  少数类: {minority_cls}')

    pool = build_minority_pool(img_dir, lbl_dir, minority_cls)
    print(f'  [CP池] {len(pool)} 个少数类目标')

    images = sorted([p for p in img_dir.iterdir()
                     if p.suffix.lower() in IMAGE_EXTS])
    n_saved = 0
    n_cp = 0

    for img_path in tqdm(images, desc=f'  Fold {fold_idx} 增强'):
        lbl_path = lbl_dir / f'{img_path.stem}.txt'
        if not lbl_path.exists():
            continue
        boxes = load_yolo_labels(lbl_path)
        if not boxes:
            continue
        stem = img_path.stem

        # 原图
        shutil.copy2(img_path, out_img / f'{stem}.jpg')
        save_yolo_labels(boxes, out_lbl / f'{stem}.txt')
        n_saved += 1

        try:
            base = Image.open(img_path).convert('RGB')
        except Exception:
            continue

        # 4 种常规增强
        for name in AUGMENTATIONS:
            h = hash((img_path.name, name, SEED)) % (2 ** 32)
            random.seed(h)
            np.random.seed(h)
            new_img, new_boxes = AUG_FNS[name](base.copy(), [b[:] for b in boxes])
            if not new_boxes:
                continue
            out_stem = f'{stem}_{name}'
            new_img.save(out_img / f'{out_stem}.jpg', quality=95)
            save_yolo_labels(new_boxes, out_lbl / f'{out_stem}.txt')
            n_saved += 1

        # 条件 CP
        n_minor = sum(1 for b in boxes if b[0] == minority_cls)
        cp_times = CP_TRIGGER.get(n_minor, 0)
        for cp_idx in range(cp_times):
            h = hash((img_path.name, f'cp_{cp_idx}', SEED)) % (2 ** 32)
            new_img, new_boxes = aug_copy_paste(base.copy(), [b[:] for b in boxes],
                                                pool, seed=h)
            if len(new_boxes) <= len(boxes):
                continue
            n_cp += len(new_boxes) - len(boxes)
            out_stem = f'{stem}_cp{cp_idx}'
            new_img.save(out_img / f'{out_stem}.jpg', quality=95)
            save_yolo_labels(new_boxes, out_lbl / f'{out_stem}.txt')
            n_saved += 1

    print(f'  Fold {fold_idx} train_aug: {n_saved} 张（含 CP 新增目标 {n_cp} 个）')


def copy_val_test(fold_idx):
    src_root = RAW_ROOT / f'fold{fold_idx}'
    dst_root = OUT_ROOT / f'fold{fold_idx}'
    for sub in ['val', 'test']:
        src_img = src_root / sub / 'images'
        src_lbl = src_root / sub / 'labels'
        dst_img = dst_root / sub / 'images'
        dst_lbl = dst_root / sub / 'labels'
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)
        for img in src_img.iterdir():
            if img.suffix.lower() not in IMAGE_EXTS:
                continue
            lbl = src_lbl / f'{img.stem}.txt'
            shutil.copy2(img, dst_img / img.name)
            if lbl.exists():
                shutil.copy2(lbl, dst_lbl / lbl.name)


def main():
    print('=' * 60)
    print('  3 折 train 增强')
    print('=' * 60)

    for fold in range(3):
        print(f'\n{"=" * 60}')
        print(f'  Fold {fold}')
        print(f'{"=" * 60}')
        augment_fold_train(fold)
        copy_val_test(fold)

        # 生成 data.yaml
        fold_dir = OUT_ROOT / f'fold{fold}'
        yaml_path = fold_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            f.write(f'path: {fold_dir.resolve()}\n')
            f.write('train: train_aug/images\n')
            f.write('val:   val/images\n')
            f.write('test:  test/images\n')
            f.write(f'nc: {len(CLASS_NAMES)}\n')
            f.write(f'names: {CLASS_NAMES}\n')
        print(f'  data.yaml: {yaml_path}')

    print(f'\n{"=" * 60}')
    print(f'  完成！输出目录: {OUT_ROOT}')
    print(f'{"=" * 60}')
    print(f'  kfold/')
    print(f'  ├── fold0/{{train_aug,val,test,data.yaml}}')
    print(f'  ├── fold1/{{train_aug,val,test,data.yaml}}')
    print(f'  └── fold2/{{train_aug,val,test,data.yaml}}')


if __name__ == '__main__':
    main()