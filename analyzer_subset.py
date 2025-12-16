import os
import glob
import pandas as pd
import numpy as np
import torch
import cv2
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

# Ultralytics/Torch Metrics
from torchvision.ops import box_iou as compute_bbox_iou
from ultralytics.utils.metrics import mask_iou as compute_mask_iou

# =========================================================
# 1. データ構造定義 (フルスペック)
# =========================================================

@dataclass
class InstanceData:
    """1つの物体（真値または予測値）を表すデータクラス"""
    index: int
    class_index: int
    bbox: torch.Tensor  # [x1, y1, x2, y2]
    mask: torch.Tensor  # Binary Mask [H, W]
    confidence: float = 1.0
    logit_map: Optional[np.ndarray] = None  # [H, W] Probability/Logit map

# =========================================================
# 2. カスタム指標計算 (能動学習・詳細分析用)
# =========================================================

def calculate_custom_metrics(instance: InstanceData) -> Dict[str, Any]:
    """
    標準機能以外に保存したい詳細データを計算する
    """
    metrics = {}
    
    # --- A. 幾何学的情報 ---
    # マスク面積 (ピクセル数)
    mask_area = float(instance.mask.sum())
    metrics['mask_area'] = mask_area

    # バウンディングボックスの面積とアスペクト比
    x1, y1, x2, y2 = instance.bbox.tolist()
    w = x2 - x1
    h = y2 - y1
    bbox_area = w * h
    metrics['bbox_area'] = bbox_area
    metrics['aspect_ratio'] = w / (h + 1e-6)

    # --- B. 不確実性・ロジット情報 (Active Learning用) ---
    # if instance.logit_map is not None:
    #     # logit_map は通常 0~1 の確率値 (Sigmoid済) と想定
    #     probs = instance.logit_map.astype(np.float32)
        
    #     # 1. 平均・標準偏差
    #     metrics['logit_mean'] = float(probs.mean())
    #     metrics['logit_std'] = float(probs.std())
        
    #     # 2. エントロピー (Binary Entropy)
    #     # 境界付近の曖昧さを測る指標
    #     p_safe = np.clip(probs, 1e-6, 1.0 - 1e-6)
    #     entropy_map = -(p_safe * np.log(p_safe) + (1 - p_safe) * np.log(1 - p_safe))
    #     metrics['entropy_mean'] = float(entropy_map.mean())
        
    # else:
    metrics['logit_mean'] = None
    metrics['logit_std'] = None
    metrics['entropy_mean'] = None

    return metrics

# =========================================================
# 3. ヘルパー関数
# =========================================================

def polygons2masks(img_h, img_w, polygons):
    if not polygons:
        return torch.zeros((0, img_h, img_w), dtype=torch.uint8)
    masks = np.zeros((len(polygons), img_h, img_w), dtype=np.uint8)
    for i, poly in enumerate(polygons):
        pts = poly.reshape(-1, 2).astype(np.int32)
        cv2.fillPoly(masks[i], [pts], color=1)
    return torch.from_numpy(masks)

def load_instances_full(label_path, logit_path, img_w, img_h, is_pred=False) -> List[InstanceData]:
    """TXTファイルとNPZファイルから全情報を読み込む"""
    if not os.path.exists(label_path):
        return []

    # NPZ (Logits/Masks raw data) の読み込み
    logits_all = None
    if is_pred and logit_path and os.path.exists(logit_path):
        try:
            # result_saver.py が 'logits' キーで保存している想定
            with np.load(logit_path) as data:
                if 'logits' in data:
                    logits_all = data['logits']
                elif 'masks' in data: # キー名が異なる場合の保険
                    logits_all = data['masks']
        except Exception:
            pass

    raw_data = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = [float(x) for x in line.strip().split()]
            if len(parts) < 5: continue
            class_index = int(parts[0])
            if is_pred:
                conf = parts[-1]
                coords = np.array(parts[1:-1])
            else:
                conf = 1.0
                coords = np.array(parts[1:])
            coords[0::2] *= img_w
            coords[1::2] *= img_h
            raw_data.append((class_index, conf, coords))

    if not raw_data:
        return []

    all_polygons = [d[2] for d in raw_data]
    all_masks = polygons2masks(img_h, img_w, all_polygons)

    instances = []
    for i, (cls_idx, conf, poly) in enumerate(raw_data):
        xs, ys = poly[0::2], poly[1::2]
        if len(xs) > 0:
            bbox = torch.tensor([np.min(xs), np.min(ys), np.max(xs), np.max(ys)], dtype=torch.float32)
        else:
            bbox = torch.zeros(4, dtype=torch.float32)
        
        # 対応するLogitマップを取得
        logit_map = None
        if logits_all is not None and i < len(logits_all):
            logit_map = logits_all[i]
            
        instances.append(InstanceData(i, cls_idx, bbox, all_masks[i], conf, logit_map))

    return instances

# =========================================================
# 4. メイン解析関数 (フル機能)
# =========================================================

def run_analysis_full(raw_data_dir, true_data_pools, output_csv_path, img_size=(640, 640)):
    """
    詳細な解析を実行し、CSVに保存する
    """
    print(f"[Analyzer] Generating FULL analysis report at {output_csv_path}...")
    img_w, img_h = img_size
    all_rows = []
    
    # フォルダ検索
    epoch_dirs = sorted(glob.glob(os.path.join(raw_data_dir, "epoch_*")))

    for ep_dir in epoch_dirs:
        dir_name = os.path.basename(ep_dir)
        try:
            parts = dir_name.split('_')
            epoch = int(parts[1])
            split = "_".join(parts[2:])
        except: continue
        
        print(f"  Processing {dir_name}...")
        
        target_images = true_data_pools.get(split, [])
        pred_label_dir = os.path.join(ep_dir, "labels")
        pred_logit_dir = os.path.join(ep_dir, "logits") # NPZがある場所

        for image_path in target_images:
            image_name = os.path.basename(image_path)
            stem = Path(image_path).stem
            
            # --- 1. 真値読み込み ---
            true_path = str(Path(image_path.replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}")).with_suffix(".txt"))
            true_instances = load_instances_full(true_path, None, img_w, img_h, is_pred=False)
            
            # --- 2. 予測値読み込み ---
            pred_txt = os.path.join(pred_label_dir, f"{stem}.txt")
            pred_npz = os.path.join(pred_logit_dir, f"{stem}.npz")
            pred_instances = load_instances_full(pred_txt, pred_npz, img_w, img_h, is_pred=True)
            
            # --- 3. マッチング ---
            matches, unmatched_preds = [], set(range(len(pred_instances)))
            matrix_mask = None
            
            if pred_instances and true_instances:
                # IoU計算
                p_bboxes = torch.stack([p.bbox for p in pred_instances])
                t_bboxes = torch.stack([t.bbox for t in true_instances])
                bbox_iou_mat = compute_bbox_iou(p_bboxes, t_bboxes).cpu().numpy()
                
                p_masks = torch.stack([p.mask for p in pred_instances]).float()
                t_masks = torch.stack([t.mask for t in true_instances]).float()
                matrix_mask = compute_mask_iou(p_masks.view(len(pred_instances), -1), t_masks.view(len(true_instances), -1)).cpu().numpy()
                
                # IoU > 0.5 でマッチング
                if bbox_iou_mat.size > 0:
                    for idx in np.argsort(bbox_iou_mat.ravel())[::-1]:
                        p_idx, t_idx = divmod(idx, len(true_instances))
                        iou = bbox_iou_mat[p_idx, t_idx]
                        if iou < 0.5: break
                        if p_idx in unmatched_preds and all(m[1] != t_idx for m in matches):
                             matches.append((p_idx, t_idx, iou))
                             unmatched_preds.remove(p_idx)

            match_map = {tid: (pid, biou) for pid, tid, biou in matches}

            # 共通情報
            base_info = {'epoch': epoch, 'split': split, 'image_name': image_name}

            # --- [TP & FN] 真値ループ ---
            for t_inst in true_instances:
                row = base_info.copy()
                row['true_object_id'] = t_inst.index
                
                # 真値側のメトリクス (面積など)
                for k, v in calculate_custom_metrics(t_inst).items():
                    row[f'true_{k}'] = v
                
                if t_inst.index in match_map:
                    # TP
                    p_idx, b_iou = match_map[t_inst.index]
                    p_inst = pred_instances[p_idx]
                    m_iou = matrix_mask[p_idx, t_inst.index] if matrix_mask is not None else 0.0
                    
                    row.update({
                        'confusion_matrix': 'TP', 
                        'confidence': p_inst.confidence, 
                        'iou_bbox': b_iou, 
                        'iou_mask': m_iou,
                        'pred_object_id': p_idx
                    })
                    # 予測側のメトリクス (Entropyなど)
                    for k, v in calculate_custom_metrics(p_inst).items():
                        row[f'pred_{k}'] = v
                else:
                    # FN
                    row.update({
                        'confusion_matrix': 'FN', 
                        'confidence': 0.0, 'iou_bbox': 0.0, 'iou_mask': 0.0,
                        'pred_object_id': -1
                    })
                
                all_rows.append(row)
            
            # --- [FP] 予測ループ ---
            for p_idx in sorted(list(unmatched_preds)):
                p_inst = pred_instances[p_idx]
                row = base_info.copy()
                row.update({
                    'true_object_id': -1,
                    'pred_object_id': p_idx,
                    'confusion_matrix': 'FP', 
                    'confidence': p_inst.confidence, 
                    'iou_bbox': 0.0, 
                    'iou_mask': 0.0
                })
                # 予測側のメトリクス
                for k, v in calculate_custom_metrics(p_inst).items():
                    row[f'pred_{k}'] = v
                
                all_rows.append(row)

    # --- CSV保存 ---
    if all_rows:
        df = pd.DataFrame(all_rows)
        # カラム並び替え
        cols_prio = ['epoch', 'split', 'image_name', 'confusion_matrix', 'confidence', 'iou_mask', 'iou_bbox', 'pred_entropy_mean']
        cols_other = sorted([c for c in df.columns if c not in cols_prio])
        df = df[cols_prio + cols_other] if set(cols_prio).issubset(df.columns) else df
        
        df.to_csv(output_csv_path, index=False)
        print(f"✅ Full Analysis CSV Saved: {output_csv_path} ({len(df)} rows)")
    else:
        print("⚠️ No data found to save.")