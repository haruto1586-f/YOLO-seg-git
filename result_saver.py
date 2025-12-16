# result_saver.py
import os
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO
import gc
import torch


def save_epoch_predictions(model: YOLO, epoch: int, split_name: str, image_paths: list, save_root_dir: str):
    """
    指定された画像の推論を行い、結果(TXT, NPY)を保存する。
    """
    print(f"  [Saver] Saving raw predictions for '{split_name}' at epoch {epoch}...")

    # 保存先ディレクトリ: runs/raw_data/epoch_1_val/
    epoch_dir = os.path.join(save_root_dir, f"epoch_{epoch}_{split_name}")
    labels_dir = os.path.join(epoch_dir, "labels")
    logits_dir = os.path.join(epoch_dir, "logits")
    os.makedirs(logits_dir, exist_ok=True)  # labelsはYOLOが作るが、logitsは自作

    # 有効なクラスIDリストを取得
    # モデルが認識しているクラスID（crackなら [0]）だけを取得する
    # これにより、事前学習の遺物（COCOクラス）が誤検出されるのを防ぐ
    target_classes = list(model.names.keys())

    # 1回に処理する枚数 (OSの制限に合わせて調整。200程度なら安全)
    CHUNK_SIZE = 200
    total_images = len(image_paths)
    
    for i in range(0, total_images, CHUNK_SIZE):
        # バッチ（小分け）を作成
        chunk_paths = image_paths[i : i + CHUNK_SIZE]
        
        # 進行状況の表示 (任意)
        print(f"    Processing batch {i} - {min(i + CHUNK_SIZE, total_images)} / {total_images}...")

        # 1. 推論 & テキスト保存 (バッチ単位)
        #    exist_ok=True なので、同じフォルダに追記・保存されていく
        results = model.predict(
            chunk_paths,
            classes=target_classes,
            task='segment',
            save=False,
            save_txt=True,
            save_conf=True,
            conf=0.01,
            verbose=False,
            project=save_root_dir,
            name=f"epoch_{epoch}_{split_name}",
            exist_ok=True,
        )

        # 2. ロジット/確率データ (.npz) の保存
        num_npz = 0
        for r in results:

            stem = Path(r.path).stem
            # 1. ボックスの有無確認
            has_boxes = r.boxes is not None and len(r.boxes) > 0
            # 2. マスクの有無確認
            has_masks = r.masks is not None
            if has_boxes and not has_masks:
                print(f"    [WARNING] {stem}: Boxes detected but NO MASKS! (Model type error?)")

            if r.masks is not None:
                try:
                    # float16 にキャスト (容量削減)
                    masks_data = r.masks.data.cpu().numpy().astype(np.float16)

                    stem = Path(r.path).stem
                    save_path = os.path.join(logits_dir, f"{stem}.npz")

                    # 圧縮保存
                    np.savez_compressed(save_path, logits=masks_data)
                    num_npz += 1
                except Exception as e:
                    print(f"    [ERROR] Failed to save numpy: {e}")

        print(f"    [Saver] Saved {num_npz} NPZ files")

        # メモリ解放とファイルハンドルのクローズを促す
        del results
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
