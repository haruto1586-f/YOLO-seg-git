import os
import glob
import gc
import torch
import re
from pathlib import Path
from ultralytics import YOLO

# プロジェクト内モジュール
from configs import crack as cfg
import data_manager as dm
import result_saver  # 推論 & 保存
import analyzer      # 解析 & CSV作成

def get_sorted_epoch_weights(weights_dir):
    """重みファイルをエポック番号順にソートして取得する"""
    files = glob.glob(os.path.join(weights_dir, "epoch_*.pt"))
    # ファイル名から数値を抽出してソート (epoch_1.pt, epoch_2.pt, ... epoch_10.pt)
    def extract_epoch(path):
        match = re.search(r'epoch_(\d+)', os.path.basename(path))
        return int(match.group(1)) if match else 0
    
    return sorted(files, key=extract_epoch)

def analyze_single_experiment(exp_dir):
    """
    1つの実験フォルダに対して、推論と解析(CSV作成)を実行する関数
    """
    dir_name = os.path.basename(exp_dir)
    print(f"\n{'='*60}")
    print(f"📂 Analyzing Experiment: {dir_name}")
    print(f"{'='*60}")

    # --- パス設定 ---
    weights_dir = os.path.join(exp_dir, 'weights_history')
    raw_data_dir = os.path.join(exp_dir, 'raw_epoch_data')
    
    # フルサイズ版の結果CSVパス
    analysis_csv_path = os.path.join(exp_dir, 'detailed_analysis_full.csv')

    # --- 事前チェック ---
    if os.path.exists(analysis_csv_path):
        print(f"✅ Already analyzed. Skipping: {dir_name}")
        return

    if not os.path.exists(weights_dir):
        print(f"❌ Weights directory not found: {weights_dir}")
        return

    # データプール(学習/未学習の分割情報)の読み込み
    data_pools = dm.load_data_pools(exp_dir)
    if data_pools is None:
        print("❌ data_pools.json not found. Cannot analyze.")
        return

    # =========================================================
    # Phase 1: 推論実行 & 生データ(TXT/NPZ)保存
    # =========================================================
    weight_files = get_sorted_epoch_weights(weights_dir)
    
    if not weight_files:
        print("❌ No weight files found.")
        return

    print(f"  Found {len(weight_files)} epochs to process.")

    for weight_path in weight_files:
        try:
            # ファイル名からエポック番号を取得
            stem = Path(weight_path).stem  # epoch_10
            epoch = int(stem.split('_')[1])
        except:
            continue
            
        # 必要なSplit（Train/Val/Test/Pool）が全て処理済みか確認
        target_splits = [s for s in ['train', 'val', 'test', 'train_pool'] if s in data_pools]
        all_splits_done = True
        for split in target_splits:
            if not os.path.exists(os.path.join(raw_data_dir, f"epoch_{epoch}_{split}")):
                all_splits_done = False
                break
        
        if all_splits_done:
            print(f"    Epoch {epoch}: Raw data exists. Skipping inference.")
            continue

        print(f"    Epoch {epoch}: Running inference...")
        
        try:
            # モデルロード
            model = YOLO(weight_path, task='segment')
            
            for split in target_splits:
                images = data_pools[split]
                if not images: continue
                
                # result_saver を呼び出して推論結果を保存
                # (以前のコードのロジックをそのまま使用)
                result_saver.save_epoch_predictions(
                    model, epoch, split, images, raw_data_dir
                )
        except Exception as e:
            print(f"    ❌ Error processing epoch {epoch}: {e}")
            continue
        finally:
            # メモリ解放
            if 'model' in locals(): del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # =========================================================
    # Phase 2: 解析実行 & CSV保存
    # =========================================================
    print(f"\n  Generating Full Analysis CSV...")
    
    try:
        # (analyzer.py に run_analysis_full が定義されている前提)
        if hasattr(analyzer, 'run_analysis_full'):
            analyzer.run_analysis_full(
                raw_data_dir, 
                data_pools, 
                analysis_csv_path, 
                img_size=(cfg.IMG_SIZE, cfg.IMG_SIZE)
            )
        else:
            # 万が一関数名が異なる場合のフォールバック
            print("  Warning: 'run_analysis_full' not found. Trying 'run_analysis'...")
            analyzer.run_analysis(
                raw_data_dir, 
                data_pools, 
                analysis_csv_path, 
                img_size=(cfg.IMG_SIZE, cfg.IMG_SIZE)
            )
            
    except Exception as e:
        print(f"❌ Analysis failed for {dir_name}: {e}")
        import traceback
        traceback.print_exc()

def main():
    print(f"--- 📊 Batch Analysis Auto-Detection Mode ---")
    
    # runsディレクトリ以下の "subset_*" フォルダをすべて検索
    search_pattern = os.path.join(cfg.AL_RUNS_DIR, "subset_*")
    exp_dirs = glob.glob(search_pattern)
    
    # フォルダのみを抽出してソート
    exp_dirs = sorted([d for d in exp_dirs if os.path.isdir(d)])
    
    if not exp_dirs:
        print(f"❌ No 'subset_*' experiment folders found in: {cfg.AL_RUNS_DIR}")
        return

    print(f"Found {len(exp_dirs)} experiments to analyze.")
    
    # 各実験フォルダに対して解析を実行
    for exp_dir in exp_dirs:
        analyze_single_experiment(exp_dir)

    print("\n✅ All batch analyses completed.")

if __name__ == "__main__":
    main()