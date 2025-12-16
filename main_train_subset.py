import os
import random
import copy
from pathlib import Path
from torch import save as save_model

# プロジェクト内モジュール
from configs import crack as cfg
import data_manager as dm
import model_interface as mi


def seed_everything(seed):
    """再現性のためのシード固定"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # 必要に応じて torch, numpy のシード固定を追加してください
    # import torch
    # torch.manual_seed(seed)
    # import numpy as np
    # np.random.seed(seed)


def main(num_selected_images=50, epochs=100, seed=42):
    """
    指定された枚数とエポック数で学習を実行するメイン関数
    
    Args:
        num_selected_images (int): 学習に使用する画像の枚数
        epochs (int): 学習エポック数
        seed (int): ランダムシード (データの選択に使用)
    """
    # 保存フォルダ名に条件を含める (例: subset_500_ep100_exp1)
    exp_prefix = f"subset_{num_selected_images}_ep{epochs}_exp"
    
    print(f"\n{'='*60}")
    print(f"🚀 実験開始: Images={num_selected_images}, Epochs={epochs}, Seed={seed}")
    print(f"{'='*60}")
    
    # 1. シード固定
    seed_everything(seed)

    # 2. データセット準備 (全量取得)
    data_pools, data_config = dm.get_dataset_pools()
    
    # --- データサンプリング処理 (Trainのみ削減、残りをPoolとして保存) ---
    full_train_list = data_pools.get('train', [])
    total_train_len = len(full_train_list)

    if num_selected_images < total_train_len:
        # ランダムに指定枚数を選択
        selected_train = random.sample(full_train_list, num_selected_images)
        
        # 選ばれなかったデータをPoolとして保存
        selected_set = set(selected_train)
        pool_train = [img for img in full_train_list if img not in selected_set]
        
        print(f"[Sampling] Selected: {len(selected_train)} / Pool: {len(pool_train)}")
    else:
        selected_train = full_train_list
        pool_train = []
        print(f"[Sampling] Requested {num_selected_images} >= Total {total_train_len}. Using full dataset.")
    
    # data_pools を更新
    data_pools['train'] = selected_train
    data_pools['train_pool'] = pool_train  # 解析用に未学習データも記録
    
    # --- ワークスペース構築 ---
    # 毎回ワークスペース (dataset_al/) を作り直すため、連続実行してもデータは混ざりません
    dm.setup_batch_workspace(data_config, cfg.AL_DATASET_DIR, cfg.AL_DATA_YAML)
    dm.populate_batch_workspace(data_pools, cfg.AL_DATASET_DIR)

    # 3. 保存先ディレクトリの決定
    base_run_dir = Path(cfg.AL_RUNS_DIR) / exp_prefix
    save_dir = dm.increment_path(base_run_dir, exist_ok=False, mkdir=True)
    
    print(f"[Main] Results will be saved to: {save_dir}")
    
    # data_pools保存 (どの画像が選ばれたかの記録)
    dm.save_data_pools(data_pools, save_dir)
    
    # 重み保存用ディレクトリ
    weights_history_dir = save_dir / 'weights_history'
    os.makedirs(weights_history_dir, exist_ok=True)

    # 4. コールバック定義: 重みの保存
    def on_epoch_end(trainer):
        current_epoch = trainer.epoch + 1
        save_interval = getattr(cfg, 'SAVE_PERIOD', 1)
        
        if current_epoch % save_interval == 0:
            target_pt = weights_history_dir / f"epoch_{current_epoch}.pt"
            # EMAがあれば優先、なければ通常モデル
            model_to_save = trainer.ema.ema if trainer.ema else trainer.model
            
            checkpoint = {
                'epoch': current_epoch,
                'model': copy.deepcopy(model_to_save).to('cpu'),
                'names': trainer.data['names'],
                'optimizer': None,
                'train_args': vars(trainer.args),
            }
            save_model(checkpoint, target_pt)

    # 5. 学習実行
    try:
        mi.train_model_with_callback(
            base_model_path=cfg.YOLO_MODEL,
            data_yaml=cfg.AL_DATA_YAML,
            epochs=epochs,             # 引数のエポック数を使用
            imgsz=cfg.IMG_SIZE,
            project_dir=save_dir.parent, 
            run_name=save_dir.name,
            on_epoch_end_callback=on_epoch_end
        )
        print(f"\n✅ 実験完了: {save_dir}")
        
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        # 連続実行を止めない場合はここで握りつぶすが、基本は止めるかログに残す
        raise e


if __name__ == "__main__":
    # 保存先ルートの作成
    os.makedirs(cfg.AL_ROOT, exist_ok=True)

    # --- 実験条件の定義 ---
    # ここに行を追加していけば、寝ている間に全ての実験が終わります
    
    main(num_selected_images=225, epochs=300)
    main(num_selected_images=450, epochs=300)
    main(num_selected_images=900, epochs=300)
    main(num_selected_images=1800, epochs=300)
    main(num_selected_images=3717, epochs=300)
    
    # # 条件1: データ数500枚でのエポック数比較
    # main(num_selected_images=500, epochs=100)
    # main(num_selected_images=500, epochs=50)
    # main(num_selected_images=500, epochs=200)

    # # 条件2: データ数100枚でのエポック数比較
    # main(num_selected_images=100, epochs=100)
    # main(num_selected_images=100, epochs=50)
    # main(num_selected_images=100, epochs=200)