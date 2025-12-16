import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
from configs import crack as cfg  # 環境に合わせて config または batch_config に変更してください

def false_rate_plot():
    # ---------------------------------------------------------
    # 1. 設定と準備
    # ---------------------------------------------------------
    base_dir = cfg.AL_RUNS_DIR
    output_dir = os.path.join(base_dir, "error_rate_comparison")
    os.makedirs(output_dir, exist_ok=True)

    print(f"ターゲットディレクトリ: {base_dir}")
    print(f"グラフ保存先: {output_dir}")

    # CSVファイルの取得
    csv_files = glob.glob(os.path.join(base_dir, "*", "results.csv"))
    if not csv_files:
        print(f"エラー: '{base_dir}' 内に 'results.csv' が見つかりませんでした。")
        return
    csv_files.sort()

    # ---------------------------------------------------------
    # 2. ターゲットとするFDR(偽発見性率),FNR(偽陰性率)の指定
    # ---------------------------------------------------------
    # ユーザー指定の2種類のみを対象にする
    target_metrics = [
        ("Box_FalseDiscoveryRate", "metrics/precision(B)"),
        ("Box_FalseNegativeRate", "metrics/recall(B)"),
        ("Mask_FalseDiscoveryRate", "metrics/precision(M)"),
        ("Mask_FalseNegativeRate", "metrics/recall(M)"),
    ]
    

    # ---------------------------------------------------------
    # 3. 未検出と過検出のグラフを作成
    # ---------------------------------------------------------
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for metric,source_col in target_metrics:
        print(f"プロット作成中: {metric} (error rate) ...")
        
        # データの存在確認フラグ
        has_data = False
        
        plt.figure(figsize=(12, 7))

        for i, file_path in enumerate(csv_files):
                try:
                    folder_name = os.path.basename(os.path.dirname(file_path))
                    
                    df = pd.read_csv(file_path)
                    df.columns = [c.strip() for c in df.columns] # 空白除去

                    # カラムが存在しない場合 (例: Maskがないモデルの場合) はスキップ
                    if source_col not in df.columns:
                        continue
                    
                    has_data = True

                    # X軸 (epoch)
                    if 'epoch' in df.columns:
                        x_data = df['epoch']
                    else:
                        x_data = df.index + 1

                    # Y軸 (エラー率の計算: 1 - Precision または 1 - Recall)
                    y_data = (1.0 - df[source_col]) * 100 # パーセンテージ表示

                    # 実験ごとに色を変えてプロット
                    color = colors[i % len(colors)]
                    plt.plot(x_data, y_data, linestyle='-', color=color, linewidth=1.5, 
                            label=folder_name)
                    
                except Exception as e:
                    print(f"  -> スキップ: {os.path.basename(file_path)} ({e})")

        if not has_data:
            print(f"  -> 警告: {source_col} のデータが見つかりませんでした。")
            plt.close()
            continue

        # ---------------------------------------------------------
        # 4. 装飾と保存
        # ---------------------------------------------------------
        clean_title = metric.replace('/', '_')
        
        if metric.endswith("FalseDiscoveryRate"):
            plt.title(f"{metric}:過検出の割合", fontsize=14)
        else:
            plt.title(f"{metric}:未検出の割合", fontsize=14)
            
        plt.xlabel("Epochs", fontsize=12)
        plt.ylabel("Error Rate(%)", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)
        
        # 凡例 (グラフの外側に配置)
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=9)
        plt.tight_layout()

        save_name = f"compare_{clean_title}.png"
        save_path = os.path.join(output_dir, save_name)
        plt.savefig(save_path, dpi=150)
        plt.close()
        
        print(f"  -> 保存完了: {save_name}")

    print("\nすべてのグラフ作成が完了しました。")

if __name__ == "__main__":
    false_rate_plot()