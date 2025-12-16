import os
import glob
import re
from configs import crack as cfg
import metrics_visualizer as visualizer

import data_manager as dm


def main_last_run():
    """ configファイルに従い、最後に実行したtrain/analysisのログを可視化
    """
    print(f"--- 📈 可視化フェーズ (Dataset: {cfg.DATASET_YAML_NAME}) ---")

    # 1. 対象ディレクトリの特定
    target_exp_dir = dm.find_latest_experiment_dir(cfg.AL_RUNS_DIR, prefix='train_exp')
    
    if target_exp_dir is None:
        print(f"❌ エラー: 実験フォルダが見つかりません。")
        return

    print(f"✅ 対象実験: {target_exp_dir}")
    
    # 2. CSVファイルの確認
    analysis_csv_path = os.path.join(target_exp_dir, 'final_analysis_debug.csv')
    
    if not os.path.exists(analysis_csv_path):
        print(f"❌ エラー: 解析CSVが見つかりません: {analysis_csv_path}")
        print("   先に 'main_analysis.py' を実行して解析データを生成してください。")
        return

    # 3. 可視化の実行
    print("\n--- [Phase 2] Running Visualizer ---")
    
    # 履歴グラフ (mAP, Precision, Recall推移など)
    visualizer.visualize_history(analysis_csv_path)
    
    # 分布グラフ (Violin Plot, Histogramなど)
    visualizer.visualize_distributions(analysis_csv_path)
    
    print("\n" + "="*50)
    print("✅ 可視化が完了しました。")
    print(f"   グラフは {target_exp_dir} 内に保存されています。")
    print("   (.html ファイルをブラウザで開いて確認してください)")
    print("="*50)


def main_specific_run(analysis_csv_path):
    """ 任意の位置に保存されているログから可視化
    """
    if not os.path.exists(analysis_csv_path):
        print(f"❌ エラー: 解析CSVが見つかりません: {analysis_csv_path}")
        print("   先に 'main_analysis.py' を実行して解析データを生成してください。")
        return

    # 分布グラフ (Violin Plot, Histogramなど)
    for vis_style in ['simple_stack', 'truth_vs_noise', 'success_vs_error', 'flow']:
        print(f"    可視化: {vis_style}")
        visualizer.plot_confusion_trends(analysis_csv_path, vis_style)

    print("\n" + "=" * 50)
    print("✅ 可視化が完了しました。")
    # print(f"   グラフは {target_exp_dir} 内に保存されています。")
    print("   (.html ファイルをブラウザで開いて確認してください)")
    print("=" * 50)


if __name__ == "__main__":
    # main_last_run()
    main_specific_run('final_analysis_debug.csv')
