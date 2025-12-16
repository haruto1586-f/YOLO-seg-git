import sys
import subprocess
import os

# 各スクリプトをモジュールとしてインポート
import main_train_subset as main_train
import main_analysis_subset as main_analysis
#import main_visualization_subset as main_visualization


def run_script(script_name):
    print(f"\n>>> Running {script_name}...")
    # 現在のPythonインタプリタを使ってスクリプトを実行
    try:
        subprocess.run([sys.executable, script_name], check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ エラーが発生しました: {script_name} (Exit code: {e.returncode})")
        sys.exit(e.returncode)


def run_pipeline():
    print("==================================================")
    print("🚀 全パイプラインの実行を開始します")
    print("==================================================")

    try:
        # --- Step 1: 学習フェーズ ---
        print("\n\n>>> Step 1: Running Training Phase...")
        main_train.main()

        # --- Step 1: 解析フェーズ ---
        print("\n\n>>> Step 2: Running Analysis Phase...")
        main_analysis.main()

        # --- Step 1: 可視化フェーズ (必要であればコメントアウトを外す) ---
        print("\n\n>>> Step 3: Running Visualization Phase...")
        #main_visualization.main()

    except Exception as e:
        print(f"\n❌ エラーが発生したためパイプラインを中断します: {e}")
        sys.exit(1)

    print("\n==================================================")
    print("✅ 全ての工程が正常に完了しました")
    print("==================================================")


if __name__ == "__main__":
    sys.path.append(os.getcwd())
    run_pipeline()