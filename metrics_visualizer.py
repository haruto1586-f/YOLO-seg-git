# metrics_visualizer.py
"""
保存された評価データを読み込み、評価指標(Metrics)を可視化するモジュール
"""
import os
import numpy as np
import pandas as pd

import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"
from plotly.subplots import make_subplots


def plot_confusion_trends(log_file_path: str, viz_style: str = 'truth_vs_noise', save_dir: str = os.path.curdir):
    """
    YOLO-segの学習ログ(CSV)を読み込み、Train/Val/Test の各splitにおける
    TP/FN/FP数の推移を横並びのサブプロットで可視化します。

    Args:
        log_file_path (str): ログファイル(*.csv)へのパス。
        viz_style (str): グラフの描画スタイル。
            - 'simple_stack': 単純積み上げ棒グラフ
            - 'truth_vs_noise': 正解(TP+FN) vs 過検出(FP) [推奨]
            - 'success_vs_error': 成果(TP) vs エラー(FN+FP)
            - 'flow': 積み上げ面グラフ (Sankey風の推移表現)
    """

    # ---------------------------------------------------------
    # 0. 初期設定
    # ---------------------------------------------------------
    label_tp = 'TP (Correct)'
    label_fn = 'FN (Undetected)'
    label_fp = 'FP (Over-detection)'

    # 配色設定
    c_tp = 'rgba(44, 160, 44, 0.9)'  # 緑
    c_fn = 'rgba(255, 127, 14, 0.9)'  # オレンジ
    c_fp = 'rgba(214, 39, 40, 0.9)'  # 赤

    if not os.path.exists(log_file_path):
        print(f"エラー: ファイルが見つかりません -> {log_file_path}")
        return

    df = pd.read_csv(log_file_path)

    # ---------------------------------------------------------
    # 1. レイアウト作成 (1行3列)
    # ---------------------------------------------------------
    # データに含まれる split の種類を確認し、順番を定義
    target_splits = [s for s in ['train', 'val', 'test'] if s in df['split'].unique()]

    if not target_splits:
        print("エラー: 有効な split (train, val, test) がデータに見つかりません。")
        return

    # サブプロットの作成
    fig = make_subplots(
        rows=1, cols=len(target_splits),
        subplot_titles=[f"<b>{s.upper()}</b> Set" for s in target_splits],  # 太字でタイトル
        horizontal_spacing=0.05,  # グラフ間の隙間
        shared_yaxes=False  # データ数が違うためY軸は独立させる
    )

    # ---------------------------------------------------------
    # 2. 各Splitごとにグラフを描画
    # ---------------------------------------------------------
    for i, split in enumerate(target_splits, start=1):

        # --- データ集計 ---
        df_subset = df[df['split'] == split].copy()
        epoch_stats = df_subset.groupby(['epoch', 'type']).size().unstack(fill_value=0)

        for col in ['TP', 'FN', 'FP']:
            if col not in epoch_stats.columns:
                epoch_stats[col] = 0

        epoch_stats = epoch_stats.reset_index()

        epochs = epoch_stats['epoch']
        tp = epoch_stats['TP']
        fn = epoch_stats['FN']
        fp = epoch_stats['FP']

        # 凡例は最初のグラフ(i==1)のみ表示し、残りは非表示にする（重複防止）
        show_legend = True if i == 1 else False

        # --- トレース追加 ---
        if viz_style == 'flow':
            # 積み上げ面グラフ
            fig.add_trace(go.Scatter(
                x=epochs, y=tp, mode='lines', line=dict(width=0.5, color=c_tp),
                stackgroup='one', name=label_tp, showlegend=show_legend,
                hovertemplate='TP: %{y}件<extra></extra>'
            ), row=1, col=i)
            fig.add_trace(go.Scatter(
                x=epochs, y=fn, mode='lines', line=dict(width=0.5, color=c_fn),
                stackgroup='one', name=label_fn, showlegend=show_legend,
                hovertemplate='FN: %{y}件<extra></extra>'
            ), row=1, col=i)
            fig.add_trace(go.Scatter(
                x=epochs, y=fp, mode='lines', line=dict(width=0.5, color=c_fp),
                stackgroup='one', name=label_fp, showlegend=show_legend,
                hovertemplate='FP: %{y}件<extra></extra>'
            ), row=1, col=i)

        elif viz_style == 'simple_stack':
            fig.add_trace(go.Bar(x=epochs, y=tp, name=label_tp, marker_color=c_tp, showlegend=show_legend), row=1,
                          col=i)
            fig.add_trace(go.Bar(x=epochs, y=fn, name=label_fn, marker_color=c_fn, showlegend=show_legend), row=1,
                          col=i)
            fig.add_trace(go.Bar(x=epochs, y=fp, name=label_fp, marker_color=c_fp, showlegend=show_legend), row=1,
                          col=i)

        elif viz_style == 'truth_vs_noise':
            # 推奨: 正解 vs ノイズ
            fig.add_trace(go.Bar(x=epochs, y=fn, name=label_fn, marker_color=c_fn, showlegend=show_legend), row=1,
                          col=i)
            fig.add_trace(go.Bar(x=epochs, y=tp, name=label_tp, marker_color=c_tp, showlegend=show_legend), row=1,
                          col=i)
            # マイナス方向
            fig.add_trace(go.Bar(x=epochs, y=-fp, name=label_fp, marker_color=c_fp, showlegend=show_legend,
                                 customdata=fp, hovertemplate='Epoch: %{x}<br>Count: %{customdata}<extra></extra>'),
                          row=1, col=i)

        elif viz_style == 'success_vs_error':
            fig.add_trace(go.Bar(x=epochs, y=tp, name=label_tp, marker_color=c_tp, showlegend=show_legend), row=1,
                          col=i)
            fig.add_trace(go.Bar(x=epochs, y=-fn, name=label_fn, marker_color=c_fn, showlegend=show_legend,
                                 customdata=fn, hovertemplate='Epoch: %{x}<br>Count: %{customdata}<extra></extra>'),
                          row=1, col=i)
            fig.add_trace(go.Bar(x=epochs, y=-fp, name=label_fp, marker_color=c_fp, showlegend=show_legend,
                                 customdata=fp, hovertemplate='Epoch: %{x}<br>Count: %{customdata}<extra></extra>'),
                          row=1, col=i)

        else:
            raise ValueError(f"Unknown style: {viz_style}")

        # --- 軸のフォーマット調整 (各サブプロット個別に実行) ---

        # Y軸の範囲計算とラベルの絶対値化 (マイナス方向があるスタイルのみ)
        if viz_style in ['truth_vs_noise', 'success_vs_error']:
            max_pos = (tp + fn).max() if viz_style == 'truth_vs_noise' else tp.max()
            max_neg = fp.max() if viz_style == 'truth_vs_noise' else (fn + fp).max()

            # データがない場合の安全策
            if pd.isna(max_pos) or max_pos == 0: max_pos = 10
            if pd.isna(max_neg) or max_neg == 0: max_neg = 10

            y_range = [-max_neg * 1.1, max_pos * 1.1]

            # 目盛りラベルの作成
            tick_vals = np.linspace(y_range[0], y_range[1], 9)  # 9分割程度
            tick_text = [f"{int(abs(v))}" for v in tick_vals]

            # 特定の列（col=i）のY軸のみ更新
            fig.update_yaxes(
                tickmode='array', tickvals=tick_vals, ticktext=tick_text, range=y_range,
                title_text="Count" if i == 1 else None,  # 左端のみ軸タイトル表示
                row=1, col=i
            )
        else:
            # simple_stack, flow の場合
            y_label = 'Volume' if viz_style == 'flow' else 'Count'
            fig.update_yaxes(title_text=y_label if i == 1 else None, row=1, col=i)

        fig.update_xaxes(title_text="Epoch", row=1, col=i)

    # ---------------------------------------------------------
    # 3. 全体のレイアウト調整
    # ---------------------------------------------------------
    # barmodeの設定
    if viz_style == 'simple_stack':
        barmode = 'stack'
    elif viz_style in ['truth_vs_noise', 'success_vs_error']:
        barmode = 'relative'
    else:
        barmode = None  # flow用

    fig.update_layout(
        title=f'Confusion Trends: {viz_style}',
        barmode=barmode,
        template='plotly_white',
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
        hovermode="x unified",
        width=1200,  # 横幅を広げる
        height=500
    )

    # 基準線 (0ライン)
    fig.add_hline(y=0, line_width=1, line_color="black")

    fig.show()

    # ★HTMLファイルとして保存
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, f"confusion_trends_{viz_style}.html")

    # include_plotlyjs='cdn' でライブラリ本体を除外し軽量化
    fig.write_html(save_path, include_plotlyjs='cdn')

    print(f"Saved plot to: {save_path} (Lightweight Mode)")


def visualize_history(csv_path):
    """
    CSVを読み込み、エポックごとの指標の変化を可視化する
    (例: Precision/Recallの推移、信頼度分布の推移など)
    """
    print(f"\n  [Visualizer] Visualizing history from {csv_path}...")

    if not os.path.exists(csv_path):
        print("  [Visualizer] ⚠️ CSV file not found.")
        return

    # ---------------------------------------------------------
    # TODO: ここに実装を追加してください
    #
    # 1. pandasでCSVを読み込む
    # 2. plotly.express.line や violin を使ってグラフを描画する
    # 3. HTMLファイルとして保存、または表示する
    # ---------------------------------------------------------

    # (動作確認用)
    df = pd.read_csv(csv_path)
    print(f"  [Visualizer] Loaded {len(df)} rows. (Visualization logic not implemented yet)")


def visualize_distributions(csv_path):
    """
    インスタンス単位の指標（信頼度、確率など）の分布を可視化する
    (例: TP/FPごとの信頼度ヒストグラムなど)
    """
    print(f"  [Visualizer] Visualizing distributions from {csv_path}...")

    # ---------------------------------------------------------
    # TODO: ここに実装を追加してください
    # ---------------------------------------------------------
