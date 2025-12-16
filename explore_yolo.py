import os
import glob
import torch
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import yaml
import plotly.express as px


def get_test_data(num_images=3):
    """
    COCO128-segをダウンロードし、テスト用の画像パスとクラス名を返す。
    """
    print("[Setup] COCO128-segデータセットをチェック・ダウンロード中...")
    
    from ultralytics.utils.checks import check_yaml as check_dataset_file
    from ultralytics.utils import SETTINGS
    from ultralytics.data.utils import download
    
    yaml_config_path = check_dataset_file('coco128-seg.yaml')

    try:
        with open(yaml_config_path, 'r') as f:
            coco_data = yaml.safe_load(f)
    except Exception as e:
        raise FileNotFoundError(f"Failed to read YAML config at {yaml_config_path}. Error: {e}")

    dataset_dir = Path(SETTINGS.get('datasets_dir', '/content/datasets'))
    
    if 'download' in coco_data:
        download(url=coco_data['download'], dir=dataset_dir, unzip=True)

    dataset_name = Path(coco_data['path']).name 
    coco_root = dataset_dir / dataset_name
    
    if not coco_root.exists() or not coco_root.is_dir():
         raise FileNotFoundError(f"Dataset directory not found at {coco_root}")
    
    train_img_dir = str(coco_root / coco_data['train'])
    image_pool = sorted(glob.glob(os.path.join(train_img_dir, '*.jpg')))
    
    class_names = coco_data.get('names', {})
    if not isinstance(class_names, dict):
        class_names = {i: name for i, name in enumerate(class_names)}

    print(f"[Setup] {len(image_pool)}枚の画像を発見。")
    return image_pool[:num_images], class_names


def analyze_confidence_and_bbox(image_result, class_names):
    """
    [演習1] 各オブジェクトの「信頼度(conf)」と「BBox」を分析する
    (変更なし)
    """
    print("\n  ### 1. BBoxと信頼度 (boxes.conf) の分析 ###")
    
    boxes = image_result.boxes
    if boxes is None or len(boxes) == 0:
        print("    -> この画像ではオブジェクトが検出されませんでした。")
        return

    print(f"    -> {len(boxes)} 個のオブジェクトを検出しました。")

    for i in range(len(boxes)):
        box = boxes[i]
        conf = box.conf.item()
        cls_id = int(box.cls.item())
        class_name = class_names.get(cls_id, "Unknown")
        xyxy = box.xyxy.cpu().numpy()[0]
        
        print(f"\n    [オブジェクト {i}]")
        print(f"    - クラス: {class_name} (ID: {cls_id})")
        print(f"    - 信頼度 (conf): {conf:.4f}")
        print(f"    - BBox (xyxy): {xyxy}")

    # ---
    # TODO: ここに実装を追加
    # (例1) 検出された全オブジェクトの信頼度をリストにまとめ、平均値や標準偏差を計算する
    # (例2) 信頼度をCSVファイルに保存する
    # (例3) `pandas` と `plotly.express.histogram` を使って、信頼度の分布グラフを作成する
    # ---


def analyze_masks_and_logits(image_result, class_names):
    """
    [演習2] 各オブジェクトの「マスク」と「ロジット/確率」を分析する
    """
    print("\n  ### 2. マスクとロジット/確率 (masks.data) の分析 ###")
    
    masks = image_result.masks
    if masks is None or len(masks) == 0:
        print("    -> この画像ではマスクが検出されませんでした。")
        return

    for i in range(len(masks)):
        mask_obj = masks[i]
        
        logit_tensor = mask_obj.data[0] 
        prob_tensor = torch.sigmoid(logit_tensor)
        
        # テンソルを CPU に移動し、Numpy 配列に変換 (Plotlyで表示するため)
        prob_heatmap = prob_tensor.cpu().numpy()
        
        print(f"\n    [オブジェクト {i}]")
        print(f"    - ロジット (Logit) mean: {logit_tensor.mean().item():.4f} (0に近いほど不確実)")
        print(f"    - 確率 (Probability) mean: {prob_tensor.mean().item():.4f} (0.5に近いほど不確実)")

        # ---
        # TODO: ここに実装を追加 (◀️ 変更)
        # (例1) 各オブジェクトの「平均確率(mean_probability)」をCSVに保存する
        
        # (例2) `plotly.express.imshow` を使って、
        #       確率テンソルをヒートマップとしてブラウザで表示する
        
        # fig_heatmap = px.imshow(
        #     prob_heatmap, 
        #     title=f"画像 {os.path.basename(image_result.path)} - オブジェクト {i} の確率ヒートマップ",
        #     color_continuous_scale='Viridis' # カラースケール
        # )
        # fig_heatmap.show() # ブラウザでインタラクティブなヒートマップが開く
        
        # (例3) ロジットや確率の平均値が、`boxes.conf`（全体の信頼度）とどう関係するか考察する
        # ---


def visualize_results(image_result):
    """
    [演習3] 結果を可視化・保存する
    """
    print("\n  ### 3. 可視化 ###")
    
    # 1. YOLO標準の描画機能 (BGR形式のNumpy配列)
    rendered_image_bgr = image_result.plot()
    
    # 2. 保存
    if image_result.save_dir:
        print(f"    -> 標準の描画結果が {image_result.save_dir} に保存されました。")
    else:
        print("    -> (model.predictに save=True を指定すると結果が画像保存されます)")

    # ---
    # TODO: ここに実装を追加 (◀️ 変更)
    # (例1) `plotly.express.imshow` を使って、結果をブラウザで表示する
    #       (PlotlyはRGB形式を期待するため、BGR -> RGB への変換が必要)
    
    # # BGR配列をRGB配列に変換
    # rendered_image_rgb = rendered_image_bgr[..., ::-1]
    
    # # Plotlyで表示
    # fig_result = px.imshow(
    #     rendered_image_rgb,
    #     title=f"描画結果: {os.path.basename(image_result.path)}"
    # )
    # fig_result.show() # ブラウザでインタラクティブな画像が開く

    # (例2) 好きなファイル名で結果を保存する (Pillowを使う例)
    # try:
    #     from PIL import Image
    #     pil_image = Image.fromarray(rendered_image_rgb)
    #     save_path = f"my_result_{os.path.basename(image_result.path)}"
    #     pil_image.save(save_path)
    #     print(f"    -> Pillowで {save_path} に画像を保存しました。")
    # except ImportError:
    #     print("    -> 画像を保存するには `pip install Pillow` が必要です。")
    # ---


# -------------------------------------------------------------------
# メイン実行ブロック (変更なし)
# -------------------------------------------------------------------
def main():
    print("--- 🔬 YOLO Instance Segmentation 探訪スクリプト ---")
    
    # 1. モデルとデータの準備
    model = YOLO('yolov8n-seg.pt') # 事前学習済みモデル
    test_image_paths, class_names = get_test_data(num_images=3) # 3枚の画像を取得

    if not test_image_paths:
        print("エラー: テスト画像が見つかりません。")
        return

    # 2. 推論の実行
    print(f"\n--- {len(test_image_paths)}枚の画像に推論を実行 ---")
    results_list = model.predict(source=test_image_paths, save=True, conf=0.25)
    
    # 3. 画像ごとのループ処理
    for r in results_list:
        print("\n" + "="*50)
        print(f"画像: {os.path.basename(r.path)}")
        print(f"元の解像度: {r.orig_shape}")
        print("="*50)
        
        # 4. 各分析関数（ブランク関数）の呼び出し
        
        # [演習1] BBoxと信頼度(conf)の分析
        analyze_confidence_and_bbox(r, class_names)
        
        # [演習2] マスクとロジット/確率(mask.data)の分析
        analyze_masks_and_logits(r, class_names)
        
        # [演習3] 可視化
        visualize_results(r)

    print("\n--- 探訪スクリプト 正常終了 ---")


if __name__ == "__main__":
    main()