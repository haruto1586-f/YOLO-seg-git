# model_interface.py
from ultralytics import YOLO


def train_model_with_callback(base_model_path, data_yaml, epochs, imgsz, project_dir, run_name,
                              on_epoch_end_callback=None):
    """
    YOLOモデルを学習する。
    on_epoch_end_callback が指定された場合、各エポック終了時にその関数を実行する。
    """
    model = YOLO(base_model_path)

    # --- コールバックの登録 ---
    if on_epoch_end_callback:
        # Ultralyticsのコールバックシステムに登録
        # "on_train_epoch_end" は学習の1エポックが終わるたびに呼ばれるイベント名
        model.add_callback("on_train_epoch_end", on_epoch_end_callback)
        print("[Model] Custom callback registered for epoch analysis.")

    print(f"\n[Model] Starting training for {run_name} ({epochs} epochs)...")

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        project=project_dir,
        name=run_name,
        patience=0,
        save=True,
        val=True,
        exist_ok=True
    )

    return results.save_dir / 'weights' / 'best.pt', results.results_dict


def run_test_evaluation(trained_model_path, data_yaml, project_dir, run_name):
    """テストデータでの最終評価"""
    print(f"\n[Model] Starting TEST evaluation using {trained_model_path}...")
    model = YOLO(trained_model_path)
    metrics = model.val(data=data_yaml, split='test', project=project_dir, name=f"{run_name}_TEST_EVAL")
    return metrics.results_dict