import os
import shutil
import glob
import re
import yaml
import json
from pathlib import Path


def get_image_list(image_dir):
    """指定されたディレクトリから画像リストを取得する"""
    patterns = [os.path.join(image_dir, f'*.{ext}') for ext in ('jpg', 'jpeg', 'png', 'bmp', 'tif')]
    images = []
    for p in patterns:
        images.extend(glob.glob(p))
    return images


def _find_label_file(img_path: str) -> str:
    """画像パスに対応するYOLOラベルファイルのパスを見つける"""
    label_path_str = img_path.replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}")
    label_path = Path(label_path_str).with_suffix(".txt")
    return str(label_path)


def get_dataset_pools():
    """データセットをダウンロードし、Train/Val/Test の画像リストとYAML情報を返す"""
    print(f"[Data Manager] Checking/Downloading dataset...")

    from ultralytics.utils.checks import check_yaml as check_dataset_file
    from ultralytics.utils import SETTINGS
    from ultralytics.data.utils import download
    from configs import crack as cfg  # 設定ファイルを参照

    yaml_config_path = check_dataset_file(cfg.DATASET_YAML_NAME)

    try:
        with open(yaml_config_path, 'r') as f:
            data_config = yaml.safe_load(f)
    except Exception as e:
        raise FileNotFoundError(f"Failed to read YAML config at {yaml_config_path}. Error: {e}")

    dataset_dir = Path(SETTINGS.get('datasets_dir', '/content/datasets'))

    if 'download' in data_config:
        print(f"Ensuring dataset is downloaded via {data_config['download']}...")
        download(url=data_config['download'], dir=dataset_dir, unzip=True)

    dataset_name = Path(data_config.get('path', '../datasets/' + Path(cfg.DATASET_YAML_NAME).stem)).name
    dataset_root = dataset_dir / dataset_name

    print(f"[Data Manager] Resolving dataset root at: {dataset_root}")

    if not dataset_root.exists() or not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found at {dataset_root}.")

    def resolve_split_path(split_key):
        if split_key not in data_config: return None
        path_val = data_config[split_key]
        if os.path.isabs(path_val):
            return Path(path_val)
        else:
            return dataset_root / path_val

    train_dir = resolve_split_path('train')
    val_dir = resolve_split_path('val')
    test_dir = resolve_split_path('test')

    pools = {}
    if train_dir and train_dir.exists():
        pools['train'] = get_image_list(str(train_dir))
    if val_dir and val_dir.exists():
        pools['val'] = get_image_list(str(val_dir))
    if test_dir and test_dir.exists():
        pools['test'] = get_image_list(str(test_dir))
    else:
        pools['test'] = []

    data_config['path'] = str(dataset_root)
    return pools, data_config


def setup_batch_workspace(data_config: dict, al_dataset_dir: str, al_data_yaml_path: str):
    """ワークスペースの作成とdata.yamlの配置"""
    print(f"[Data Manager] Setting up batch workspace at {al_dataset_dir}")

    if os.path.exists(al_dataset_dir):
        print(f"  -> Cleaning up old workspace: {al_dataset_dir}")
        shutil.rmtree(al_dataset_dir)

    splits = ['train', 'val', 'test']
    for split in splits:
        os.makedirs(os.path.join(al_dataset_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(al_dataset_dir, 'labels', split), exist_ok=True)

    names_dict = data_config['names']
    nc_calculated = len(names_dict)

    al_yaml_content = {
        'path': os.path.abspath(al_dataset_dir),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': nc_calculated,
        'names': names_dict
    }

    with open(al_data_yaml_path, 'w') as f:
        yaml.dump(al_yaml_content, f, default_flow_style=False)

    print(f"[Data Manager] Created batch data.yaml at {al_data_yaml_path}")


def _copy_files(image_paths: list, dst_img_dir: str, dst_lbl_dir: str):
    """画像とラベルのコピー"""
    if not image_paths: return
    print(f"  -> Copying {len(image_paths)} files to {os.path.basename(dst_img_dir)}...")

    cache_path = Path(dst_lbl_dir) / f"{Path(dst_lbl_dir).name}.cache"
    if cache_path.exists():
        os.remove(cache_path)

    for src_img_path in image_paths:
        src_lbl_path = _find_label_file(src_img_path)
        if os.path.exists(src_img_path) and os.path.exists(src_lbl_path):
            shutil.copy(src_img_path, dst_img_dir)
            shutil.copy(src_lbl_path, dst_lbl_dir)


def populate_batch_workspace(pools: dict, al_dataset_dir: str):
    """ワークスペースへのデータコピー"""
    print("[Data Manager] Populating workspace with dataset files...")
    _copy_files(pools.get('train', []), os.path.join(al_dataset_dir, 'images', 'train'),
                os.path.join(al_dataset_dir, 'labels', 'train'))
    _copy_files(pools.get('val', []), os.path.join(al_dataset_dir, 'images', 'val'),
                os.path.join(al_dataset_dir, 'labels', 'val'))
    _copy_files(pools.get('test', []), os.path.join(al_dataset_dir, 'images', 'test'),
                os.path.join(al_dataset_dir, 'labels', 'test'))


# =========================================================
# データプール管理関数
# =========================================================

def save_data_pools(data_pools, save_dir):
    """データプールの中身(パスリスト)をJSONとして保存する"""
    json_path = os.path.join(save_dir, 'data_pools.json')
    try:
        with open(json_path, 'w') as f:
            json.dump(data_pools, f, indent=4)
        print(f"[Data Manager] Saved data_pools to {json_path}")
    except Exception as e:
        print(f"[Data Manager] Warning: Failed to save data_pools. {e}")


def load_data_pools(exp_dir):
    """
    実験ディレクトリから data_pools.json を読み込む。
    なければ None を返す。
    """
    json_path = os.path.join(exp_dir, 'data_pools.json')
    if os.path.exists(json_path):
        print(f"[Data Manager] Loading data_pools from {json_path}...")
        try:
            with open(json_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"[Data Manager] Error loading JSON: {e}")
    return None


# =========================================================
# パス・ディレクトリ管理関数
# =========================================================
def increment_path(path, exist_ok=False, sep='', mkdir=False):
    """
    パスをインクリメントする (例: runs/exp -> runs/exp1, runs/exp2...)
    """
    path = Path(path)
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
        dirs = glob.glob(f"{path}{sep}*")
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        path = Path(f"{path}{sep}{n}{suffix}")
    
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)
    return path


def find_latest_experiment_dir(base_dir, prefix='train_exp'):
    """
    runsディレクトリの中から、最新の実験フォルダ(数値が最大のもの)を見つける
    """
    if not os.path.exists(base_dir): return None
    
    candidates = glob.glob(os.path.join(base_dir, f"{prefix}*"))
    if not candidates: return None
    
    def extract_number(path):
        name = os.path.basename(path)
        match = re.search(rf"{prefix}(\d+)", name)
        return int(match.group(1)) if match else 1

    return sorted(candidates, key=extract_number)[-1]