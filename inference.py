import cv2
import numpy as np
import torch
import sys

sys.path.append('..')

import numpy as np
import torch
import yaml
from munch import Munch
import os
import os.path as osp
from svgnet.model.svgnet import SVGNet
from svgnet.util  import get_root_logger, load_checkpoint
import json
from svgnet.data import SVGDataset
import matplotlib.pyplot as plt
from svgnet.data.svg import SVG_CATEGORIES

from to_svg import preprocess_floorplan
from to_svg import analyze_cad_drawing


COMMANDS = ['Line', 'Arc','circle', 'ellipse']


# def extract_primitives_from_contours(contours, min_length=20, angle_threshold=5):
#     """
#     輪郭から基本要素（直線、円弧など）を抽出します
    
#     Parameters:
#     -----------
#     contours : list
#         cv2.findContoursの結果
#     min_length : int
#         最小長さ
#     angle_threshold : float
#         垂直/水平判定の角度閾値
        
#     Returns:
#     --------
#     dict : 検出された要素の辞書
#     """
#     elements = {
#         'commands': [],  # Line, Arc, circle, ellipse
#         'args': [],      # 4点のサンプリング点
#         'lengths': [],   # 要素の長さ
#         'semanticIds': [],  # セマンティックID（とりあえず全て wall=33）
#         'instanceIds': [],  # インスタンスID（連番）
#     }
    
#     instance_id = 0
    
#     for contour in contours:
#         # 輪郭を近似
#         epsilon = 0.01 * cv2.arcLength(contour, True)
#         approx = cv2.approxPolyDP(contour, epsilon, True)
#         points = approx.reshape(-1, 2)
        
#         # 輪郭の各セグメントを処理
#         for i in range(len(points)):
#             p1 = points[i]
#             p2 = points[(i + 1) % len(points)]
            
#             # 長さチェック
#             length = np.sqrt(np.sum((np.array(p2) - np.array(p1)) ** 2))
#             if length < min_length:
#                 continue
            
#             # 角度計算
#             angle = np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0])) % 180
            
#             # 垂直/水平判定
#             is_vertical = min(abs(angle - 90), abs(angle - 270)) < angle_threshold
#             is_horizontal = min(abs(angle), abs(angle - 180)) < angle_threshold
            
#             if is_vertical or is_horizontal:
#                 # 直線の場合
#                 if is_vertical:
#                     x = (p1[0] + p2[0]) / 2
#                     y1, y2 = min(p1[1], p2[1]), max(p1[1], p2[1])
#                     # 4点のサンプリング
#                     _points = [
#                         [x, y1],  # 始点
#                         [x, y1 + (y2-y1)/3],  # 1/3点
#                         [x, y1 + 2*(y2-y1)/3],  # 2/3点
#                         [x, y2]   # 終点
#                     ]
#                 else:
#                     y = (p1[1] + p2[1]) / 2
#                     x1, x2 = min(p1[0], p2[0]), max(p1[0], p2[0])
#                     _points = [
#                         [x1, y],
#                         [x1 + (x2-x1)/3, y],
#                         [x1 + 2*(x2-x1)/3, y],
#                         [x2, y]
#                     ]
                
#                 # データの追加
#                 elements['commands'].append(COMMANDS.index('Line'))
#                 elements['args'].append([p for point in _points for p in point])
#                 elements['lengths'].append(length)
#                 elements['semanticIds'].append(32)  # wall = 33, 0-based indexing
#                 elements['instanceIds'].append(instance_id)
#                 instance_id += 1
            
#             else:
#                 # 曲線（Arc）として処理
#                 mid_point = (p1 + p2) / 2
#                 # 中間点を少しずらして曲線を表現
#                 normal = np.array([-angle, angle]) / np.linalg.norm([-angle, angle])
#                 control_point = mid_point + normal * (length / 4)
                
#                 # 4点のサンプリング
#                 t_values = [0, 1/3, 2/3, 1.0]
#                 _points = []
#                 for t in t_values:
#                     # 2次ベジエ曲線のパラメトリック方程式
#                     point = (1-t)**2 * p1 + 2*(1-t)*t * control_point + t**2 * p2
#                     _points.append(point)
                
#                 elements['commands'].append(COMMANDS.index('Arc'))
#                 elements['args'].append([p for point in _points for p in point])
#                 elements['lengths'].append(length * 1.2)  # 曲線なので直線より少し長めに
#                 elements['semanticIds'].append(32)
#                 elements['instanceIds'].append(instance_id)
#                 instance_id += 1
    
#     # numpy配列に変換
#     for key in elements:
#         elements[key] = np.array(elements[key])
    
#     return elements


def extract_primitives_from_lines(lines, min_length=20, angle_threshold=5):
    """
    線分リストから基本要素（直線、円弧など）を抽出します
    
    Parameters:
    -----------
    lines : List[Line]
        線分のリスト
    min_length : int
        最小長さ
    angle_threshold : float
        垂直/水平判定の角度閾値
        
    Returns:
    --------
    dict : 検出された要素の辞書
    """
    elements = {
        'commands': [],  # Line, Arc, circle, ellipse
        'args': [],      # 4点のサンプリング点
        'lengths': [],   # 要素の長さ
        'semanticIds': [],  # セマンティックID（とりあえず全て wall=33）
        'instanceIds': [],  # インスタンスID（連番）
    }
    
    instance_id = 0
    
    for line in lines:
        # 始点と終点を取得
        p1 = np.array([line.start.x, line.start.y])
        p2 = np.array([line.end.x, line.end.y])
        
        # 長さチェック
        length = np.sqrt(np.sum((p2 - p1) ** 2))
        if length < min_length:
            continue
        
        # 角度計算
        angle = np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0])) % 180
        
        # 垂直/水平判定
        is_vertical = min(abs(angle - 90), abs(angle - 270)) < angle_threshold
        is_horizontal = min(abs(angle), abs(angle - 180)) < angle_threshold
        
        if is_vertical or is_horizontal:
            # 直線の場合
            if is_vertical:
                x = (p1[0] + p2[0]) / 2
                y1, y2 = min(p1[1], p2[1]), max(p1[1], p2[1])
                # 4点のサンプリング
                _points = [
                    [x, y1],  # 始点
                    [x, y1 + (y2-y1)/3],  # 1/3点
                    [x, y1 + 2*(y2-y1)/3],  # 2/3点
                    [x, y2]   # 終点
                ]
            else:
                y = (p1[1] + p2[1]) / 2
                x1, x2 = min(p1[0], p2[0]), max(p1[0], p2[0])
                _points = [
                    [x1, y],
                    [x1 + (x2-x1)/3, y],
                    [x1 + 2*(x2-x1)/3, y],
                    [x2, y]
                ]
            
            # データの追加
            elements['commands'].append(COMMANDS.index('Line'))
            elements['args'].append([p for point in _points for p in point])
            elements['lengths'].append(length)
            elements['semanticIds'].append(32)  # wall = 33, 0-based indexing
            elements['instanceIds'].append(instance_id)
            instance_id += 1
        
        else:
            # 曲線（Arc）として処理
            mid_point = (p1 + p2) / 2
            # 中間点を少しずらして曲線を表現
            normal = np.array([-angle, angle]) / np.linalg.norm([-angle, angle])
            control_point = mid_point + normal * (length / 4)
            
            # 4点のサンプリング
            t_values = [0, 1/3, 2/3, 1.0]
            _points = []
            for t in t_values:
                # 2次ベジエ曲線のパラメトリック方程式
                point = (1-t)**2 * p1 + 2*(1-t)*t * control_point + t**2 * p2
                _points.append(point)
            
            elements['commands'].append(COMMANDS.index('Arc'))
            elements['args'].append([p for point in _points for p in point])
            elements['lengths'].append(length * 1.2)  # 曲線なので直線より少し長めに
            elements['semanticIds'].append(32)
            elements['instanceIds'].append(instance_id)
            instance_id += 1
    
    # numpy配列に変換
    for key in elements:
        elements[key] = np.array(elements[key])
    
    return elements


def detect_and_convert_to_json(preprocessed_results):
    """
    前処理済み画像から要素を検出しJSONデータに変換します
    
    Parameters:
    -----------
    preprocessed_results : dict
        前処理結果
        
    Returns:
    --------
    dict : JSON形式のデータ
    """
    binary = preprocessed_results['binary']
    height, width = binary.shape

    # # 輪郭検出
    # contours, _ = cv2.findContours(binary, cv2.RETR_LIST, 
    #                              cv2.CHAIN_APPROX_SIMPLE)

    lines, _, _ = analyze_cad_drawing(image_path, min_line_length=10)

    # 要素の抽出
    elements = extract_primitives_from_lines(lines)

    # JSONデータの作成
    json_dicts = {
        "commands": elements['commands'].tolist(),
        "args": elements['args'].tolist(),
        "lengths": elements['lengths'].tolist(),
        "semanticIds": elements['semanticIds'].tolist(),
        "instanceIds": elements['instanceIds'].tolist(),
        "width": width,
        "height": height,
        "rgb": [[0, 178, 0]] * len(elements['commands']),  # デフォルトカラー
        "layerIds": list(range(len(elements['commands']))),
        "widths": [0.1] * len(elements['commands'])  # デフォルトの線幅
    }
    return json_dicts


class SVGInferencePipeline:
    def __init__(self, output_dir):
        """
        SVG推論パイプラインの初期化
        """
        self.output_dir = output_dir

        workdir="./work_dirs/svg/svg_pointT/baseline_nclsw_grelu"
        config_path = osp.join(workdir, "svg_pointT.yaml")
        checkpoint = osp.join(workdir, "best.pth")

        self.logger = get_root_logger()

        # 設定の読み込み
        cfg_txt = open(config_path, "r").read()
        self.cfg = Munch.fromDict(yaml.safe_load(cfg_txt))
        self.device = 'cuda'

        # モデルの初期化と重みの読み込み
        self.model = SVGNet(self.cfg.model).to(self.device)
        load_checkpoint(checkpoint, self.logger, self.model)
        self.model.eval()

        self.logger.info(f"Model loaded from {checkpoint}")

    def process_image(self, image_path):
        """
        画像を処理してSVGを生成し、モデルで推論を行います
        
        Parameters:
        -----------
        image_path : str
            入力画像のパス
        output_dir : str, optional
            出力ディレクトリ
            
        Returns:
        --------
        dict : 推論結果
        """
        # 画像の前処理
        preprocessed = preprocess_floorplan(image_path)

        # SVG要素の検出とJSON形式への変換
        json_data = detect_and_convert_to_json(preprocessed)
        
        # JSONファイルの一時保存（必要な場合）
        os.makedirs(self.output_dir, exist_ok=True)
        temp_json = osp.join(self.output_dir, 'temp.json')
        with open(temp_json, 'w') as f:
            json.dump(json_data, f, indent=4)

        # モデル入力の準備
        coord, feat, label, offset, lengths = self.prepare_model_input(json_data)
        
        # 推論の実行
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.cfg.fp16):
                batch = (
                    coord.to(self.device),
                    feat.to(self.device),
                    label.to(self.device) if label is not None else None,
                    offset.to(self.device),
                    lengths.to(self.device) if lengths is not None else None
                )
                # 0 torch.Size([1, 2048, 3])
                # 1 torch.Size([1, 2048, 6])
                # 2 torch.Size([1, 2048, 2])
                # 3 torch.Size([1])
                # 4 torch.Size([1, 2048])

                # 0 torch.Size([17173, 3])
                # 1 torch.Size([17173, 6])
                # 2 torch.Size([17173, 2])
                # 3 torch.Size([8])
                # 4 torch.Size([17173])
                results = self.model(batch, return_loss=False)

        self.logger.info("Inference done")
        self.logger.info(results)

        return self.postprocess_results(results, json_data)

    def prepare_model_input(self, json_data):
        """
        JSONデータからモデル入力を準備します
        
        Parameters:
        -----------
        json_data : dict
            検出されたSVG要素のデータ
            
        Returns:
        --------
        tuple : モデルの入力テンソル
        """
        coord, feat, label,lengths = SVGDataset.load(json_data, 0, min_points=2048)
        coord -= np.mean(coord, 0)
        coord = torch.FloatTensor(coord)
        feat = torch.FloatTensor(feat)
        label = torch.LongTensor(label)
        lengths = torch.FloatTensor(lengths)

        offset, count = [], 0
        for item in [coord]:
            count += item.shape[0]
            offset.append(count)
        # lengths = torch.cat(lengths) if lengths is not None else None

        # PyTorchテンソルに変換
        return (torch.FloatTensor(coord),
                torch.FloatTensor(feat),
                torch.LongTensor(label) if label is not None else None,
                torch.IntTensor(offset),
                torch.FloatTensor(lengths) if lengths is not None else None)

    def postprocess_results(self, results, original_data):
        """
        モデルの出力を処理して可視化します
        
        Parameters:
        -----------
        results : dict
            モデルの出力
        original_data : dict
            元のSVGデータ
            
        Returns:
        --------
        dict : 後処理された結果
        """
        processed_results = {}
        
        # セマンティックセグメンテーション結果の処理
        if "semantic_scores" in results:
            sem_preds = torch.argmax(results["semantic_scores"], dim=1).cpu().numpy()
            processed_results["semantic_predictions"] = sem_preds
        
        # インスタンスセグメンテーション結果の処理
        if "instances" in results:
            processed_results["instance_predictions"] = results["instances"]

        # 結果の可視化
        height = original_data['height']
        width = original_data['width']
        save_path = osp.join(self.output_dir, 'inference_result.png')
        visualize_results(processed_results, original_data, 
                        (height, width), save_path)

        return processed_results


def visualize_results(results, original_data, image_shape, save_path):
    """
    推論結果を可視化します
    
    Parameters:
    -----------
    results : dict
        モデルの推論結果
    original_data : dict
        元のSVGデータ
    image_shape : tuple
        元画像のサイズ (height, width)
    save_path : str, optional
        保存先のパス
    """
    # SVGカテゴリーごとの色の定義
    category_colors = {
        item['id']: item['color'] for item in SVG_CATEGORIES
    }
    category_colors[0] = (128, 128, 128)  # 未分類
    category_colors[35] = (0, 0, 0)  # 背景

    category_names = {
        item['id']: item['name'] for item in SVG_CATEGORIES
    }
    category_names[0] = 'unknown'
    category_names[35] = 'background'

    # 可視化用の画像を作成
    height, width = image_shape
    semantic_vis = np.zeros((height, width, 3), dtype=np.uint8)
    instance_vis = np.zeros((height, width, 3), dtype=np.uint8)
    combined_vis = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 元のSVG要素の座標を復元
    args = np.array(original_data["args"]).reshape(-1, 8)
    num_elements = len(args)
    
    # セマンティックセグメンテーション結果の描画
    unique_classes = set()
    semantic_preds = results['semantic_predictions']
    for i in range(num_elements):
        if i >= len(semantic_preds):
            break
            
        # 要素の座標を取得
        coords = args[i].reshape(4, 2)
        pred_class = semantic_preds[i]
        color = category_colors.get(pred_class, (128, 128, 128))
        unique_classes.add(pred_class)
        
        # コマンドタイプに応じて描画
        command_type = original_data['commands'][i]
        if command_type == COMMANDS.index('Line'):
            # 直線の描画
            cv2.line(semantic_vis, 
                    tuple(coords[0].astype(int)), 
                    tuple(coords[-1].astype(int)),
                    color, 2)
        elif command_type == COMMANDS.index('Arc'):
            # 円弧の描画
            points = coords.astype(np.int32)
            cv2.polylines(semantic_vis, [points], False, color, 2)
        elif command_type in [COMMANDS.index('circle'), COMMANDS.index('ellipse')]:
            # 円または楕円の描画
            center = np.mean(coords, axis=0).astype(int)
            axes = (int(original_data['lengths'][i] / 2), 
                   int(original_data['lengths'][i] / 2))
            cv2.ellipse(semantic_vis, center, axes, 0, 0, 360, color, 2)
    # 凡例を色とともに表示
    for i, _cls in enumerate(unique_classes):
        cv2.putText(semantic_vis, category_names[_cls], (10, 30 * (i + 1)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, category_colors[_cls], 2)
    
    # インスタンスセグメンテーション結果の描画
    if 'instance_predictions' in results:
        for inst in results['instance_predictions']:
            masks = inst['masks']
            label = inst['labels']
            score = inst['scores']
            
            if score > 0.5:  # スコアの閾値
                color = category_colors.get(label, (128, 128, 128))
                for i in range(num_elements):
                    if masks[i]:
                        coords = args[i].reshape(4, 2)
                        # 要素の描画（セマンティック描画と同様）
                        command_type = original_data['commands'][i]
                        if command_type == COMMANDS.index('Line'):
                            cv2.line(instance_vis,
                                   tuple(coords[0].astype(int)),
                                   tuple(coords[-1].astype(int)),
                                   color, 2)
                        # ... 他のコマンドタイプの描画
    
    # 結果の合成
    combined_vis = cv2.addWeighted(semantic_vis, 0.7, instance_vis, 0.3, 0)
    
    # 結果の表示と保存
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(semantic_vis)
    plt.title('Semantic Segmentation')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(instance_vis)
    plt.title('Instance Segmentation')
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(combined_vis)
    plt.title('Combined Result')
    plt.axis('off')
    
    plt.tight_layout()
    
    plt.savefig(save_path)
    # 個別の結果も保存
    cv2.imwrite(save_path.replace('.png', '_semantic.png'), 
                cv2.cvtColor(semantic_vis, cv2.COLOR_RGB2BGR))
    cv2.imwrite(save_path.replace('.png', '_instance.png'),
                cv2.cvtColor(instance_vis, cv2.COLOR_RGB2BGR))
    cv2.imwrite(save_path.replace('.png', '_combined.png'),
                cv2.cvtColor(combined_vis, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    image_path = os.path.join(os.path.dirname(__file__), '_input/sample.png')
    output_dir = os.path.join(os.path.dirname(__file__), '_output')

    print(f"Process image: {image_path}")
    pipeline = SVGInferencePipeline(output_dir=output_dir)
    results = pipeline.process_image(image_path)
    print(results)
