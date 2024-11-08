from collections import defaultdict
import cv2
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from svgwrite import Drawing
import math
import json
import sys
import os
import copy

# # プロジェクトのルートディレクトリのパスを取得
# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(current_dir)
# lcnn_dir = os.path.join(parent_dir, 'lcnn')

# # パスを追加
# sys.path.append(parent_dir)
# sys.path.append(lcnn_dir)

from lcnn.config import C, M


def preprocess_floorplan(image_path, target_size=(140, 140), visualize=True):
    """
    建築図面の画像を前処理し、線分検出に適した形式に変換します
    
    Parameters:
    -----------
    image_path : str
        入力画像のパス
    target_size : tuple
        出力画像のサイズ（幅, 高さ）
    visualize : bool
        前処理の各ステップを可視化するかどうか
        
    Returns:
    --------
    dict : 前処理された画像と中間結果を含む辞書
    """
    # 画像の読み込み
    original = cv2.imread(image_path)
    if original is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    print(original.shape)

    # リサイズ
    # resized = cv2.resize(original, target_size, interpolation=cv2.INTER_AREA)
    resized = original
    
    # グレースケール変換
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    # # ノイズ除去
    # denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

    # コントラスト調整
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contrast_enhanced = clahe.apply(gray)

    # 二値化
    # binary = cv2.adaptiveThreshold(contrast_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #         cv2.THRESH_BINARY, 11, 2)
    _, binary = cv2.threshold(contrast_enhanced, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # モルフォロジー演算でノイズ除去と線の連結
    # kernel = np.ones((2,2), np.uint8)
    # morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # エッジ検出
    edges = cv2.Canny(binary, 50, 150)

    # # 細線化（線を1ピクセル幅に）
    # thinned = cv2.ximgproc.thinning(binary)

    results = {
        'original': cv2.cvtColor(original, cv2.COLOR_BGR2RGB),
        'resized': cv2.cvtColor(resized, cv2.COLOR_BGR2RGB),
        'gray': gray,
        # 'denoised': denoised,
        'contrast': contrast_enhanced,
        'binary': binary,
        # 'morph': morph,
        'edges': edges,
        # 'thinned': thinned,
    }
    
    if visualize:
        visualize_preprocessing(results)

    return results


def visualize_preprocessing(results):
    """
    前処理の各ステップを可視化
    
    Parameters:
    -----------
    results : dict
        preprocess_floorplanの返り値
    """
    # プロット設定
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Floor Plan Preprocessing Steps', fontsize=16)
    
    # 各ステップの結果をプロット
    titles = ['Original', 'Resized', 'Grayscale',
             'Contrast Enhanced', 'Binary',
            'Edges',]
    
    images = [results[k] for k in ['original', 'resized', 'gray',
                                  'contrast', 'binary',
                                  'edges']]
    
    for ax, img, title in zip(axes.flat, images, titles):
        ax.imshow(img, cmap='gray' if img.ndim == 2 else None)
        ax.axis('off')
        ax.set_title(title)
    
    plt.tight_layout()
    plt.show()


def detect_lines(preprocessed_results, min_line_length=20, max_line_gap=8):
    """
    前処理された画像から線分を検出します
    
    Parameters:
    -----------
    preprocessed_results : dict
        preprocess_floorplanの返り値
    min_line_length : int
        検出する線分の最小長さ
    max_line_gap : int
        線分間の最大ギャップ
    
    Returns:
    --------
    tuple : (検出された線分, 可視化用の画像)
    """
    edges = preprocessed_results['edges']
    
    # Hough変換による線分検出
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 20,
                           minLineLength=min_line_length,
                           maxLineGap=max_line_gap)
    
    # 検出結果の可視化
    visualization = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(visualization, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    return lines, visualization


def merge_overlapping_lines(lines, angle_threshold=5, distance_threshold=3, visualize=True, img_shape=(140, 140)):
    """
    重複する線分をマージします
    
    Parameters:
    -----------
    lines : numpy.ndarray
        HoughLinesP で検出された線分の配列 shape: (N, 1, 4)
    angle_threshold : float
        同じ方向とみなす最大角度差（度）
    distance_threshold : float
        線分間の最大許容距離
    visualize : bool
        結果を可視化するかどうか
    img_shape : tuple
        元の画像のサイズ (height, width)
        
    Returns:
    --------
    numpy.ndarray : マージされた線分の配列
    """
    if lines is None or len(lines) == 0:
        return None
    
    # 線分の配列を整形 (N, 4) の形式に
    lines = lines.reshape(-1, 4)
    
    # 各線分の特徴を計算
    def get_line_features(line):
        x1, y1, x2, y2 = line
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        if angle < 0:
            angle += 180
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return angle, length
    
    # 線分を方向でグループ化
    grouped_lines = {}
    for i, line in enumerate(lines):
        angle, _ = get_line_features(line)
        
        # 既存のグループを探す
        found_group = False
        for group_angle in list(grouped_lines.keys()):
            if abs(angle - group_angle) < angle_threshold:
                grouped_lines[group_angle].append(i)
                found_group = True
                break
        
        # 新しいグループを作成
        if not found_group:
            grouped_lines[angle] = [i]
    
    # 各グループ内で重複する線分をマージ
    merged_lines = []
    for group_indices in grouped_lines.values():
        group_lines = lines[group_indices]
        
        # グループ内の線分を処理
        while len(group_lines) > 0:
            base_line = group_lines[0]
            overlapping = [0]  # 重複している線分のインデックス
            
            # 基準線分を表す点群を生成
            num_points = 100
            base_x = np.linspace(base_line[0], base_line[2], num_points)
            base_y = np.linspace(base_line[1], base_line[3], num_points)
            base_points = np.column_stack((base_x, base_y))
            
            # 他の線分との重複チェック
            for i in range(1, len(group_lines)):
                line = group_lines[i]
                # 比較する線分の点群を生成
                x = np.linspace(line[0], line[2], num_points)
                y = np.linspace(line[1], line[3], num_points)
                points = np.column_stack((x, y))
                
                # 点群間の最小距離を計算
                distances = cdist(base_points, points)
                min_distance = np.min(distances)
                
                if min_distance < distance_threshold:
                    overlapping.append(i)
            
            # 重複する線分をマージ
            overlapping_lines = group_lines[overlapping]
            
            # マージされた線分の端点を計算
            all_points = np.vstack([
                overlapping_lines[:, :2],  # 始点
                overlapping_lines[:, 2:]   # 終点
            ])
            
            # 主方向を計算
            angle = np.arctan2(base_line[3] - base_line[1],
                             base_line[2] - base_line[0])
            
            # 点を主方向に投影
            projected_vals = all_points[:, 0] * np.cos(angle) + all_points[:, 1] * np.sin(angle)
            
            # 最小と最大の投影値に対応する点を見つける
            min_idx = np.argmin(projected_vals)
            max_idx = np.argmax(projected_vals)
            merged_line = np.array([
                all_points[min_idx, 0], all_points[min_idx, 1],
                all_points[max_idx, 0], all_points[max_idx, 1]
            ])
            
            merged_lines.append(merged_line)
            
            # 処理済みの線分を削除
            group_lines = np.delete(group_lines, overlapping, axis=0)
    
    merged_lines = np.array(merged_lines)
    
    if visualize:
        visualize_merged_lines(lines, merged_lines, img_shape)
    
    return merged_lines


def visualize_merged_lines(original_lines, merged_lines, img_shape):
    """
    マージ前後の線分を可視化します
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # マージ前の線分を描画
    ax1.set_xlim(0, img_shape[1])
    ax1.set_ylim(img_shape[0], 0)
    for line in original_lines:
        x1, y1, x2, y2 = line
        ax1.plot([x1, x2], [y1, y2], 'b-', alpha=0.5, linewidth=1)
    ax1.set_title(f'Original Lines: {len(original_lines)}')
    ax1.grid(True)
    
    # マージ後の線分を描画
    ax2.set_xlim(0, img_shape[1])
    ax2.set_ylim(img_shape[0], 0)
    for line in merged_lines:
        x1, y1, x2, y2 = line
        ax2.plot([x1, x2], [y1, y2], 'r-', alpha=0.5, linewidth=1)
    ax2.set_title(f'Merged Lines: {len(merged_lines)}')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()


def detect_shapes(preprocessed_results):
    """
    前処理された画像から各種図形要素を検出します
    
    Parameters:
    -----------
    preprocessed_results : dict
        前処理結果
        
    Returns:
    --------
    dict : 検出された図形要素
    """
    binary = preprocessed_results['binary']
    edges = preprocessed_results['edges']
    
    # 輪郭検出
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, 
                                  cv2.CHAIN_APPROX_SIMPLE)
    
    # 線分検出
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 20,
                           minLineLength=20, maxLineGap=8)
    if lines is not None:
        lines = merge_overlapping_lines(lines, 
                                      angle_threshold=5,
                                      distance_threshold=3,
                                      visualize=False)
    
    # 円検出
    circles = cv2.HoughCircles(preprocessed_results['gray'],
                             cv2.HOUGH_GRADIENT, 1, 20,
                             param1=50, param2=30,
                             minRadius=5, maxRadius=50)
    
    # 矩形検出
    rectangles = []
    for contour in contours:
        # 輪郭を近似
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # 4点の場合は矩形として処理
        if len(approx) == 4:
            # 矩形らしさをチェック
            rect = cv2.minAreaRect(approx)
            area = cv2.contourArea(approx)
            if area > 100:  # 小さすぎる矩形は除外
                rectangles.append(approx)
    
    return {
        'lines': lines,
        'circles': circles,
        'rectangles': rectangles,
        'contours': contours
    }


def convert_to_svg(image_path, output_path):
    """
    建築図面をSVGに変換します
    
    Parameters:
    -----------
    image_path : str
        入力画像のパス
    output_path : str
        出力SVGのパス

    Returns:
    --------
    tuple : (preprocessed_results, shapes)
        前処理結果と検出された図形要素
    """
    # 前処理
    preprocessed = preprocess_floorplan(image_path, visualize=True)
    height, width = preprocessed['original'].shape[:2]
    
    # 図形要素の検出
    shapes = detect_shapes(preprocessed)
    
    # SVGドキュメントの作成
    dwg = Drawing(output_path, size=(width, height))
    
    # スタイル定義
    dwg.defs.add(dwg.style("""
        .wall { stroke: #000000; stroke-width: 1; fill: none; }
        .window { stroke: #0000FF; stroke-width: 1; fill: none; }
        .door { stroke: #FF0000; stroke-width: 1; fill: none; }
    """))
    
    # 線分の追加
    if shapes['lines'] is not None:
        for line in shapes['lines']:
            x1, y1, x2, y2 = line
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            # 線分の長さと角度を計算
            length = math.sqrt((x2-x1)**2 + (y2-y1)**2)
            angle = math.degrees(math.atan2(y2-y1, x2-x1))
            
            # 壁らしい線分（長い直線）は太く描画
            if length > 100:
                dwg.add(dwg.line(start=(x1, y1), end=(x2, y2),
                               class_='wall'))
            else:
                dwg.add(dwg.line(start=(x1, y1), end=(x2, y2),
                               stroke='black', stroke_width=0.5))
    
    # # 円の追加（窓や設備記号の可能性）
    # if shapes['circles'] is not None:
    #     circles = np.uint16(np.around(shapes['circles']))
    #     for i in circles[0, :]:
    #         center = (i[0], i[1])
    #         radius = i[2]
    #         dwg.add(dwg.circle(center=center, r=radius,
    #                          class_='window'))
    
    # # 矩形の追加（ドアや窓の可能性）
    # for rect in shapes['rectangles']:
    #     # 矩形の特徴を分析
    #     rect_points = rect.reshape(-1, 2)
    #     width = np.linalg.norm(rect_points[0] - rect_points[1])
    #     height = np.linalg.norm(rect_points[1] - rect_points[2])
    #     ratio = min(width, height) / max(width, height)
        
    #     # ドアや窓らしい矩形を選別
    #     if 0.1 < ratio < 0.5:  # 細長い矩形はドアの可能性
    #         points = rect.reshape(-1, 2)
    #         path_data = 'M {} {} '.format(points[0][0], points[0][1])
    #         path_data += ' '.join('L {} {}'.format(x, y) for x, y in points[1:])
    #         path_data += ' Z'
    #         dwg.add(dwg.path(d=path_data, class_='door'))
    
    # その他の輪郭の追加
    for contour in shapes['contours']:
        # 小さすぎる輪郭は除外
        if cv2.contourArea(contour) < 50:
            continue
            
        points = contour.reshape(-1, 2)
        path_data = 'M {} {} '.format(points[0][0], points[0][1])
        path_data += ' '.join('L {} {}'.format(x, y) for x, y in points[1:])
        path_data += ' Z'
        
        # 一般的な輪郭は細い線で描画
        dwg.add(dwg.path(d=path_data,
                        stroke='gray', stroke_width=0.2,
                        fill='none'))
    
    # SVGファイルの保存
    dwg.save()

    return preprocessed, shapes


def visualize_detected_shapes(preprocessed_results, shapes, save_path=None):
    """
    検出された図形要素を可視化します
    
    Parameters:
    -----------
    preprocessed_results : dict
        前処理結果
    shapes : dict
        検出された図形要素
    save_path : str, optional
        結果を保存するパス
    """
    # 元画像のコピーを作成
    image = preprocessed_results['original'].copy()
    height, width = image.shape[:2]
    
    # 結果表示用のサブプロット作成
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    fig.suptitle('Detected Shapes in Floor Plan', fontsize=16)
    
    # 1. 線分の可視化
    ax_lines = axes[0, 0]
    ax_lines.imshow(image)
    if shapes['lines'] is not None:
        for line in shapes['lines']:
            x1, y1, x2, y2 = line
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            if length > 100:  # 壁らしい線分
                ax_lines.plot([x1, x2], [y1, y2], 'r-', linewidth=2, alpha=0.7)
            else:  # その他の線分
                ax_lines.plot([x1, x2], [y1, y2], 'b-', linewidth=1, alpha=0.5)
    ax_lines.set_title(f'Detected Lines: {len(shapes["lines"]) if shapes["lines"] is not None else 0}')
    ax_lines.axis('off')
    
    # 2. 円の可視化
    ax_circles = axes[0, 1]
    ax_circles.imshow(image)
    if shapes['circles'] is not None:
        circles = np.uint16(np.around(shapes['circles']))
        for circle in circles[0, :]:
            center = (circle[0], circle[1])
            radius = circle[2]
            circle_patch = plt.Circle(center, radius, fill=False, color='g', linewidth=2)
            ax_circles.add_patch(circle_patch)
    ax_circles.set_title(f'Detected Circles: {len(shapes["circles"][0]) if shapes["circles"] is not None else 0}')
    ax_circles.axis('off')
    
    # 3. 矩形の可視化
    ax_rects = axes[1, 0]
    ax_rects.imshow(image)
    for rect in shapes['rectangles']:
        rect_points = rect.reshape(-1, 2)
        width = np.linalg.norm(rect_points[0] - rect_points[1])
        height = np.linalg.norm(rect_points[1] - rect_points[2])
        ratio = min(width, height) / max(width, height)
        
        # 比率に基づいて色を変更
        if 0.1 < ratio < 0.5:  # ドアや窓らしい
            color = 'y'
        else:
            color = 'm'
        
        ax_rects.plot(rect_points[[0,1,2,3,0], 0], 
                     rect_points[[0,1,2,3,0], 1], 
                     color=color, linewidth=2, alpha=0.7)
    ax_rects.set_title(f'Detected Rectangles: {len(shapes["rectangles"])}')
    ax_rects.axis('off')
    
    # 4. すべての輪郭の可視化
    ax_contours = axes[1, 1]
    ax_contours.imshow(image)
    for contour in shapes['contours']:
        if cv2.contourArea(contour) < 50:  # 小さすぎる輪郭は除外
            continue
        contour = contour.reshape(-1, 2)
        ax_contours.plot(contour[:, 0], contour[:, 1], 'c-', linewidth=1, alpha=0.5)
    ax_contours.set_title(f'Detected Contours: {len(shapes["contours"])}')
    ax_contours.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    plt.show()


import cv2
import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import List, Set, Tuple, Optional


class IntersectionType(Enum):
    NONE = 0          # 交差なし
    L_SHAPE = 1       # L字型（端点同士が近い）
    T_SHAPE = 2       # T字型（端点が線分上にある）
    CROSS = 3         # 完全な交差
    COLLINEAR = 4     # 同一直線上

@dataclass
class Point:
    x: float
    y: float

    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)

@dataclass
class Line:
    start: Point
    end: Point

    def length(self) -> float:
        return np.sqrt((self.end.x - self.start.x) ** 2 + 
                      (self.end.y - self.start.y) ** 2)
    
    def direction_vector(self) -> Tuple[float, float]:
        return (self.end.x - self.start.x, self.end.y - self.start.y)

@dataclass
class Intersection:
    type: IntersectionType
    point: Point
    line1_idx: int
    line2_idx: int
    distance: float = 0.0

    def __str__(self):
        return f'{self.type.name} at ({self.point.x}, {self.point.y} (line1={self.line1_idx}, line2={self.line2_idx}, dist={self.distance:.2f})'

class LineIntersectionDetector:
    def __init__(self, endpoint_threshold=5.0, line_threshold=5.0, angle_threshold=9.0):
        """
        Parameters:
        endpoint_threshold (float): 端点同士の最大許容距離（L字型判定用）
        line_threshold (float): 点と線分の最大許容距離（T字型判定用）
        angle_threshold (float): 180度からの許容角度差（度単位）
                               この値より小さい角度差の場合、同一直線とみなす
        """
        self.endpoint_threshold = endpoint_threshold
        self.line_threshold = line_threshold
        self.angle_threshold = angle_threshold

    def detect_lines(self, image_path: str, min_line_length=20, max_line_gap=5) -> List[Line]:
        """
        画像から線分を検出する
        """
        # 既存のdetect_lines関数の処理を行う
        processed = preprocess_floorplan(image_path, visualize=False)
        lines, _ = detect_lines(processed, min_line_length, max_line_gap)

        # 検出された線分をLine型に変換
        detected_lines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                detected_lines.append(Line(
                    start=Point(float(x1), float(y1)),
                    end=Point(float(x2), float(y2))
                ))

        return detected_lines

    def are_lines_collinear(self, line1: Line, line2: Line, connection_point: Point) -> bool:
        """
        2つの線分が同一直線上にあるかどうかを判定
        connection_point: 2つの線分の接続点（L字型やT字型の交点）
        """
        # 各線分のベクトルを計算
        vec1 = (line1.end.x - line1.start.x, line1.end.y - line1.start.y)
        vec2 = (line2.end.x - line2.start.x, line2.end.y - line2.start.y)

        # connection_pointに近い方の端点を基準にベクトルの向きを調整
        if self.distance_point_to_point(line1.end, connection_point) < self.distance_point_to_point(line1.start, connection_point):
            vec1 = (-vec1[0], -vec1[1])
        if self.distance_point_to_point(line2.start, connection_point) < self.distance_point_to_point(line2.end, connection_point):
            vec2 = (-vec2[0], -vec2[1])

        # 角度を計算
        angle = self.calculate_angle(vec1, vec2)

        # 180度に近いかどうかを判定
        return abs(180 - angle) < self.angle_threshold

    def calculate_angle(self, vector1: Tuple[float, float], vector2: Tuple[float, float]) -> float:
        """
        2つのベクトル間の角度を計算（度単位）
        戻り値: 0-180度の範囲の角度
        """
        # ベクトルの長さを計算
        len1 = np.sqrt(vector1[0]**2 + vector1[1]**2)
        len2 = np.sqrt(vector2[0]**2 + vector2[1]**2)
        
        # ゼロベクトルチェック
        if len1 == 0 or len2 == 0:
            return 0.0
        
        # 内積を計算
        dot_product = vector1[0]*vector2[0] + vector1[1]*vector2[1]
        
        # cosθを計算（-1から1の範囲に収める）
        cos_theta = np.clip(dot_product / (len1 * len2), -1.0, 1.0)
        
        # 角度を度単位で返す
        return np.degrees(np.arccos(cos_theta))

    def distance_point_to_point(self, p1: Point, p2: Point) -> float:
        """2点間の距離を計算"""
        return np.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2)

    def distance_point_to_line(self, point: Point, line: Line) -> Tuple[float, Point]:
        """点と線分との最短距離と、線分上の最近接点を計算"""
        # 線分の方向ベクトル
        line_vec = Point(line.end.x - line.start.x, line.end.y - line.start.y)
        # 点から線分の始点へのベクトル
        point_vec = Point(point.x - line.start.x, point.y - line.start.y)

        # 線分の長さの2乗
        line_length_sq = line_vec.x ** 2 + line_vec.y ** 2

        # 内積を計算
        dot_product = (point_vec.x * line_vec.x + point_vec.y * line_vec.y)

        # パラメータtを計算（線分上の最近接点の位置）
        t = max(0, min(1, dot_product / line_length_sq))

        # 線分上の最近接点
        nearest = Point(
            line.start.x + t * line_vec.x,
            line.start.y + t * line_vec.y
        )

        # 距離を計算
        distance = self.distance_point_to_point(point, nearest)

        return distance, nearest

    def update_endpoints_to_intersection_points(self, lines: List[Line], intersections: List[Intersection]) -> List[Line]:
        """
        L字型およびT字型の交点に基づいて線分の端点を更新
        複数の線分が同じ角を共有する場合も考慮
        """
        # 端点の位置をキーとして、関連する線分のインデックスと端点の種類（start/end）を格納
        endpoint_connections = defaultdict(list)
        
        # 各線分の端点情報を収集
        for i, line in enumerate(lines):
            # タプルをキーとして使用するため、Point座標をタプルに変換
            start_key = (line.start.x, line.start.y)
            end_key = (line.end.x, line.end.y)
            endpoint_connections[start_key].append((i, 'start'))
            endpoint_connections[end_key].append((i, 'end'))

        # 更新された線分のリスト（元のリストのコピー）
        updated_lines = [Line(start=Point(line.start.x, line.start.y),
                            end=Point(line.end.x, line.end.y)) 
                        for line in lines]

        # L字型の交点に基づいて端点を更新
        for intersection in intersections:
            if intersection.type != IntersectionType.L_SHAPE:
                continue

            # 交点の座標をキーとして使用
            intersection_key = (intersection.point.x, intersection.point.y)
            line1 = lines[intersection.line1_idx]
            line2 = lines[intersection.line2_idx]

            # 最も近い端点を特定
            endpoints1 = [(line1.start, 'start'), (line1.end, 'end')]
            endpoints2 = [(line2.start, 'start'), (line2.end, 'end')]

            # 各線分について、交点に最も近い端点を見つける
            for line_idx, line in [(intersection.line1_idx, line1), (intersection.line2_idx, line2)]:
                min_dist = float('inf')
                closest_endpoint_type = None
                
                for endpoint, endpoint_type in [(line.start, 'start'), (line.end, 'end')]:
                    dist = self.distance_point_to_point(endpoint, intersection.point)
                    if dist < min_dist:
                        min_dist = dist
                        closest_endpoint_type = endpoint_type

                # 該当する端点を更新
                if closest_endpoint_type == 'start':
                    updated_lines[line_idx].start = intersection.point
                else:
                    updated_lines[line_idx].end = intersection.point

                # この端点に接続している他の線分も更新
                original_endpoint_key = (
                    line.start.x if closest_endpoint_type == 'start' else line.end.x,
                    line.start.y if closest_endpoint_type == 'start' else line.end.y
                )
                
                # 同じ端点を共有する他の線分も更新
                for connected_line_idx, connected_endpoint_type in endpoint_connections[original_endpoint_key]:
                    if connected_line_idx != line_idx:  # 自分自身は除外
                        if connected_endpoint_type == 'start':
                            updated_lines[connected_line_idx].start = intersection.point
                        else:
                            updated_lines[connected_line_idx].end = intersection.point

        return updated_lines

    def find_intersection(self, line1: Line, line2: Line) -> Optional[Intersection]:
        """2つの線分の交差関係を判定"""
        # まず端点同士の距離をチェック（L字型の判定）
        endpoints1 = [line1.start, line1.end]
        endpoints2 = [line2.start, line2.end]

        for p1 in endpoints1:
            for p2 in endpoints2:
                dist = self.distance_point_to_point(p1, p2)
                if dist <= self.endpoint_threshold:
                    # 接続点を計算
                    connection_point = Point((p1.x + p2.x)/2, (p1.y + p2.y)/2)

                    # 同一直線上かどうかをチェック
                    if self.are_lines_collinear(line1, line2, connection_point):
                         return Intersection(
                            type=IntersectionType.COLLINEAR,
                            point=connection_point,
                            line1_idx=-1,
                            line2_idx=-1,
                            distance=dist
                        )

                    return Intersection(
                        type=IntersectionType.L_SHAPE,
                        point=Point((p1.x + p2.x)/2, (p1.y + p2.y)/2),
                        line1_idx=-1,  # インデックスは呼び出し側で設定
                        line2_idx=-1,
                        distance=dist
                    )

        # T字型の判定（端点が他の線分上にあるか）
        for p1 in endpoints1:
            dist, nearest = self.distance_point_to_line(p1, line2)
            if dist <= self.line_threshold:
                # 同一直線上かどうかをチェック
                if self.are_lines_collinear(line1, line2, p1):
                     return Intersection(
                            type=IntersectionType.COLLINEAR,
                            point=p1,
                            line1_idx=-1,
                            line2_idx=-1,
                            distance=dist
                        )

                return Intersection(
                    type=IntersectionType.T_SHAPE,
                    point=nearest,
                    line1_idx=-1,
                    line2_idx=-1,
                    distance=dist
                )

        for p2 in endpoints2:
            dist, nearest = self.distance_point_to_line(p2, line1)
            if dist <= self.line_threshold:
                # 同一直線上かどうかをチェック
                if self.are_lines_collinear(line1, line2, p2):
                    return Intersection(
                        type=IntersectionType.COLLINEAR,
                        point=p2,
                        line1_idx=-1,
                        line2_idx=-1,
                        distance=dist
                    )

                return Intersection(
                    type=IntersectionType.T_SHAPE,
                    point=nearest,
                    line1_idx=-1,
                    line2_idx=-1,
                    distance=dist
                )

        # 完全な交差の判定
        # 線分の方向ベクトル
        d1 = Point(line1.end.x - line1.start.x, line1.end.y - line1.start.y)
        d2 = Point(line2.end.x - line2.start.x, line2.end.y - line2.start.y)

        # 交差判定の行列式
        det = d1.x * d2.y - d1.y * d2.x

        if abs(det) > 1e-10:  # 平行でない場合
            t = ((line2.start.x - line1.start.x) * d2.y - 
                 (line2.start.y - line1.start.y) * d2.x) / det
            u = ((line2.start.x - line1.start.x) * d1.y - 
                 (line2.start.y - line1.start.y) * d1.x) / det

            if 0 <= t <= 1 and 0 <= u <= 1:
                intersection_point = Point(
                    line1.start.x + t * d1.x,
                    line1.start.y + t * d1.y
                )
                return Intersection(
                    type=IntersectionType.CROSS,
                    point=intersection_point,
                    line1_idx=-1,
                    line2_idx=-1,
                    distance=0.0
                )

        return None

    def analyze_intersections(self, lines: List[Line]) -> List[Intersection]:
        """すべての線分の交差関係を分析"""
        intersections = []

        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                intersection = self.find_intersection(lines[i], lines[j])
                if intersection is not None:
                    intersection.line1_idx = i
                    intersection.line2_idx = j
                    intersections.append(intersection)

        return intersections

    def merge_collinear_lines(self, lines: List[Line], intersections: List[Intersection]) -> Tuple[List[Line], List[Intersection]]:
        """同一直線上の線分を結合して新しい線分リストを生成"""
        # 結合済みの線分のインデックスを記録
        merged_indices = set()
        new_lines = []
        new_intersections = []
        
        # 古い線分から新しい線分へのマッピング
        old_to_new_map = {}
        
        # COLLINEARの交点を処理
        for intersection in intersections:
            if intersection.type == IntersectionType.COLLINEAR:
                if intersection.line1_idx in merged_indices or intersection.line2_idx in merged_indices:
                    continue
                
                line1 = lines[intersection.line1_idx]
                line2 = lines[intersection.line2_idx]
                
                # 新しい線分の端点を決定（最も遠い端点同士を使用）
                points = [line1.start, line1.end, line2.start, line2.end]
                max_dist = 0
                new_start = new_end = points[0]
                
                for i in range(len(points)):
                    for j in range(i + 1, len(points)):
                        dist = self.distance_point_to_point(points[i], points[j])
                        if dist > max_dist:
                            max_dist = dist
                            new_start = points[i]
                            new_end = points[j]
                
                new_line = Line(new_start, new_end)
                new_line_idx = len(new_lines)
                new_lines.append(new_line)
                
                # マッピングを更新
                old_to_new_map[intersection.line1_idx] = new_line_idx
                old_to_new_map[intersection.line2_idx] = new_line_idx
                merged_indices.add(intersection.line1_idx)
                merged_indices.add(intersection.line2_idx)
        
        # 結合されなかった線分を追加
        for i, line in enumerate(lines):
            if i not in merged_indices:
                new_line_idx = len(new_lines)
                new_lines.append(line)
                old_to_new_map[i] = new_line_idx
        
        # 交点情報を更新
        for intersection in intersections:
            if intersection.type != IntersectionType.COLLINEAR:
                # 線分のインデックスを新しいものに更新
                new_intersection = Intersection(
                    type=intersection.type,
                    point=intersection.point,
                    line1_idx=old_to_new_map[intersection.line1_idx],
                    line2_idx=old_to_new_map[intersection.line2_idx],
                    distance=intersection.distance
                )
                new_intersections.append(new_intersection)
        
        return new_lines, new_intersections

    def find_and_remove_isolated_lines(self, lines: List[Line], intersections: List[Intersection]) -> Tuple[List[Line], List[Intersection]]:
        """交点を持たない孤立した線分を特定し削除"""
        # 各線分の接続関係をグラフとして構築
        connected_lines = defaultdict(set)
        for intersection in intersections:
            connected_lines[intersection.line1_idx].add(intersection.line2_idx)
            connected_lines[intersection.line2_idx].add(intersection.line1_idx)
        
        # 接続のある線分のインデックスを特定
        connected_indices = set()
        
        def dfs(node: int, visited: Set[int]):
            visited.add(node)
            for neighbor in connected_lines[node]:
                if neighbor not in visited:
                    dfs(neighbor, visited)
        
        # 連結成分を見つける
        components = []
        visited = set()
        for i in range(len(lines)):
            if i not in visited and i in connected_lines:
                component = set()
                dfs(i, component)
                components.append(component)
                visited.update(component)
        
        # 最大の連結成分を見つける
        max_component = max(components, key=len, default=set())
        
        # 新しい線分リストと交点リストを作成
        new_lines = []
        old_to_new_map = {}
        
        # 接続のある線分のみを保持
        for i, line in enumerate(lines):
            if i in max_component:
                new_idx = len(new_lines)
                new_lines.append(line)
                old_to_new_map[i] = new_idx
        
        # 交点情報を更新
        new_intersections = []
        for intersection in intersections:
            if (intersection.line1_idx in max_component and 
                intersection.line2_idx in max_component):
                new_intersection = Intersection(
                    type=intersection.type,
                    point=intersection.point,
                    line1_idx=old_to_new_map[intersection.line1_idx],
                    line2_idx=old_to_new_map[intersection.line2_idx],
                    distance=intersection.distance
                )
                new_intersections.append(new_intersection)
        
        return new_lines, new_intersections

    def find_and_update_intersections(self, lines: List[Line]) -> Tuple[List[Line], List[Intersection]]:
        """
        交点を検出し、線分の端点を更新する
        """
        # まず交点を検出
        intersections = self.analyze_intersections(lines)
        
        # 線分の端点を更新
        updated_lines = self.update_endpoints_to_intersection_points(lines, intersections)
        
        # 更新された線分で再度交点を検出
        updated_intersections = self.analyze_intersections(updated_lines)
        
        return updated_lines, updated_intersections

    def visualize_intersections(self, image_path: str, lines: List[Line], 
                              intersections: List[Intersection], scale=1.0):
        """交差点の可視化"""
        image = cv2.imread(image_path)
        if scale != 1.0:
            image = cv2.resize(image, None, fx=scale, fy=scale)

        # 線分の描画
        for line in lines:
            start = (int(line.start.x * scale), int(line.start.y * scale))
            end = (int(line.end.x * scale), int(line.end.y * scale))
            cv2.line(image, start, end, (0, 255, 0), 1)

        # 交差点の描画
        colors = {
            IntersectionType.L_SHAPE: (255, 0, 0),    # 青
            IntersectionType.T_SHAPE: (0, 0, 255),    # 赤
            IntersectionType.CROSS: (0, 255, 255)     # 黄
        }

        for intersection in intersections:
            point = (int(intersection.point.x * scale), 
                    int(intersection.point.y * scale))
            color = colors.get(intersection.type)
            if color:
                cv2.circle(image, point, 2, color, -1)

        return image


class SpatialIndex:
    def __init__(self, cell_size: int, image_width: int, image_height: int):
        self.cell_size = cell_size
        self.width = image_width
        self.height = image_height
        self.grid = defaultdict(list)
    
    def get_cell_coords(self, point: Point) -> Tuple[int, int]:
        return (int(point.x // self.cell_size), 
                int(point.y // self.cell_size))
    
    def add_intersection(self, intersection: Intersection, idx: int):
        cell_x, cell_y = self.get_cell_coords(intersection.point)
        self.grid[(cell_x, cell_y)].append((intersection, idx))
    
    def get_intersections_in_window(self, 
                                  window: Tuple[int, int, int, int]) -> List[Tuple[Intersection, int]]:
        x1, y1, x2, y2 = window
        start_cell_x = int(x1 // self.cell_size)
        start_cell_y = int(y1 // self.cell_size)
        end_cell_x = int(x2 // self.cell_size) + 1
        end_cell_y = int(y2 // self.cell_size) + 1
        
        result = []
        for cell_x in range(start_cell_x, end_cell_x):
            for cell_y in range(start_cell_y, end_cell_y):
                result.extend(self.grid[(cell_x, cell_y)])
        
        return result


class WindowOptimizer:
    def __init__(self, 
                 window_size: int = 20,      # ウィンドウのサイズ（ピクセル）
                 stride: int = 2,           # ウィンドウのスライド幅
                 density_threshold: float = 0.5,  # 交点密度の閾値
                 grid_size: int = 2,       # 格子点の間隔
                 cell_size: int = 50):     # 空間インデックスのセルサイズ
        self.window_size = window_size
        self.stride = stride
        self.density_threshold = density_threshold
        self.grid_size = grid_size
        self.cell_size = cell_size

    def get_grid_points_count(self, window: Tuple[int, int, int, int]) -> int:
        """ウィンドウ内の格子点の数を計算"""
        x1, y1, x2, y2 = window
        width = x2 - x1
        height = y2 - y1
        
        # 格子点の数を計算（端の余白を考慮）
        cols = (width - self.grid_size) // self.grid_size
        rows = (height - self.grid_size) // self.grid_size
        
        return cols * rows

    def is_point_in_window(self, point: Point, window: Tuple[int, int, int, int]) -> bool:
        """点がウィンドウ内にあるかどうかを判定"""
        x1, y1, x2, y2 = window
        return x1 <= point.x <= x2 and y1 <= point.y <= y2

    def calculate_intersection_density(self, 
                                    window: Tuple[int, int, int, int],
                                    intersections: List[Intersection]) -> float:
        """
        ウィンドウ内のTまたはL字型交点の密度を計算
        Returns:
            float: 格子点に対する交点の割合
        """
        grid_points_count = self.get_grid_points_count(window)
        if grid_points_count == 0:
            return 0.0
        
        # TまたはL字型の交点をカウント
        intersection_count = sum(
            1 for intersection in intersections
            if (intersection.type in [IntersectionType.T_SHAPE, IntersectionType.L_SHAPE] and
                self.is_point_in_window(intersection.point, window))
        )
        
        return intersection_count / grid_points_count

    def optimize_intersections(self, 
                             image_shape: Tuple[int, int],
                             lines: List[Line],
                             intersections: List[Intersection]) -> Tuple[List[Line], List[Intersection]]:
        """
        ウィンドウベースで交点と線分を最適化
        密度の高い領域の交点と関連する線分を削除
        """
        height, width = image_shape
        intersections_to_remove = set()
        lines_to_remove = set()
        
        # 空間インデックスの構築
        spatial_index = SpatialIndex(self.cell_size, width, height)
        for i, intersection in enumerate(intersections):
            if intersection.type in [IntersectionType.T_SHAPE, IntersectionType.L_SHAPE]:
                spatial_index.add_intersection(intersection, i)

        # グリッドサイズに基づく格子点数の事前計算
        points_per_window = ((self.window_size - self.grid_size) // self.grid_size) ** 2
        
        # ウィンドウをスライド
        for y in range(0, height - self.window_size + 1, self.stride):
            for x in range(0, width - self.window_size + 1, self.stride):
                window = (x, y, x + self.window_size, y + self.window_size)
                
                # ウィンドウ内の交点を効率的に取得
                window_intersections = spatial_index.get_intersections_in_window(window)
                intersection_count = len([
                    i for i, _ in window_intersections
                    if self.is_point_in_window(i.point, window)
                ])
                
                density = intersection_count / points_per_window if points_per_window > 0 else 0
                
                if density >= self.density_threshold:
                    for intersection, idx in window_intersections:
                        intersections_to_remove.add(idx)
                        lines_to_remove.add(intersection.line1_idx)
                        lines_to_remove.add(intersection.line2_idx)
        
        # 新しい線分リストと交点リストを作成
        new_lines = []
        old_to_new_map = {}
        
        for i, line in enumerate(lines):
            if i not in lines_to_remove:
                new_idx = len(new_lines)
                new_lines.append(line)
                old_to_new_map[i] = new_idx
        
        new_intersections = []
        for i, intersection in enumerate(intersections):
            if i not in intersections_to_remove:
                if (intersection.line1_idx in old_to_new_map and 
                    intersection.line2_idx in old_to_new_map):
                    new_intersection = Intersection(
                        type=intersection.type,
                        point=intersection.point,
                        line1_idx=old_to_new_map[intersection.line1_idx],
                        line2_idx=old_to_new_map[intersection.line2_idx],
                        distance=intersection.distance
                    )
                    new_intersections.append(new_intersection)
        
        return new_lines, new_intersections


def filter_overlapping_lines(model_lines: np.ndarray, 
                           model_scores: np.ndarray,
                           detected_lines: List[Line],
                           threshold: float = 0.01,
                           tolerance: float = 1e9) -> List[Line]:
    """
    モデルで検出された線分とオーバーラップする線分を削除
    モデルの線分を必ず優先する
    
    Args:
        model_lines: モデルが出力した線分 (N, 2, 2) shape
        model_scores: モデルが出力したスコア (N,) shape
        detected_lines: OpenCVで検出した線分のリスト
        threshold: オーバーラップ判定の閾値
        tolerance: 許容誤差
    """

    from demo import postprocess as lcnn_postprocess

    # Line型からnumpy配列に変換
    detected_array = np.array([
        [[line.start.x, line.start.y], 
         [line.end.x, line.end.y]] 
        for line in detected_lines
    ])

    # 検出線分のスコアを0.5に設定（モデル線分のスコアより必ず低くする）
    detected_scores = np.full(len(detected_lines), 0.5)

    # モデル線分のスコアを2に設定（必ず優先されるようにする）
    # model_scores_high = np.full_like(model_scores, 2.0)
    model_scores_high = model_scores

    # 全ての線分とスコアを結合
    all_lines = np.concatenate([model_lines, detected_array])
    all_scores = np.concatenate([model_scores_high, detected_scores])

    # postprocessで重複除去
    _, processed_scores = lcnn_postprocess(
        all_lines, 
        all_scores,
        threshold=threshold,
        tol=tolerance,
        do_clip=False
    )

    # スコアが0.5のものが残った検出線分（オーバーラップしていないもの）
    surviving_indices = np.where(np.isclose(processed_scores, 0.5))[0]
    surviving_indices = surviving_indices - len(model_lines)  # インデックスを調整

    # 残った線分を元のLine型で返す
    filtered_lines = [
        detected_lines[idx]
        for idx in surviving_indices
        if 0 <= idx < len(detected_lines)
    ]

    return filtered_lines


def filter_overlapping_lines_v2(model_lines: np.ndarray,
                                    detected_lines: List[Line],
                                    angle_threshold: float = 5.0,  # 度単位
                                    overlap_threshold: float = 0.3,  # 重なり率の閾値
                                    distance_threshold: float = 2.0  # 最短距離の閾値
                                    ) -> List[Line]:
    """
    角度差と重なり具合を考慮して、オーバーラップする線分を削除
    
    Args:
        model_lines: モデルが出力した線分 (N, 2, 2) shape [[y1,x1], [y2,x2]]
        detected_lines: OpenCVで検出した線分のリスト
        angle_threshold: 許容する角度差（度単位）
        overlap_threshold: 重なり具合の閾値（0-1）
        distance_threshold: 線分間の許容距離
    """
    def calculate_angle(line):
        """線分の角度を計算（度単位）"""
        dy = line[1][0] - line[0][0]  # y2 - y1
        dx = line[1][1] - line[0][1]  # x2 - x1
        angle = np.degrees(np.arctan2(dy, dx))
        return angle % 180  # 0-180度の範囲に正規化

    def calculate_overlap(line1, line2):
        """
        2つの線分の重なり具合を計算
        戻り値: 重なり率（0-1）と最短距離
        """
        # 線分を方向ベクトルで表現
        v1 = np.array([line1[1][1] - line1[0][1], line1[1][0] - line1[0][0]])
        v2 = np.array([line2[1][1] - line2[0][1], line2[1][0] - line2[0][0]])
        
        # 線分の長さ
        l1 = np.linalg.norm(v1)
        l2 = np.linalg.norm(v2)
        
        # 単位ベクトル化
        v1_unit = v1 / l1 if l1 > 0 else v1
        v2_unit = v2 / l2 if l2 > 0 else v2
        
        # 線分1の端点を線分2に投影
        p1_start = np.array([line1[0][1], line1[0][0]])
        p1_end = np.array([line1[1][1], line1[1][0]])
        p2_start = np.array([line2[0][1], line2[0][0]])
        p2_end = np.array([line2[1][1], line2[0][0]])
        
        # 線分2の方向への投影
        proj_start = np.dot(p1_start - p2_start, v2_unit)
        proj_end = np.dot(p1_end - p2_start, v2_unit)
        
        # 重なり部分の長さを計算
        overlap_start = max(0, min(proj_start, proj_end))
        overlap_end = min(l2, max(proj_start, proj_end))
        overlap_length = max(0, overlap_end - overlap_start)
        
        # 重なり率を計算（両方の線分の長さで正規化）
        overlap_ratio = overlap_length / max(l1, l2)
        
        # 最短距離の計算
        # 端点間の最短距離を計算
        min_distance = float('inf')
        points1 = [p1_start, p1_end]
        points2 = [p2_start, p2_end]
        
        for p1 in points1:
            for p2 in points2:
                dist = np.linalg.norm(p1 - p2)
                min_distance = min(min_distance, dist)
        
        return overlap_ratio, min_distance

    # OpenCV検出の線分をnumpy配列に変換
    detected_array = np.array([
        [[line.start.y, line.start.x],  # y, x の順番に注意
         [line.end.y, line.end.x]]
        for line in detected_lines
    ])
    
    # フィルタリング後の線分を格納するリスト
    filtered_lines = []
    
    # 各検出線分について、モデル線分との重なりをチェック
    for i, detected_line in enumerate(detected_lines):
        should_keep = True
        detected_angle = calculate_angle(detected_array[i])
        
        for model_line in model_lines:
            # 角度差のチェック
            model_angle = calculate_angle(model_line)
            angle_diff = min(
                abs(detected_angle - model_angle),
                180 - abs(detected_angle - model_angle)
            )
            
            if angle_diff <= angle_threshold:
                # 角度が近い場合、重なり具合をチェック
                overlap_ratio, min_dist = calculate_overlap(detected_array[i], model_line)
                
                if (overlap_ratio >= overlap_threshold or 
                    min_dist <= distance_threshold):
                    should_keep = False
                    break
        
        if should_keep:
            filtered_lines.append(detected_line)
    
    return filtered_lines


def filter_overlapping_lines_by_endpoints(
    model_lines: np.ndarray,
    detected_lines: List[Line],
    endpoint_threshold: float = 5.0,
    angle_threshold: float = 5.0,  # 度単位
    verbose: bool = False
) -> List[Line]:
    """
    端点からの距離と角度に基づいて重複する線分を削除
    条件：
    1. 両端点が閾値未満、または
    2. どちらかの端点が閾値未満かつ角度差が閾値未満
    
    Args:
        model_lines: モデルが出力した線分 (N, 2, 2) shape [[y1,x1], [y2,x2]]
        detected_lines: OpenCVで検出した線分のリスト
        endpoint_threshold: 端点と線分の距離閾値
        angle_threshold: 許容する角度差（度単位）
        verbose: 進捗表示フラグ
    """
    def point_to_line_distance(point, line_start, line_end):
        """点から線分までの最短距離を計算"""
        line_vec = line_end - line_start
        point_vec = point - line_start
        
        line_length_sq = np.dot(line_vec, line_vec)
        
        if line_length_sq == 0:
            return np.linalg.norm(point_vec)
        
        t = np.dot(point_vec, line_vec) / line_length_sq
        
        if t < 0:
            return np.linalg.norm(point_vec)
        elif t > 1:
            return np.linalg.norm(point - line_end)
        else:
            projection = line_start + t * line_vec
            return np.linalg.norm(point - projection)

    def calculate_line_angle(line_start, line_end):
        """線分の角度を計算（度単位）"""
        dx = line_end[0] - line_start[0]
        dy = line_end[1] - line_start[1]
        angle = np.degrees(np.arctan2(dy, dx))
        return angle % 180  # 0-180度の範囲に正規化

    def calculate_angle_difference(angle1, angle2):
        """2つの角度の差を計算（0-90度の範囲）"""
        diff = abs(angle1 - angle2)
        return min(diff, 180 - diff)

    def calculate_endpoint_to_line_distances(model_line, detected_line):
        """検出線分の端点からモデル線分までの最短距離を計算"""
        model_start = np.array([model_line[0][1], model_line[0][0]])
        model_end = np.array([model_line[1][1], model_line[1][0]])
        detect_start = np.array([detected_line.start.x, detected_line.start.y])
        detect_end = np.array([detected_line.end.x, detected_line.end.y])
        
        start_dist = point_to_line_distance(detect_start, model_start, model_end)
        end_dist = point_to_line_distance(detect_end, model_start, model_end)
        
        return start_dist, end_dist

    remaining_lines = detected_lines.copy()
    if verbose:
        print(f"Initially: {len(remaining_lines)} detected lines")
    
    for i, model_line in enumerate(model_lines):
        initial_count = len(remaining_lines)
        j = len(remaining_lines) - 1

        # モデル線分の角度を計算
        model_start = np.array([model_line[0][1], model_line[0][0]])
        model_end = np.array([model_line[1][1], model_line[1][0]])
        model_angle = calculate_line_angle(model_start, model_end)
        
        while j >= 0:
            detected_line = remaining_lines[j]
            start_dist, end_dist = calculate_endpoint_to_line_distances(model_line, detected_line)
            
            # 検出線分の角度を計算
            detect_start = np.array([detected_line.start.x, detected_line.start.y])
            detect_end = np.array([detected_line.end.x, detected_line.end.y])
            detect_angle = calculate_line_angle(detect_start, detect_end)
            
            # 角度差を計算
            angle_diff = calculate_angle_difference(model_angle, detect_angle)
            
            # 削除条件：
            # 1. 両端点が閾値未満、または
            # 2. どちらかの端点が閾値未満かつ角度差が閾値未満
            if (start_dist < endpoint_threshold and end_dist < endpoint_threshold) or \
               ((start_dist < endpoint_threshold or end_dist < endpoint_threshold) and \
                angle_diff < angle_threshold):
                remaining_lines.pop(j)
            j -= 1
        
        if verbose and initial_count != len(remaining_lines):
            print(f"After model line {i}: removed {initial_count - len(remaining_lines)} lines, "
                  f"{len(remaining_lines)} remaining")
            
    return remaining_lines


def analyze_cad_drawing(image_path: str, 
                       endpoint_threshold=5.0, 
                       line_threshold=5.0,
                       min_line_length=20,
                       max_line_gap=5):
    """
    CAD図面の解析を行う主関数
    """

    image = cv2.imread(image_path)
    image_shape = image.shape[:2]

    # LCNNによる線分検出の結果を取得
    from demo import main as lcnn_main
    lcnn_results = lcnn_main([image_path])
    lcnn_result = lcnn_results[0]  # 最初の結果のみ使用

    # モデルの線分をLine型に変換（全て採用）
    model_lines_converted = [
        Line(
            start=Point(float(line[0][1]), float(line[0][0])),
            end=Point(float(line[1][1]), float(line[1][0]))
        )
        for line in lcnn_result['lines']
    ]
    print(f"Model lines: {len(model_lines_converted)} lines")

    # OpenCVによる線分検出
    detector = LineIntersectionDetector(endpoint_threshold, line_threshold)

    # 線分の検出
    lines = detector.detect_lines(image_path, min_line_length, max_line_gap)
    print(f"Detected {len(lines)} lines")

    # モデル線分とオーバーラップする線分を削除
    # lines = filter_overlapping_lines_v2(
    #     lcnn_result['lines'],
    #     lines,
    #     angle_threshold=5.0,
    #     overlap_threshold=0.3,
    #     distance_threshold=2.0
    # )

    # diag = (image_shape[0] ** 2 + image_shape[1] ** 2) ** 0.5
    # lines = filter_overlapping_lines(
    #     lcnn_result['lines'],
    #     lcnn_result['scores'],
    #     lines,
    #     threshold=diag * 0.01,
    #     tolerance=0.1,
    # )

    # lines = filter_overlapping_lines_by_endpoints(
    #     lcnn_result['lines'],
    #     lines,
    # )

    print(f"Filtered to: {len(lines)} lines")

    lines = model_lines_converted + lines
    print(f"Total initial lines: {len(lines)} lines")

    # 交差関係の分析
    # intersections = detector.analyze_intersections(lines)
    lines, intersections = detector.find_and_update_intersections(lines)
    print(f"Found {len(intersections)} intersections")

    # 同一直線上の線分を結合
    for _ in range(3):
        lines, intersections = detector.merge_collinear_lines(lines, intersections)
        print(f"Merged collinear lines: {len(lines)} lines")

    # # 孤立した線分を削除
    # lines, intersections = detector.find_and_remove_isolated_lines(lines, intersections)
    # print(f"Removed isolated lines: {len(lines)} lines")

    window_optimizer = WindowOptimizer()
    lines, intersections = window_optimizer.optimize_intersections(
        image_shape, lines, intersections)
    print(f"Optimized intersections: {len(lines)} lines")

    # 結果の可視化
    visualization = detector.visualize_intersections(image_path, lines, intersections)

    return lines, intersections, visualization


if __name__ == "__main__":
    convert_to_svg('path_to_your_image.jpg', 'output.svg')

    # 画像の処理とSVG変換
    preprocessed, shapes = convert_to_svg('path_to_your_image.jpg', 'output.svg')

    # 検出結果の可視化
    visualize_detected_shapes(preprocessed, shapes, 'detected_shapes.png')
