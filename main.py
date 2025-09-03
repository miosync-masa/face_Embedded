"""
Ida アパレルブランド顔マッチングシステム
128次元顔特徴量を使用したブランド適合度スコアリング
"""

import numpy as np
import pandas as pd
import face_recognition
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.spatial.distance import mahalanobis, cosine
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==================== データ構造定義 ====================

@dataclass
class BrandProfile:
    """ブランドプロファイル"""
    name: str
    embeddings: np.ndarray  # (N, 128)の顔特徴量
    mean_vector: np.ndarray  # 平均顔ベクトル
    cov_matrix: np.ndarray  # 共分散行列
    important_dims: List[int]  # 重要な特徴次元
    pca_model: Optional[PCA] = None
    
    def __post_init__(self):
        if self.mean_vector is None:
            self.mean_vector = np.mean(self.embeddings, axis=0)
        if self.cov_matrix is None:
            self.cov_matrix = np.cov(self.embeddings.T)

@dataclass
class MatchingResult:
    """マッチング結果"""
    brand: str
    score: float
    percentile: float  # そのブランド内での順位
    explanation: Dict[str, any]

# ==================== 顔特徴量抽出 ====================

class FaceEncoder:
    """顔画像から128次元特徴量を抽出"""
    
    def __init__(self, batch_size: int = 32):
        self.batch_size = batch_size
        
    def encode_single(self, image_path: str) -> Optional[np.ndarray]:
        """単一画像の特徴量抽出"""
        try:
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            
            if len(encodings) == 0:
                logger.warning(f"顔が検出できません: {image_path}")
                return None
            elif len(encodings) > 1:
                logger.info(f"複数の顔を検出、最初の顔を使用: {image_path}")
                
            return encodings[0]
            
        except Exception as e:
            logger.error(f"エラー: {image_path} - {e}")
            return None
    
    def encode_batch(self, image_paths: List[str], n_workers: int = 4) -> Dict[str, np.ndarray]:
        """バッチ処理で複数画像を特徴量化"""
        results = {}
        
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            future_to_path = {
                executor.submit(self.encode_single, path): path 
                for path in image_paths
            }
            
            for future in tqdm(as_completed(future_to_path), total=len(image_paths), desc="顔特徴量抽出"):
                path = future_to_path[future]
                try:
                    encoding = future.result()
                    if encoding is not None:
                        results[path] = encoding
                except Exception as e:
                    logger.error(f"処理エラー: {path} - {e}")
                    
        return results

# ==================== ブランドモデル学習 ====================

class BrandModelTrainer:
    """各ブランドの採用パターンを学習"""
    
    def __init__(self):
        self.brand_profiles: Dict[str, BrandProfile] = {}
        self.global_scaler = StandardScaler()
        
    def train_brand_model(self, brand_name: str, embeddings: np.ndarray) -> BrandProfile:
        """単一ブランドのモデル学習"""
        logger.info(f"{brand_name}のモデル学習開始 (サンプル数: {len(embeddings)})")
        
        # 基本統計量
        mean_vec = np.mean(embeddings, axis=0)
        cov_mat = np.cov(embeddings.T)
        
        # 重要次元の特定（分散が小さい = 一貫性がある）
        std_per_dim = np.std(embeddings, axis=0)
        important_dims = np.where(std_per_dim < np.percentile(std_per_dim, 30))[0].tolist()
        
        # PCA学習（ブランドの主要パターン抽出）
        pca = PCA(n_components=20)
        pca.fit(embeddings)
        
        profile = BrandProfile(
            name=brand_name,
            embeddings=embeddings,
            mean_vector=mean_vec,
            cov_matrix=cov_mat,
            important_dims=important_dims,
            pca_model=pca
        )
        
        self.brand_profiles[brand_name] = profile
        logger.info(f"{brand_name}の重要特徴次元: {important_dims[:10]}")
        
        return profile
    
    def train_all_brands(self, brand_data: Dict[str, np.ndarray]):
        """全ブランドのモデル学習"""
        all_embeddings = []
        
        for brand_name, embeddings in brand_data.items():
            self.train_brand_model(brand_name, embeddings)
            all_embeddings.extend(embeddings)
        
        # 全体での正規化パラメータ学習
        self.global_scaler.fit(all_embeddings)
        
    def save_models(self, save_path: str):
        """学習済みモデルの保存"""
        with open(save_path, 'wb') as f:
            pickle.dump({
                'brand_profiles': self.brand_profiles,
                'scaler': self.global_scaler
            }, f)
        logger.info(f"モデルを保存: {save_path}")
    
    def load_models(self, load_path: str):
        """学習済みモデルの読み込み"""
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
            self.brand_profiles = data['brand_profiles']
            self.global_scaler = data['scaler']
        logger.info(f"モデルを読み込み: {load_path}")

# ==================== スコアリング ====================

class BrandMatcher:
    """新規応募者とブランドのマッチング"""
    
    def __init__(self, trainer: BrandModelTrainer):
        self.trainer = trainer
        
    def calculate_mahalanobis_score(self, embedding: np.ndarray, profile: BrandProfile) -> float:
        """マハラノビス距離によるスコア（0-1）"""
        try:
            inv_cov = np.linalg.pinv(profile.cov_matrix)
            dist = mahalanobis(embedding, profile.mean_vector, inv_cov)
            # 距離を確率的スコアに変換
            score = np.exp(-dist**2 / 20)  # パラメータ調整可能
            return np.clip(score, 0, 1)
        except:
            # 共分散行列が特異な場合はコサイン類似度を使用
            return self.calculate_cosine_score(embedding, profile)
    
    def calculate_cosine_score(self, embedding: np.ndarray, profile: BrandProfile) -> float:
        """コサイン類似度スコア（0-1）"""
        similarity = 1 - cosine(embedding, profile.mean_vector)
        return (similarity + 1) / 2
    
    def calculate_weighted_score(self, embedding: np.ndarray, profile: BrandProfile) -> float:
        """重要次元を重視したスコア"""
        if not profile.important_dims:
            return self.calculate_cosine_score(embedding, profile)
        
        # 重要次元での距離
        important_diff = embedding[profile.important_dims] - profile.mean_vector[profile.important_dims]
        important_dist = np.linalg.norm(important_diff)
        
        # その他の次元での距離
        other_dims = [i for i in range(128) if i not in profile.important_dims]
        other_diff = embedding[other_dims] - profile.mean_vector[other_dims]
        other_dist = np.linalg.norm(other_diff)
        
        # 重み付き統合（重要次元を2倍重視）
        combined_dist = (important_dist * 2 + other_dist) / 3
        score = np.exp(-combined_dist / 10)
        
        return np.clip(score, 0, 1)
    
    def calculate_percentile(self, embedding: np.ndarray, profile: BrandProfile) -> float:
        """そのブランド内での順位（パーセンタイル）"""
        # 既存スタッフとの距離を計算
        distances = [np.linalg.norm(embedding - e) for e in profile.embeddings]
        percentile = (np.sum(np.array(distances) > np.median(distances)) / len(distances)) * 100
        return percentile
    
    def match_single_brand(self, embedding: np.ndarray, brand_name: str) -> MatchingResult:
        """単一ブランドとのマッチング"""
        profile = self.trainer.brand_profiles[brand_name]
        
        # 複数の手法でスコア計算
        mahal_score = self.calculate_mahalanobis_score(embedding, profile)
        cos_score = self.calculate_cosine_score(embedding, profile)
        weighted_score = self.calculate_weighted_score(embedding, profile)
        
        # 統合スコア（重み付き平均）
        final_score = (mahal_score * 0.4 + cos_score * 0.3 + weighted_score * 0.3)
        
        # パーセンタイル計算
        percentile = self.calculate_percentile(embedding, profile)
        
        # 説明情報
        explanation = {
            'mahalanobis_score': mahal_score,
            'cosine_score': cos_score,
            'weighted_score': weighted_score,
            'important_dims_match': self._explain_important_dims(embedding, profile),
            'closest_existing_distance': min([np.linalg.norm(embedding - e) for e in profile.embeddings[:100]])
        }
        
        return MatchingResult(
            brand=brand_name,
            score=final_score,
            percentile=percentile,
            explanation=explanation
        )
    
    def _explain_important_dims(self, embedding: np.ndarray, profile: BrandProfile) -> Dict:
        """重要次元での一致度を説明"""
        if not profile.important_dims:
            return {}
        
        explanations = {}
        for dim in profile.important_dims[:5]:  # TOP5の重要次元
            diff = abs(embedding[dim] - profile.mean_vector[dim])
            match_level = 'high' if diff < 0.5 else 'medium' if diff < 1.0 else 'low'
            explanations[f'dim_{dim}'] = {
                'difference': diff,
                'match_level': match_level
            }
        
        return explanations
    
    def match_all_brands(self, embedding: np.ndarray, top_k: int = 10) -> List[MatchingResult]:
        """全ブランドとマッチングしてTOP-Kを返す"""
        results = []
        
        for brand_name in self.trainer.brand_profiles.keys():
            result = self.match_single_brand(embedding, brand_name)
            results.append(result)
        
        # スコアでソート
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results[:top_k]

# ==================== 分析・可視化 ====================

class BrandAnalyzer:
    """ブランド間の関係性分析"""
    
    def __init__(self, trainer: BrandModelTrainer):
        self.trainer = trainer
    
    def calculate_brand_similarity_matrix(self) -> pd.DataFrame:
        """ブランド間の類似度マトリックス"""
        brands = list(self.trainer.brand_profiles.keys())
        similarity_matrix = np.zeros((len(brands), len(brands)))
        
        for i, brand1 in enumerate(brands):
            for j, brand2 in enumerate(brands):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    profile1 = self.trainer.brand_profiles[brand1]
                    profile2 = self.trainer.brand_profiles[brand2]
                    
                    # 中心ベクトルの類似度
                    cos_sim = 1 - cosine(profile1.mean_vector, profile2.mean_vector)
                    similarity_matrix[i, j] = (cos_sim + 1) / 2
        
        return pd.DataFrame(similarity_matrix, index=brands, columns=brands)
    
    def plot_brand_clusters(self, n_components: int = 2):
        """ブランドの2D/3D可視化"""
        from sklearn.manifold import TSNE
        
        all_means = []
        labels = []
        
        for brand_name, profile in self.trainer.brand_profiles.items():
            all_means.append(profile.mean_vector)
            labels.append(brand_name)
        
        # t-SNE で次元削減
        tsne = TSNE(n_components=n_components, random_state=42)
        reduced = tsne.fit_transform(all_means)
        
        # プロット
        plt.figure(figsize=(12, 8))
        
        if n_components == 2:
            plt.scatter(reduced[:, 0], reduced[:, 1], s=100)
            for i, label in enumerate(labels):
                plt.annotate(label, (reduced[i, 0], reduced[i, 1]))
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')
        
        plt.title('ブランド間の関係性マップ')
        plt.tight_layout()
        return plt
    
    def analyze_dimension_importance(self) -> pd.DataFrame:
        """全ブランドで重要な次元を分析"""
        dim_importance = np.zeros(128)
        
        for profile in self.trainer.brand_profiles.values():
            # 各ブランドでの標準偏差の逆数を重要度とする
            std_per_dim = np.std(profile.embeddings, axis=0)
            importance = 1 / (std_per_dim + 1e-6)
            dim_importance += importance / len(self.trainer.brand_profiles)
        
        # TOP20の重要次元
        top_dims = np.argsort(dim_importance)[-20:][::-1]
        
        return pd.DataFrame({
            'dimension': top_dims,
            'importance': dim_importance[top_dims]
        })

# ==================== メインパイプライン ====================

class IdaMatchingPipeline:
    """統合パイプライン"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.encoder = FaceEncoder()
        self.trainer = BrandModelTrainer()
        self.matcher = None
        self.analyzer = None
        
        if model_path and Path(model_path).exists():
            self.trainer.load_models(model_path)
            self._initialize_components()
    
    def _initialize_components(self):
        """コンポーネント初期化"""
        self.matcher = BrandMatcher(self.trainer)
        self.analyzer = BrandAnalyzer(self.trainer)
    
    def train_from_images(self, brand_image_paths: Dict[str, List[str]]):
        """画像データから学習"""
        brand_embeddings = {}
        
        for brand_name, image_paths in brand_image_paths.items():
            logger.info(f"{brand_name}の画像処理開始: {len(image_paths)}枚")
            
            # 特徴量抽出
            encodings_dict = self.encoder.encode_batch(image_paths)
            embeddings = np.array(list(encodings_dict.values()))
            
            if len(embeddings) > 0:
                brand_embeddings[brand_name] = embeddings
                logger.info(f"{brand_name}: {len(embeddings)}個の顔特徴量を抽出")
            else:
                logger.warning(f"{brand_name}: 顔特徴量を抽出できませんでした")
        
        # モデル学習
        self.trainer.train_all_brands(brand_embeddings)
        self._initialize_components()
        
        return self
    
    def predict(self, image_path: str, top_k: int = 10) -> Dict:
        """新規画像の予測"""
        # 特徴量抽出
        embedding = self.encoder.encode_single(image_path)
        
        if embedding is None:
            return {'error': '顔を検出できませんでした'}
        
        # マッチング
        results = self.matcher.match_all_brands(embedding, top_k)
        
        # 結果の整形
        output = {
            'top_matches': [],
            'recommendation': None
        }
        
        for i, result in enumerate(results):
            match_info = {
                'rank': i + 1,
                'brand': result.brand,
                'score': float(result.score),
                'percentile': float(result.percentile),
                'confidence': self._score_to_confidence(result.score),
                'explanation': result.explanation
            }
            output['top_matches'].append(match_info)
        
        # 最適なブランドの推薦
        if results and results[0].score > 0.7:
            output['recommendation'] = {
                'brand': results[0].brand,
                'confidence': 'high',
                'reason': 'このブランドの採用パターンと高い一致'
            }
        elif results and results[0].score > 0.5:
            output['recommendation'] = {
                'brand': results[0].brand,
                'confidence': 'medium',
                'reason': 'このブランドでの採用可能性あり'
            }
        else:
            output['recommendation'] = {
                'brand': None,
                'confidence': 'low',
                'reason': '明確な適合ブランドが見つかりません'
            }
        
        return output
    
    def _score_to_confidence(self, score: float) -> str:
        """スコアを信頼度レベルに変換"""
        if score > 0.8:
            return '⭐⭐⭐⭐⭐'
        elif score > 0.6:
            return '⭐⭐⭐⭐'
        elif score > 0.4:
            return '⭐⭐⭐'
        elif score > 0.2:
            return '⭐⭐'
        else:
            return '⭐'
    
    def save_model(self, path: str):
        """モデル保存"""
        self.trainer.save_models(path)
    
    def generate_report(self) -> Dict:
        """分析レポート生成"""
        report = {
            'total_brands': len(self.trainer.brand_profiles),
            'brand_statistics': {},
            'brand_similarity': self.analyzer.calculate_brand_similarity_matrix().to_dict(),
            'important_dimensions': self.analyzer.analyze_dimension_importance().to_dict()
        }
        
        for brand_name, profile in self.trainer.brand_profiles.items():
            report['brand_statistics'][brand_name] = {
                'sample_count': len(profile.embeddings),
                'important_dims': profile.important_dims[:10],
                'variance_explained': float(profile.pca_model.explained_variance_ratio_[:5].sum()) if profile.pca_model else 0
            }
        
        return report

# ==================== 実行サンプル ====================

def main():
    """実行サンプル"""
    
    # データ準備（実際のパスに置き換えてください）
    brand_image_paths = {
        'LOWRYS FARM': ['path/to/lowrys/*.jpg'],
        'earth music': ['path/to/earth/*.jpg'],
        'CHANEL': ['path/to/chanel/*.jpg'],
        'ZARA': ['path/to/zara/*.jpg'],
        # ... 他のブランド
    }
    
    # パイプライン初期化と学習
    pipeline = IdaMatchingPipeline()
    
    # 学習実行
    # pipeline.train_from_images(brand_image_paths)
    # pipeline.save_model('models/ida_brand_matcher.pkl')
    
    # または学習済みモデル読み込み
    # pipeline = IdaMatchingPipeline('models/ida_brand_matcher.pkl')
    
    # 予測
    # result = pipeline.predict('new_applicant.jpg', top_k=10)
    # print(json.dumps(result, indent=2, ensure_ascii=False))
    
    # レポート生成
    # report = pipeline.generate_report()
    # print(json.dumps(report, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
