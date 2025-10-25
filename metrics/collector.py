from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np
from loguru import logger
from psycopg2.extras import RealDictCursor
from .utils import safe_parse_json, extract_company_from_model_id, extract_architecture_from_tags_or_raw, ensure_list_from_json_field
from datetime import datetime, timedelta

@dataclass
class NumericMetrics:
    downloads_mean: Optional[float] = None
    downloads_median: Optional[float] = None
    downloads_variance: Optional[float] = None
    total_models: int = 0
    private_count: int = 0
    likes_top_n: List[Dict[str, Any]] = None
    pipeline_counts: Dict[str, int] = None
    license_counts: Dict[str, int] = None

    def to_dict(self):
        return asdict(self)

class MetricsCollector:
    def __init__(self, conn, logger_obj=logger):
        self.conn = conn
        self.logger = logger_obj

    def _df_from_query(self, query, params=None) -> pd.DataFrame:
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, params or {})
                rows = cur.fetchall()
            df = pd.DataFrame(rows)
            self.logger.debug(f"Получено строк: {len(df)}")
            return df
        except Exception as e:
            self.logger.exception("Ошибка при выполнении SQL: {}", e)
            return pd.DataFrame()

    def load_full_models_df(self, limit: Optional[int] = None) -> pd.DataFrame:
        q = "SELECT * FROM hf_models"
        if limit:
            q += f" LIMIT {int(limit)}"
        df = self._df_from_query(q)
        # нормализуем некоторые поля
        if 'tags' in df.columns:
            df['tags_list'] = df['tags'].apply(lambda x: ensure_list_from_json_field(x))
        else:
            df['tags_list'] = [[] for _ in range(len(df))]
        # Normalize booleans
        if 'private' in df.columns:
            df['private_bool'] = df['private'].apply(lambda x: str(x).lower() in ("true", "t", "1", "yes") if x is not None else False)
        else:
            df['private_bool'] = False
        # parse raw_data into dict
        if 'raw_data' in df.columns:
            df['raw_json'] = df['raw_data'].apply(lambda x: safe_parse_json(x) or {})
        else:
            df['raw_json'] = [{} for _ in range(len(df))]
        # createdAt normalization
        for col in ('createdat', 'createdAt', 'created_at'):
            if col in df.columns:
                df['created_at_ts'] = pd.to_datetime(df[col], errors='coerce')
                break
        if 'created_at_ts' not in df.columns:
            df['created_at_ts'] = pd.NaT
        # lastModified
        for col in ('lastmodified', 'lastModified', 'last_modified'):
            if col in df.columns:
                df['last_modified_ts'] = pd.to_datetime(df[col], errors='coerce')
                break
        if 'last_modified_ts' not in df.columns:
            df['last_modified_ts'] = pd.NaT

        # downloads and likes numeric
        df['downloads_num'] = pd.to_numeric(df.get('downloads', 0), errors='coerce').fillna(0).astype(int)
        df['likes_num'] = pd.to_numeric(df.get('likes', 0), errors='coerce').fillna(0).astype(int)

        # extract company and architecture
        df['company'] = df['model_id'].apply(lambda x: extract_company_from_model_id(x) if pd.notna(x) else None)
        df['architecture'] = df.apply(lambda r: extract_architecture_from_tags_or_raw(r.get('tags_list'), r.get('raw_json')), axis=1)

        return df

    def numeric_summary(self, df: pd.DataFrame, top_likes_n: int = 10) -> NumericMetrics:
        if df.empty:
            self.logger.warning("DataFrame пустой при вычислении numeric_summary")
            return NumericMetrics()

        downloads = df['downloads_num'].replace([np.inf, -np.inf], np.nan).dropna()
        mean = float(downloads.mean()) if not downloads.empty else 0.0
        median = float(downloads.median()) if not downloads.empty else 0.0
        variance = float(downloads.var(ddof=0)) if not downloads.empty else 0.0

        pipeline_counts = df['pipeline_tag'].fillna('unknown').value_counts().to_dict()
        license_counts = df['license'].fillna('unknown').value_counts().to_dict()

        private_count = int(df['private_bool'].sum())
        total_models = int(len(df))

        top_likes = df.sort_values('likes_num', ascending=False).head(top_likes_n)[['model_id', 'likes_num', 'downloads_num']].to_dict(orient='records')

        return NumericMetrics(
            downloads_mean=mean,
            downloads_median=median,
            downloads_variance=variance,
            total_models=total_models,
            private_count=private_count,
            likes_top_n=top_likes,
            pipeline_counts=pipeline_counts,
            license_counts=license_counts
        )

    def top_tags(self, df: pd.DataFrame, top_n: int = 30):
        # собираем все теги и считаем частоту
        all_tags = {}
        for tags in df.get('tags_list', []):
            if not tags:
                continue
            for t in tags:
                if not isinstance(t, str):
                    continue
                all_tags[t.lower()] = all_tags.get(t.lower(), 0) + 1
        # Возврат списка (tag, count)
        items = sorted(all_tags.items(), key=lambda x: x[1], reverse=True)
        return items[:top_n]

    def license_trends_by_year(self, df: pd.DataFrame, years: int = 5):
        if df.empty:
            return {}
        now = pd.Timestamp.utcnow()
        start_year = now.year - years + 1
        df = df.copy()
        df['year'] = df['created_at_ts'].dt.year
        df = df[df['year'].notna()]
        df['license_cat'] = df['license'].fillna('unknown')
        grouped = df[df['year'] >= start_year].groupby(['year', 'license_cat']).size().reset_index(name='count')
        # Pivot to {year: {license: count}}
        result = {}
        for year in sorted(grouped['year'].unique()):
            per_year = grouped[grouped['year'] == year][['license_cat', 'count']].set_index('license_cat')['count'].to_dict()
            result[int(year)] = per_year
        return result

    def pipeline_trend_last_n_years(self, df: pd.DataFrame, n_years: int = 5):
        if df.empty:
            return {}
        now = pd.Timestamp.utcnow()
        start_year = now.year - n_years + 1
        df['year'] = df['created_at_ts'].dt.year
        df = df[df['year'] >= start_year]
        grouped = df.groupby(['year', 'pipeline_tag']).size().reset_index(name='count')
        result = {}
        for year in sorted(grouped['year'].unique()):
            per_year = grouped[grouped['year'] == year][['pipeline_tag', 'count']].set_index('pipeline_tag')['count'].to_dict()
            result[int(year)] = per_year
        return result

    def cluster_by_architecture(self, df: pd.DataFrame, n_clusters: int = 8):
        """
        Попытка кластеризовать модели по 'architecture' и по embeddings признаков (если есть).
        Здесь мы используем простую стратегию:
        - если в колонке architecture много уникальных значений, берем распределение архитектур.
        - если данных архитектур недостаточно, кластеризуем по вектору признаков, собранному из тегов/полиенов.
        Для полной рантайм кластеризации нужны признаки модели (например, size, num_layers) — если их нет, результат зависит от parsed architecture/tags.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.cluster import KMeans

        arch_series = df['architecture'].fillna('unknown').astype(str)
        # если архитектуры достаточно информативны -> кластер по уникальным архитектурам (группировка)
        unique_archs = arch_series.unique().tolist()
        if len(unique_archs) >= n_clusters:
            # TF-IDF по строкам architecture
            vect = TfidfVectorizer(max_features=500)
            X = vect.fit_transform(arch_series.tolist())
            km = KMeans(n_clusters=min(n_clusters, X.shape[0]), random_state=42)
            labels = km.fit_predict(X)
            df['arch_cluster'] = labels
            clusters = {}
            for i in range(int(df['arch_cluster'].max()) + 1):
                members = df[df['arch_cluster'] == i]
                clusters[i] = {
                    "count": int(len(members)),
                    "top_architectures": members['architecture'].value_counts().head(10).to_dict(),
                    "sample_models": members['model_id'].head(10).tolist()
                }
            return clusters
        else:
            # мало архитектур — просто вернуть частоты
            counts = arch_series.value_counts().to_dict()
            return {"architecture_counts": counts}

    def top_companies(self, df: pd.DataFrame, top_n: int = 20):
        cnt = df['company'].fillna('unknown').value_counts().head(top_n)
        return cnt.to_dict()

    def time_series_models_by_domain(self, df: pd.DataFrame, domain_mapping: Dict[str, str] = None, years: int = 5):
        """
        domain_mapping: маппинг ключевых тегов/словарь -> domain name, например
         {'medical':'medicine', 'legal':'legal', 'code':'code-assistant'}
        Если mapping не задан, алгоритм попытается вывести домены по наличию ключевых слов в теге.
        Возвращает dict {year: {domain: count}}
        """
        if df.empty:
            return {}
        now = pd.Timestamp.utcnow()
        start_year = now.year - years + 1
        df['year'] = df['created_at_ts'].dt.year
        df = df[df['year'] >= start_year]
        # Build domain for each row
        def infer_domain(row):
            tags = row.get('tags_list', []) or []
            text = " ".join([str(t).lower() for t in tags])
            # simple heuristics
            if "medical" in text or "health" in text or "medicine" in text or "radiology" in text:
                return "medicine"
            if "code" in text or "programming" in text or "github" in text or "code-generation" in text:
                return "code-assistant"
            if "legal" in text or "law" in text or "legal-advisory" in text:
                return "legal"
            if "nlp" in text or "text-generation" in text or "question-answering" in text:
                return "nlp"
            if "vision" in text or "image" in text or "segmentation" in text or "object-detection" in text:
                return "vision"
            if "audio" in text or "speech" in text or "wav2vec" in text or "speech-to-text" in text:
                return "speech"
            # fallback: pipeline_tag
            pt = (row.get('pipeline_tag') or "").lower()
            if pt:
                return pt
            return "other"

        df['domain'] = df.apply(infer_domain, axis=1)
        grouped = df.groupby(['year', 'domain']).size().reset_index(name='count')
        result = {}
        for year in sorted(grouped['year'].unique()):
            result[int(year)] = grouped[grouped['year'] == year].set_index('domain')['count'].to_dict()
        return result

    def required_fields_check(self, df: pd.DataFrame):
        """
        Проверяет, какие поля присутствуют и какие требуются для полной аналитики.
        Возвращает список отсутствующих/частично присутствующих полей с рекомендациями.
        """
        needed = ['model_id','pipeline_tag','license','downloads','private','tags','likes','createdat','lastmodified','raw_data']
        present = {c.lower(): (c in df.columns) for c in needed}
        missing = [k for k,v in present.items() if not v]
        advice = {}
        if 'architecture' not in df.columns or df['architecture'].isna().all():
            advice['architecture'] = ("Рекомендуется иметь теги вида 'architecture:...' или в raw_data поля "
                                      "'config.architectures'/'model_type' для корректной кластеризации.")
        if 'company' not in df.columns:
            advice['company'] = ("Рекомендуется, чтобы model_id имел формат 'company/modelname' или в raw_data "
                                 "имелось поле 'author'/'owner' для корректного определения компаний.")
        return {"missing_columns": missing, "advice": advice}
