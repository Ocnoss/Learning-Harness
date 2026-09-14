"""第三层 headless A/B 评分 rubric + bundle-hash 缓存（契约 §6.1 评估侧）。

四个维度（相关性 / 事实性 / 教学适配 / 简洁性）加权聚合为 judge_score∈[0,1]。
缓存键由 bundle.to_json()（或任意上下文规范化串）+ rubric.signature() 确定性
派生——同一 (bundle, rubric) 永远命中同一分数，不重复调用裁判，满足确定性纪律。

裁判客户端可为 StubLLMClient（默认，离线 canned 回放）或真实 flagship（live
模式，由 test_layer3 门控决定）；本模块只依赖 client.complete_json 契约。
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# 四维 rubric：维度名（中英对照，用于裁判提示）与默认权重（和为 1.0）
DIMENSIONS: tuple[str, ...] = ("relevance", "factuality", "pedagogy", "conciseness")
DIMENSION_ZH = {
    "relevance": "相关性（上下文是否切中学习目标）",
    "factuality": "事实性（陈述是否可回溯到证据、无臆造）",
    "pedagogy": "教学适配（是否契合学习者当前水平与节奏）",
    "conciseness": "简洁性（是否无冗余、token 效率）",
}
DEFAULT_WEIGHTS: dict[str, float] = {
    "relevance": 0.35,
    "factuality": 0.30,
    "pedagogy": 0.20,
    "conciseness": 0.15,
}


class JudgeRubric:
    """确定性评分标尺：维度权重 + 聚合函数 + 裁判提示构造。"""

    def __init__(self, weights: dict[str, float] | None = None):
        w = dict(DEFAULT_WEIGHTS if weights is None else weights)
        missing = [d for d in DIMENSIONS if d not in w]
        if missing:
            raise ValueError(f"rubric 权重缺维度: {missing}")
        total = sum(float(w[d]) for d in DIMENSIONS)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"rubric 权重之和必须为 1.0，实际 {total}")
        self.weights = {d: float(w[d]) for d in DIMENSIONS}

    def signature(self) -> str:
        """rubric 的确定性指纹（键序固定），进缓存键。"""
        return json.dumps(
            {"dimensions": list(DIMENSIONS),
             "weights": {d: self.weights[d] for d in DIMENSIONS}},
            ensure_ascii=False, sort_keys=True,
        )

    def aggregate(self, dim_scores: dict) -> float:
        """按权重聚合各维度分（0..1）为总分；缺失维度按 0 计，越界裁剪到 [0,1]。"""
        total = 0.0
        for d in DIMENSIONS:
            raw = dim_scores.get(d, 0.0)
            try:
                v = float(raw)
            except (TypeError, ValueError):
                v = 0.0
            v = max(0.0, min(1.0, v))
            total += self.weights[d] * v
        return round(max(0.0, min(1.0, total)), 6)

    def judge_messages(self, goal: str, answer: str) -> list[dict]:
        """构造裁判提示：要求仅输出各维度 0..1 打分的 JSON。"""
        dims_desc = "\n".join(f"- {d}: {DIMENSION_ZH[d]}" for d in DIMENSIONS)
        schema = json.dumps({d: 0.0 for d in DIMENSIONS}, ensure_ascii=False)
        return [
            {"role": "system",
             "content": ("你是严格的学习上下文质量裁判。仅输出 JSON，"
                         "为下列每个维度打 0..1 分（1 最好）：\n"
                         f"{dims_desc}\n"
                         f"输出格式（不要多余文字）：{schema}")},
            {"role": "user",
             "content": f"学习目标：{goal}\n\n待评答案：\n{answer}\n\n请打分（仅 JSON）。"},
        ]


class RubricScoreCache:
    """bundle-hash 缓存：键 = sha256(上下文规范化串 + '|' + rubric.signature())。

    确定性：同一 (bundle.to_json(), rubric) 永远派生同一键；命中即复用分数，
    不再调用裁判客户端（第三层"缓存"要求）。
    """

    def __init__(self):
        self._store: dict[str, tuple[float, dict]] = {}
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key_for(context_repr: str, rubric: JudgeRubric) -> str:
        payload = f"{context_repr}|{rubric.signature()}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, key: str):
        if key in self._store:
            self.hits += 1
            return self._store[key]
        self.misses += 1
        return None

    def put(self, key: str, score: float, dim_scores: dict) -> None:
        self._store[key] = (score, dict(dim_scores))

    def __len__(self) -> int:
        return len(self._store)


def _coerce_dim_scores(raw: dict) -> dict:
    """把裁判返回的打分统一成 {dim: float} 的四维字典（缺失/非法按 0.0）。

    命中与未命中路径共用本函数，保证 judge_answer 两路返回的 dim_scores
    形状完全一致（键集合 = DIMENSIONS，值均为 float）。
    """
    out: dict[str, float] = {}
    for d in DIMENSIONS:
        v = raw.get(d, 0.0)
        try:
            fv = float(v)
        except (TypeError, ValueError):
            fv = 0.0
        out[d] = max(0.0, min(1.0, fv))
    return out


async def judge_answer(client, rubric: JudgeRubric, goal: str, answer: str,
                       cache: RubricScoreCache | None = None,
                       cache_key: str | None = None) -> tuple[float, dict]:
    """用裁判客户端按 rubric 给答案打分；命中缓存则直接复用（不再调用 client）。

    返回 (judge_score, dim_scores)，dim_scores 恒为四维 float 字典（见 _coerce_dim_scores）。
    cache/cache_key 均可选：给了才走缓存；写入缓存的即为过滤后的字典。
    """
    if cache is not None and cache_key is not None:
        cached = cache.get(cache_key)
        if cached is not None:
            cached_score, cached_dims = cached
            return cached_score, _coerce_dim_scores(cached_dims)

    dim_scores = await client.complete_json(rubric.judge_messages(goal, answer))
    if isinstance(dim_scores, list):  # 防御：裁判可能返回 list 包裹
        dim_scores = dim_scores[0] if dim_scores else {}
    if not isinstance(dim_scores, dict):
        dim_scores = {}
    score = rubric.aggregate(dim_scores)
    dims = _coerce_dim_scores(dim_scores)

    if cache is not None and cache_key is not None:
        cache.put(cache_key, score, dims)  # 存过滤后的字典
    return score, dims
