"""第二/三层测试配套 harness（契约 §7.2 黄金场景 + §7.3 配套工具）。

本子包只在测试侧使用，绝不 import 也不修改 core/**。提供：
    - fake_learner：§7.3 模拟学习者，把 §7.2 五场景脚本化为确定性 LHEvent 流；
    - judge_rubric：第三层 headless A/B 的评分 rubric + bundle-hash 缓存。

确定性纪律：时间全部相对固定锚点 AS_OF（day 0），禁用 wall-clock / random /
网络（默认 stub 模式）。
"""
