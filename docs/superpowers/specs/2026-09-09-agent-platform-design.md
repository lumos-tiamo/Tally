# 多 Agent 数字员工平台 —— 设计文档

- 日期：2026-09-09
- 状态：设计已确认，待编写实施计划
- 主场景：金融尽调 / 财报深度分析（SEC EDGAR，后扩展 A 股）
- 投入规模：3 个月以上（旗舰）
- 硬约束：模型 API 零付费预算；本地 Apple M1 Pro / 16GB（本地推理上限约 8B）

## 1. 目标与定位

做一个 **Agent 基础设施平台 + 一个深度样板场景 + 三个轻量适配场景** 的旗舰项目。

平台负责 Agent 跑得动的底层能力（执行基底、上下文工程、模型路由、可观测与评测）；
主场景负责产出可扛深度追问的量化指标；三个轻量场景负责证明平台抽象成立。

### 为什么零预算是优势而不是缺陷

如果用强付费模型跑出好成绩，无法区分是模型强还是架构好。
用免费小模型跑出好成绩，功劳只能归架构。因此零预算恰好使下面这个论证成立：

> scaffold + 免费 Flash 级模型的得分，对比同模型裸调用、以及强模型裸调用
> —— 直接量化"架构贡献 vs 模型贡献"。

同时它逼出三项真实工程能力：多 provider 路由与降级、极致 token 预算管理、
本地/云混合推理调度。

### 平台抽象的验收标准（硬约束）

**接入一个新场景，只允许新增 AgentSpec + 工具模块，不允许修改 runtime 任何一行。**

这条约束使"我做了一个平台"成为可验证的陈述，而不是形容词。

## 2. 架构分层

```
Scenario Layer   四个场景各一份声明式定义
                 AgentSpec(persona, skills, toolset, budget, termination) + EvalSet
─────────────────────────────────────────────────────
Orchestration    TaskGraph / 子 Agent 派生与上下文隔离 / HITL 中断点 / checkpoint & resume
Context Layer    Context Ledger · Compactor · 分层 Memory · Skill Index   ← 创新点 1
Execution Layer  Code Sandbox · Tool Module Registry · Workspace FS       ← 创新点 2
Model Layer      Provider Router（配额感知/降级/缓存）· 本地 8B 模型      ← 创新点 3
Observability    Trace · Token&Cost 记账 · Eval Harness · LLM-as-Judge    ← 创新点 4
```

### 已否决的方案与理由

| 方案 | 否决理由 |
|---|---|
| B：结构化 tool calling + Context Ledger | token 消耗大，零预算下消融实验跑不完；且本质是既有项目加强版，差异度低 |
| C：确定性 workflow + 节点级自主 | 这是工作流引擎而非 Agent 平台，证明不了多 Agent 能力。**保留为消融实验对照组** |

## 3. 执行基底：Code Execution as Substrate

Agent 的唯一动作是在沙箱内**写并运行 Python**。工具不是 JSON schema，而是沙箱内可
`import` 的模块。

```python
from tools import sec
filings = sec.search_filings(cik="0000320193", form="10-K", years=3)
tbl = sec.extract_tables(filings[0], section="income_statement")
tbl.to_parquet("workspace/aapl_income.parquet")   # 中间结果落盘，不回 context
print(tbl.head())                                  # 只有 stdout 进 context
```

**关键设计后果**：Agent 只把 stdout 带回 context，其余留在文件系统。这一条同时解决：

1. context 爆炸（大数据留在磁盘）
2. long-horizon 状态保持（任务状态在磁盘而非对话历史）
3. checkpoint / 断点恢复（workspace 即 checkpoint）

### 沙箱规格

- Docker 容器；默认无网络，按场景开白名单出网
- 数据集只读挂载；workspace 读写挂载
- CPU / 内存 / 墙钟三重限额；无特权运行
- 依赖由 uv 预装为固定集合，**禁止 Agent 自行装包**（供应链风险 + 实验可复现性）

### MCP → Python stub 生成器

把 MCP server 的工具编译为沙箱内可 import 的 Python 模块，Agent 只看到函数签名与
docstring，而非完整 tool schema。避免 schema 成为 token 黑洞。

工具数超过阈值时，用本地 embedding 做**工具检索**，只注入本步相关工具的签名。

### 错误收敛循环

小模型写错代码是常态，反馈必须结构化：异常类型 + 定位 + 该模块 docstring 重新注入。
设收敛上限：同一错误连续 2 次则升级到强模型，或触发 HITL 中断点。

## 4. Context Ledger：槽位竞价式上下文预算

每次构造请求前，Ledger 取得预算 `B = 窗口 − 输出预留 − 安全余量`，各槽位在 B 内竞争：

| 槽位 | 硬下限 | 弹性 | 淘汰策略 |
|---|---|---|---|
| System / Persona | 固定 | 无 | 不可淘汰 |
| 当前 Skill 指令 | 按需 | 中 | 只载入本步命中的 skill，其余只留索引 |
| 工具 stub 签名 | 动态 | 高 | 本地 embedding 检索，只注入本步相关工具 |
| 工作区状态 | 小 | 中 | 文件树 + 产物摘要，**绝不放文件内容** |
| 长期记忆 | 可为 0 | 高 | 向量相关性 × 时效衰减 × 置信度，取 top-k |
| 会话历史 | 保尾部 | 高 | 尾部原文 + 中段 compaction 摘要 + 头部丢弃 |

### 两条关键判断

1. **超预算时先牺牲弹性大的槽位，硬下限不可侵犯。**
   防止"记忆挤掉指令"这类典型故障。
2. **Compaction 只在 step 边界触发**（占用 > 70%），不在推理中途触发。
   中途压缩会打断推理链。压缩产物写入 workspace 可回溯，不丢弃。

### 可审计性

每次构造产出一条 ledger 记录进 trace：各槽位占用 token 数、淘汰了什么、淘汰原因。
使"context 是怎么来的"完全可审计，并可输出 token 分配堆叠图。

## 5. Provider Router：零成本的实现基础

### 能力分级路由

| 环节 | 路由目标 |
|---|---|
| plan / decide / reflect | 免费强模型（Gemini Flash 层 / GLM-Flash / Groq / Cerebras） |
| extract / classify / rewrite | 本地 Qwen3-8B |
| embed / rerank | 本地 BGE-m3 + bge-reranker |

### 机制

- **配额感知降级**：每 provider 维护配额窗口计数，接近上限主动切换；429 / 超时经指数
  退避后降级到下一级
- **请求指纹缓存**：`(model, prompt_hash) → 结果` 落盘。这是消融实验能否跑完的决定性
  因素（7 组消融 × 多轮迭代，无缓存则免费额度撑不过第一天）
- 全链路 token 与成本记账，按 Agent / 任务 / 场景三个维度归属

## 6. Eval 体系

### Ground truth 构造流水线

1. **样本**：40–60 家公司 × 3 年 ≈ 150 份 10-K 作主集，另留 20 份 held-out 防过拟合。
   必须含难例：财务重述、分部报告、非 GAAP 调节、并购年份。
2. **真值来源**：SEC XBRL `companyfacts` API。
   **已知坑**：同一概念在不同公司用不同 tag（`Revenues` /
   `RevenueFromContractWithCustomerExcludingAssessedTax` / …），需构建 **US-GAAP 概念
   映射层**归一到统一字段。此层是护城河——有它才能半自动扩展到数百份真值。
3. **任务分三档**（产出能力曲线，比单一分数信息量更大）：
   - **L1 抽取**：全文 → 12 个财务字段 + 页级引用
   - **L2 计算**：8 个财务比率，必须给出分子分母来源
   - **L3 判断**：跨 3 年趋势归因与风险识别，用 judge + 人工抽检

### 四类自动指标

| 指标 | 抓什么 |
|---|---|
| 数值准确率（容差分档：精确 / ±0.5% / ±2%） | 基础正确性 |
| **引用可验证率** | 引用页能否找到该数字——幻觉的硬指标 |
| **计算一致性** | 比率是否真等于其声明的分子÷分母——抓"数对算错""算对数编" |
| **拒答正确性** | 文档中确实缺失的字段是否正确拒答而非编造 |

### Trace 级指标

步数、总 token、墙钟耗时、工具失败率、沙箱异常恢复成功率、单份报告成本。

### LLM-as-judge 校准（不可省）

人工标注 50 条作校准集，报告 judge 与人工的一致率（Cohen's κ）。
**κ < 0.6 的维度不用 judge，改人工抽检。** 主动报告评测工具自身的不可靠性。

### 消融实验组

| 实验组 | 变更 | 证明什么 |
|---|---|---|
| Full | — | baseline |
| −Ledger | 朴素尾部截断 | context 工程的价值 |
| −CodeExec | 改纯 tool calling | 执行范式的 token 与准确率收益 |
| −CiteCheck | 去掉引用校验回环 | 幻觉抑制的价值 |
| −Memory | 无跨会话记忆 | 记忆层的价值 |
| Workflow-C | 固化 DAG | 自主性的成本/收益权衡 |
| Strong-naked | 付费强模型裸调用（少量额度） | **架构贡献 vs 模型贡献** |

**关于 −Memory**：本场景跨会话依赖弱，该组很可能提升不明显。
如实报告"记忆层在本场景收益有限"是加分项，证明用数据说话而非护着自己的设计。

## 7. 三个轻量适配场景

严格遵守"不改 runtime"约束，各自压榨不同的平台能力：

| 场景 | 压榨的核心平台能力 |
|---|---|
| 垂类 Deep Research | 并行子 Agent 隔离 + 来源可信度与引用溯源 |
| 企业 BI 分析 | Schema 上下文注入 + 执行安全边界（防写、防全表扫） |
| 代码工程 Agent | 沙箱文件系统 + 迭代收敛循环（测试作反馈信号） |

四场景对**上下文形态、工具形态、终止条件**的要求完全不同。能用同一 runtime 跑通，
即为"抽象正确"的硬证明。

交付指标：**接入新场景平均新增代码行数 / 耗时**。

A 股中文年报接入作为"跨语言 / 异构文档泛化"的补充证据。

## 8. 路线图（3 个月 / 14 周）

排序原则：**观测先于优化，baseline 先于架构。** 无基线数字则所有优化无法证明有效。

| 阶段 | 周 | 内容 | 交付的数字 |
|---|---|---|---|
| 0 地基 | 1 | SEC 拉取 + XBRL 概念映射 + 30 份 L1 真值集 + trace/记账骨架 + 最笨 baseline（全文硬塞） | baseline 分数、token 分布图 |
| 1 执行基底 | 2–4 | Docker 沙箱 + workspace + MCP→Python stub 生成器 + SEC 工具模块 + 错误收敛循环 | L1 提升幅度、token 降幅 |
| 2 Context | 5–7 | 槽位竞价 + step 边界 compaction + 工具检索 + 分层记忆 | −Ledger 消融数据、token 分配图 |
| 3 编排 | 8–9 | 子 Agent 隔离 + checkpoint/resume + HITL + 引用校验回环；L2/L3 上线 | 中断恢复成功率、引用可验证率 |
| 4 成本 | 10–11 | 分级路由 + 配额降级 + 请求缓存 + 本地 8B/BGE 接入 | 成本前后对比 |
| 5 平台性 | 12–13 | 接入三场景，严守"不改 runtime"；A 股年报泛化 | 接入新场景平均新增行数 / 耗时 |
| 6 收口 | 14 | 完整消融 + judge 校准 + 开源文档 + 技术博客 | 全套指标表 |

每阶段结束都是一个可立即写入简历的状态。中途停在阶段 3 仍显著强于现状。

## 9. 简历表述原则

改写前后对比：

> **原**：设计基于分层记忆机制的 Agent 上下文体系，将角色人格、长期记忆、任务焦点与
> 会话历史动态注入 Prompt，实现跨会话状态保持。

> **改**：设计 Context Ledger 上下文预算机制，以槽位竞价在 token 预算内分配
> persona / skill / 工具签名 / 工作区状态 / 长期记忆 / 会话历史，硬下限保护关键槽位，
> 并在 step 边界触发 compaction 避免打断推理链；全链路 context 组成可审计。消融显示
> 移除该机制后 L1 抽取准确率下降 XX pt、单任务 token 上升 XX%。

（本节所有数字均为格式占位符，须由阶段 6 的实测结果替换；禁止在未实测前写入简历。）

差别不在辞藻，而在**每个决策都带理由，每个理由都带数据**。

四条核心叙述：

1. **执行范式**：Agent 以沙箱内写 Python 为唯一动作，工具编译为可 import 模块而非注入
   schema，中间结果落盘、仅 stdout 回流 context——同时解决 context 爆炸、long-horizon
   状态与断点恢复
2. **上下文工程**：Context Ledger 预算竞价 + step 边界 compaction + 工具检索
3. **成本工程**：能力分级路由 + 配额感知降级 + 请求指纹缓存 + 本地小模型承接固定环节，
   在零付费预算下完成 150 份文档 × 7 组消融的完整评测
4. **评测体系**：基于 SEC XBRL 半自动构造 ground truth，四类自动指标（含引用可验证率与
   拒答正确性），并报告 LLM-judge 与人工标注的 κ 校准值

压顶话术（数字待实测，实验设计已成立）：

> 在免费 Flash 级模型上，scaffold 使 L2 财务比率任务准确率从 43% → 91%，
> 超过将模型换为强付费模型裸调用的 76%——架构贡献大于模型升级贡献。

## 10. 主要风险

| 风险 | 缓解 |
|---|---|
| 8B 本地模型写不出可用代码 | 决策环节走免费云强模型，本地只承接抽取/分类/embed；错误收敛循环 + 升级机制 |
| 免费额度不足以跑完消融 | 请求指纹缓存（决定性）+ eval 分层抽样 + 阶段 4 的成本工程提前到需要时 |
| XBRL 概念映射工作量超预期 | 阶段 0 先只覆盖 12 个 L1 字段的映射，L2/L3 字段按需增量扩展 |
| 沙箱依赖管理拖慢迭代 | 依赖固定集合、镜像预构建、禁止 Agent 装包 |
| 四场景摊薄深度 | 主场景独占深度 eval 与消融；三场景只验证抽象，不做深度评测 |
| 10-K 表格解析质量差导致指标失真 | 阶段 0 baseline 即暴露该问题；必要时本地小模型专做表格结构化 |

## 附录 A：L1 字段与 L2 比率定义

字段与比率必须在阶段 0 就锁定，否则真值集无法构造、跨阶段指标无法对比。

### L1 —— 12 个抽取字段

来源均取自 XBRL `companyfacts` 的 us-gaap taxonomy，经概念映射层归一。

| # | 字段 | 归一后键名 | 常见 XBRL tag（映射层需覆盖多个） |
|---|---|---|---|
| 1 | 营业收入 | `revenue` | `Revenues`, `RevenueFromContractWithCustomerExcludingAssessedTax` |
| 2 | 营业成本 | `cost_of_revenue` | `CostOfRevenue`, `CostOfGoodsAndServicesSold` |
| 3 | 营业利润 | `operating_income` | `OperatingIncomeLoss` |
| 4 | 净利润 | `net_income` | `NetIncomeLoss` |
| 5 | 总资产 | `total_assets` | `Assets` |
| 6 | 总负债 | `total_liabilities` | `Liabilities` |
| 7 | 股东权益 | `total_equity` | `StockholdersEquity` |
| 8 | 流动资产 | `current_assets` | `AssetsCurrent` |
| 9 | 流动负债 | `current_liabilities` | `LiabilitiesCurrent` |
| 10 | 应收账款净额 | `accounts_receivable` | `AccountsReceivableNetCurrent` |
| 11 | 经营活动现金流 | `cash_from_operations` | `NetCashProvidedByUsedInOperatingActivities` |
| 12 | 资本支出 | `capex` | `PaymentsToAcquirePropertyPlantAndEquipment` |

每个字段的输出契约：`{value, unit, fiscal_year, page, quote}`。
`page` 与 `quote` 用于计算引用可验证率；字段在文档中确实缺失时须输出
`{value: null, reason: "not_disclosed"}`，用于计算拒答正确性。

### L2 —— 8 个财务比率

每个比率必须同时输出 `value`、`numerator`、`denominator` 及两者的 L1 字段来源键，
用于计算一致性校验（校验 `value ≈ numerator / denominator`）。

| # | 比率 | 公式 |
|---|---|---|
| 1 | 毛利率 | (revenue − cost_of_revenue) / revenue |
| 2 | 营业利润率 | operating_income / revenue |
| 3 | 净利率 | net_income / revenue |
| 4 | ROE | net_income / total_equity |
| 5 | ROA | net_income / total_assets |
| 6 | 流动比率 | current_assets / current_liabilities |
| 7 | 应收账款周转率 | revenue / accounts_receivable |
| 8 | 自由现金流 | cash_from_operations − capex |

注：第 8 项为绝对额而非比率，但沿用同一"分子/分母来源可溯"的校验契约
（`numerator = cash_from_operations`、`denominator = capex`、运算符为减法）。

### 容差分档

| 档位 | 判定 |
|---|---|
| 精确 | 相对误差 = 0 |
| 严格 | 相对误差 ≤ 0.5% |
| 宽松 | 相对误差 ≤ 2% |

L1 主指标报告"严格"档；L2 因存在多步运算，主指标报告"宽松"档，同时附报三档全表。

## 附录 B：本文档的实施拆分

本设计覆盖 14 周，**范围过大，不适合单份实施计划**。实施应按阶段拆分，
每个阶段独立走 plan → 实现 → 验证 的循环：

- 首份实施计划只覆盖 **阶段 0（地基与 baseline）**
- 阶段 1 起，每阶段在上一阶段实测数字产出后再编写计划
- 理由与路线图的排序原则一致：无基线数字则无法判断下一步该优化什么
