# P2 实验侦探：架构与模块接口（开发者必读）

> 这份文件是并行开发的唯一接口约定。改接口先改这里。所有文字面向中学生的部分用中文。

## 0. 产品与创新点（物理为主）

**产品**：拍一张力学实验装置的照片 → 认出是什么实验（开放集，力学优先）→ 把这台装置变成一个**参数化物理孪生**
→ 实验前给出预测和不确定度预算 → 带学生一步步测量，每填一行数据就和孪生对照 → 数据偏离时用贝叶斯推断找出系统误差来源，
推荐最能区分各原因的下一次测量 → 依据物理事实单讲解，并关联知识点。完全离线，本地小模型。

**自己的创新点**（与格物论文无任何相似；格物是文本题库上的解题辅导 + 代数沙箱 + 投票）：
1. **误差物理孪生**：12 个中学力学实验的可执行前向模型，带系统误差物理（摩擦、车轮/滑轮转动惯量、绳子质量、遮光片宽度、弹性限度、调零方向……）和仪器噪声模型。
2. **可辨识性图谱**（测量物理）：课本标准方案能/不能暴露哪些误差，哪一次追加测量能把它们分开（贝叶斯实验设计）。
3. **看图建模**：照片 → 装置构型 → 选定实验孪生并实例化参数。
4. **物理引擎当老师**：模型的主要训练标签来自对含噪数据做贝叶斯逆推断的物理引擎，不是教师模型的解题轨迹。
5. **真实器材闭环**：故障注入实验检验孪生的预测（sim-to-real）。
6. **探究式教学**：实验前预测、实验中实时一致性检查、按信息增益选"下一个问题/下一次测量"、误差与知识点关联。

**模型从不自己算物理**：所有数值来自物理引擎（确定性代码），模型负责看图、规划、解释和对话。
没有代码沙箱、没有工具调用循环、没有投票。蒸馏只借鉴格物的一点：教师盲做，只保留和**物理引擎判定一致**的输出；总条数 ≤ 5,248。

## 1. 目录与模块归属

```
lab_detective/
  labsim/core.py            公共接口（已定稿，见文件头建模约定）
  labsim/registry.py        自动发现 experiments/*.py 里的 Experiment 子类
  labsim/experiments/*.py   每个实验一个文件（hooke.py 是样板）
  labsim/generic.py         通用关系族 + 通用故障算子（"其他实验"和程序化任务族）
  engine.py                 贝叶斯后验（已定稿）
  panel.py                  特征面板（已定稿）
  eig.py                    期望信息增益、模拟闭环
  atlas.py                  可辨识性图谱
  uncertainty.py            不确定度预算（实验前）与传播
  twin.py                   孪生实例化：实验 id + 参数 → 预测曲线、预期读数、幽灵曲线
  facts.py                  事实单 FactSheet（引擎输出的唯一出口）
  knowledge/                实验卡 + 知识点图谱（JSON 数据 + 加载器）
  vision/                   照片 → 构型 → 实验（YOLO 线索 + VLM 提示 + 解析 + 澄清问题）
  tutor/                    探究式引导：会话状态机、上下文组装、模板讲解、声明校验器
  llm/                      模型客户端（mlx_vlm.server / OpenAI 兼容 API / 脚本化兜底）
  training/                 训练样本生成、蒸馏流水线、打包
  bench/                    物理 bench、识别 bench（生成、冻结、评分）
server/                     5090 上的脚本（引导、训练、合并、评测、转 MLX）
app.py + ui/                Streamlit 界面
tests/                      pytest，每个模块一个文件
```

## 2. 关键数据结构（JSON 可序列化）

### 2.1 FactSheet（facts.py）
```json
{"experiment_id": "hooke", "experiment_name": "...", "protocol": "standard",
 "data": {"n_points": 5, "table": [{"setting": 50, "values": [1.2]}]},
 "panel": {...Panel.to_dict()...},
 "agreement": {"quantity": "k", "value": 24.1, "uncertainty": 0.6, "unit": "N/m", "z": 2.4, "band": "consistent|tension|inconsistent"},
 "diagnosis": {"ranked": [{"cause_id": "...", "name": "...", "prob": 0.71, "evidence": "截距 +1.5 cm（z=6.2）"}],
               "not_in_list": false, "misfit_p": 0.4},
 "next_measurement": {"id": "weigh_hanger", "label": "...", "eig_bits": 0.8,
                      "predicted": [{"cause_id": "...", "value": 23.0, "unit": "g"}]},
 "fix": {"cause_id": "...", "text": "...", "predicted_rerun": "..."},
 "knowledge_points": ["kp.hooke_law", "kp.systematic_error"],
 "numbers": [{"name": "intercept", "value": 1.52, "unit": "cm"}]}
```
`numbers` 列出事实单里允许出现在讲解文字中的所有数字（校验器用）。

### 2.2 实验卡与知识点（knowledge/）
- `knowledge/data/cards/<experiment_id>.json`：
  `id, name_zh, name_en, grade("初中"|"高中"), goal, principle(文字), formula(纯文本), variables[{symbol,name,unit,role:"control|measured|derived"}],
   apparatus[], steps[], data_table{columns[]}, safety[], common_causes[cause_id], knowledge_points[kp_id], relation_family, labsim_id, yolo_cues[]`
- `knowledge/data/knowledge_points.json`：`[{id, name_zh, statement, formula, prerequisites[kp_id], misconceptions[{id, text, symptom}], experiments[id]}]`
- 加载器：`knowledge.load_card(id)`, `knowledge.all_cards()`, `knowledge.kp(id)`, `knowledge.graph()`。

### 2.3 识别结果（vision/schema.py）
```json
{"letter": "A"...,"experiment_id": "hooke"|null, "open_name": "单摆测重力加速度", "cues": ["弹簧竖直悬挂", "下端挂钩码"],
 "apparatus": ["spring", "slotted_mass", "ruler"], "confidence": 0.8, "needs_question": false}
```
字母表：课内与留出实验各一个大写字母，`Z` = 其他。所有模型（本地、API）共用同一份字母表和 JSON schema。

### 2.4 会话状态（tutor/session.py）
阶段：`identify → plan → predict → measure → check → diagnose → explain → reflect`。
`LabSession`：experiment_id、card、protocol、observations、student_predictions、fact_sheet、phase、hints_used、kp_touched、turns（只存精简摘要）。
上下文组装 `build_context(session, budget_tokens)`：按阶段只放必要信息（实验卡要点 + 最新事实单 + 最近 2 轮），**不放完整对话历史**。

### 2.5 训练样本（training/format.py）——服务器端训练脚本只认这个格式
JSONL，每行：
```json
{"id": "diag-hooke-000123", "task": "diagnose|design|recognize|guide|explain|card|knowledge",
 "source": "engine|template|distill-deepseek|distill-hy4|photo",
 "messages": [{"role": "system", "content": "..."}, {"role": "user", "content": [{"type": "image", "image": "images/xxx.jpg"}, {"type": "text", "text": "..."}]},
              {"role": "assistant", "content": "{...JSON 或中文...}"}],
 "meta": {"experiment_id": "hooke", "split": "train|val", "seed": 123}}
```
- 图片路径相对数据包根目录；图片统一缩放到长边 ≤ 768 px、JPEG 质量 85。
- `task` 和 `source` 必填；测试集样本绝不进训练包（泄漏检查器按 id 和内容哈希双重检查）。
- 系统提示：训练时看到的指令必须和推理时一致（蒸馏样本里教师看到的指令也照样放进训练提示）。

## 3. 统一约定
- 纯 numpy/scipy，不引入新依赖（需要时在 pyproject 里集中加，由主会话处理）。
- 随机性只用 `np.random.Generator`，所有生成函数接受 `seed`。
- 每个模块配 `tests/test_<模块>.py`，单个测试 < 10 秒；整套测试 < 2 分钟。
- 本机内存上限：我们启动的所有进程合计 ≤ 24 GB；测试和生成任务控制在 < 2 GB。
- 不调用任何外部 API（蒸馏脚本除外，且只由用户提供 key 后手动运行）。
- 提交信息不写 AI 工具名；AI 写的代码记进 docs/AI_USAGE.md。
