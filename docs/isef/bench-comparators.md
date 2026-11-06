# DeepSeek-V4-Flash 与对比基线调研（供物理评测 + 识别评测使用）

> 用途说明：这里列的模型只作评测对照（DeepSeek 另可作离线蒸馏教师）。产品运行时不调用其中任何一个，也不调用任何 API。

调查日期：2026-11-05。信息来自 DeepSeek 官方文档、Hugging Face 模型卡和第三方 API 聚合页。厂商自报数字未经独立复现。能直接核对官方页的结论标"已核实"，只有第三方来源或页面读不全的标"未核实"。

## a. DeepSeek-V4-Flash 是否存在

**存在，但这个名字现在对应不止一个版本。** 需要先确定要对标哪一个。

| 版本 | 时间 | 规模与能力 | 开源与 API |
|---|---|---|---|
| V4-Flash（预览） | 2026-04-24 | 总参数 284B，激活 13B，MoE，上下文 1M，仅文本 | HF 模型卡写 MIT；API id `deepseek-v4-flash` |
| V4-Flash-0731 | 2026-07-31 | HF 元数据显示 304B，仅文本（pipeline 为 Text Generation） | MIT；官方更新日志称 "public beta" |
| V4-Flash-Vision-Exp | API 2026-08-21 | 实验性多模态，HF 显示约 305B | API id `deepseek-v4-flash-vision-exp` |
| V4.1-Flash | 2026-09-10 | 总参数 552B（HF 元数据显示 763B），MoE，预填充激活 8B、解码激活 16B，架构为 Causal Encoder-Decoder，上下文 1M，原生图像输入，输出为文本 | HF 卡写 MIT；API id `deepseek-flash`；2026-10-17 更新为 V4.1-Flash-1017 |

- **旧 id 已改指向。** 官方更新日志写明：V4 Flash 与 V4 Flash Vision Exp 已退役，`deepseek-v4-flash` 和 `deepseek-v4-flash-vision-exp` 现在被路由到 V4.1-Flash（当前为 1017 版）并按 Flash 价格计费。现在调用 `deepseek-v4-flash`，拿到的是支持图像的 V4.1，不是当初的纯文本预览版。
- **关于"是否支持图像"：** 预览版和 0731 版是纯文本。V4.1-Flash 起支持图像输入，官方价格页的功能表对 `deepseek-flash` 标 Vision: Yes，`deepseek-v4-pro` 标不支持。
- **价格（官方价格页，每百万 token）：**

| 档位 | 缓存命中输入 | 缓存未命中输入 | 输出 |
|---|---|---|---|
| 闲时（峰时的一半） | $0.003 | $0.15 | $0.6 |
| 峰时（UTC 工作日 01-04 点、06-10 点） | $0.006 | $0.3 | $1.2 |

  Artificial Analysis 记 $0.30 / $1.20，与峰时价一致。旧预览版价格 $0.14 / $0.28 只在第三方页面看到，未核实。OpenRouter 上第三方托管的 V4 Flash 0423 为 $0.05–0.60 输入、$0.18–0.56 输出。
- **关闭思考：** 思考默认开启。OpenAI 格式用 `{"thinking":{"type":"disabled"}}`（SDK 里放进 `extra_body`）。推理强度可选 low/high/max，默认 high。思考模式下 temperature 等采样参数被忽略。
- **JSON 输出：** 官方功能表标 JSON Output: Yes。用法是 `response_format={'type':'json_object'}`，提示词里必须出现 "json" 并给格式示例，`max_tokens` 要给足。输出不是合法 JSON 时 `content` 可能返回空白，需要重试。官方页没说明思考模式下能否使用 JSON 输出，未核实。
- **能否本地跑：** 不能。V4.1-Flash 有 552B 参数，V4-Flash 也有 284B，只能走 API。

来源：
- [更新日志](https://api-docs.deepseek.com/updates/)
- [V4 预览公告](https://api-docs.deepseek.com/news/news260424/)
- [价格页](https://api-docs.deepseek.com/quick_start/pricing)
- [思考模式](https://api-docs.deepseek.com/guides/thinking_mode)
- [JSON 模式](https://api-docs.deepseek.com/guides/json_mode)
- [V4.1-Flash 公告](https://www.deepseek.com/en/news/deepseek-v4-1-flash/)
- [HF V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)
- [HF V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
- [HF V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
- [HF 组织页（各版本列表）](https://huggingface.co/deepseek-ai)
- [Artificial Analysis](https://artificialanalysis.ai/models/deepseek-v4-1-flash-1017)
- [OpenRouter V4 Flash](https://openrouter.ai/deepseek/deepseek-v4-flash)

## b. 识别评测（图像输入、开放集）的对照模型

**纯文本的 V4-Flash 无法参加识别评测。** 但 API 上的 `deepseek-flash`（V4.1-Flash）能接收图像，可以直接用。下面是候选模型：

| 候选 | 模型 id | 许可证 | API 与价格（每百万 token） | 备注 |
|---|---|---|---|---|
| DeepSeek-V4.1-Flash | `deepseek-flash` | MIT（HF 卡） | 官方 API，$0.15–0.3 输入 / $0.6–1.2 输出 | 与"对标 V4-Flash"一脉相承，最直接；总参数 552B |
| Gemma 4 31B-it | `google/gemma-4-31b-it` | Apache 2.0 | OpenRouter 多家托管，输入约 $0.10–0.75，输出约 $0.32–1.15 | 稠密 31B，文本+图像输入，上下文 256K；与 27B 量级最接近；HF 技报标 2026-07-02，另有来源称 2026-04-02 发布，未核实。关闭思考：去掉 `<\|think\|>`，但 31B 仍会输出空思考块 |
| Qwen3.8-27B | `Qwen/Qwen3.8-27B` | Apache 2.0 | DeepInfra 标准档 $0.15 输入 / $1.875 输出，支持视觉和 JSON | 2026-08 发布，稠密 27B，原生图像/视频输入，上下文 262K（可扩到 1M）；用 `enable_thinking:false` 关闭思考；GPQA Diamond 89.2；也是学生的本地模型 |
| Kimi K3（可选，上限参考） | `moonshotai/Kimi-K3` | Kimi K3 License | 官方 API id 为 `kimi-k3`，官方价格未取到；OpenRouter 各家输入 $0.15–4.50、输出 $9.3–22.5 | 总参数 2.8T，激活 104B，文本+图像输入；思考永远开启，无法关闭，输出贵；许可证对 MaaS 商业用途有 $2000 万年收入门槛，内部评测不受影响（我的非法律解读） |

- **学生的基座模型本身就是多模态。** Qwen3.5-4B 的 HF 标签为 image-text-to-text，Apache 2.0，自报 MMMU 77.6、MathVision 74.6。微调后的 4B 能直接接图像，识别评测可以由同一个模型完成。
- **已排除的模型：** GLM-5V-Turbo 不是开放权重，不适合作为"开放权重"对照。InternVL 在 2026 年的新版本没查到，未核实。

来源：
- [HF Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
- [DeepInfra Qwen3.8-27B](https://deepinfra.com/Qwen/Qwen3.8-27B)
- [HF Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)
- [HF Gemma 4 31B-it](https://huggingface.co/google/gemma-4-31B-it)
- [OpenRouter Gemma 4](https://openrouter.ai/google/gemma-4-31b-it)
- [HF Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3)
- [Kimi K3 LICENSE](https://huggingface.co/moonshotai/Kimi-K3/blob/main/LICENSE)
- [OpenRouter Kimi K3](https://openrouter.ai/moonshotai/kimi-k3)
- [Kimi 定价页](https://platform.kimi.ai/docs/pricing/chat)

## c. 物理基准上的公开数字

**没有找到任何 DeepSeek V4 系列在 PhyX、PhysBench 或 OlympiadBench 上的公开数字。**
- [PhyX 项目页](https://phyx-bench.github.io/) 的文本里没有榜单行，也没有 2026 年 DeepSeek、Qwen3.x、Kimi、Gemma 的行，绝对准确率只有 GPT-4o 32.5%、Claude 3.7 Sonnet 42.2%、o4-mini 45.8%（2025 年数字）。
- 官方卡上提到的 "Physical Engineering's Last Exam"（疑似 17 分）无法确认是哪个模型的数字，未核实。

只能用科学推理基准做侧面参考。GPQA 只部分涉及物理，且与"校园力学实验的系统误差归因"差距很大，只能当 sanity check：

| 模型 | GPQA Diamond | HLE |
|---|---|---|
| V4-Flash（HF 卡，非思考 / High / Max） | 71.2 / 87.4 / 88.1 | 8.1 / 29.4 / 34.8 |
| V4.1-Flash（Max） | 90.9 | 36.8 |
| Qwen3.8-27B | 89.2 | 30.8 |
| Gemma 4 31B | 84.3 | 未列出 |

- V4-Flash 的非思考与 Max 模式差 17 分，说明关掉思考后它会明显变弱。这对评测设置很重要。
- 视觉侧：Qwen3.5-4B 为 MMMU-Pro 66.3、MathVision 74.6；Gemma 4 31B 为 MMMU-Pro 76.9、MATH-Vision 85.6；Qwen3.8-27B 为 MathVision 90.0。这些都是厂商自报，评测协议不同，不可直接比较。

## 建议

1. **物理评测（文本、JSON 输出）：**
   - 主对照用 `deepseek-flash`，分两档报告：`thinking.type=disabled` 的非思考档，以及 high 强度思考的上限档。
   - 再加 Qwen3.8-27B（非思考与思考各一档）和未微调的 Qwen3.5-4B 作为消融。
   - 因为微调模型是端侧短输出，主表应以非思考档对比；思考档只作为天花板参考。
2. **识别评测（图像、开放集）：**
   - 主对照用 Gemma 4 31B 和 Qwen3.8-27B，它们与学生模型同属开放权重的稠密模型，量级也接近。
   - DeepSeek-V4.1-Flash 作为"用户点名对标"的强基线，必须在文档里写成 V4.1-Flash。
   - Kimi K3 可选，只作上限参考，因为它思考永远开启、输出贵。
3. **注意事项：**
   - 每个 API 基线都要记录调用日期、返回里的 `model` 字段和完整参数（思考开关、强度、temperature）。别名 `deepseek-v4-flash` 实际指向 V4.1-Flash-1017，会随厂商更新而变化。
   - 巨型 MoE 无法在本机运行（学生本机有 24GB 内存上限），只能走 API。API 输出只作为评测基线，绝不进入产品，也不用于微调数据。
   - 结论要措辞谨慎：只能说"在学生自建的评测集上，4B 微调模型相当于或接近某某基线"，不能外推为通用物理能力。评测集必须与 LoRA 训练数据严格隔离。
   - 开放集识别的提示词要允许输出"未知/不在列表内"，并对所有模型使用同一份标签表。
   - 发给 API 的实验照片要先检查，不能含人脸或个人信息。
4. **ISEF Rule 8：** 本报告的网址仅作为评测方案和工程文档的参考。若将来要写进论文或海报的参考文献，请学生自己核对原始页面并自行整理引用，不要直接照搬 AI 给出的引用列表。
