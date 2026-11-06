"""实验卡（card）任务的训练样本：精选清单之外的实验（程序化任务族 labsim/generic.py）→ 实验卡 JSON。

用途：开放集——学生做的实验不在精选清单里时，模型先按名称和器材写一张实验卡（变量、关系族、公式、步骤、记录表），
后面的孪生和诊断由代码按这张卡组合（generic.spec_from_card）。目标全部由程序化实例的 spec 确定性给出，模型不算物理。
- 字段是 knowledge/schema.py CARD_SCHEMA 的子集（name_zh、goal、formula、relation_family、variables、apparatus、
  steps、data_table），子模式直接取自 CARD_SCHEMA，保证与正式实验卡同一写法。
- 提示给实验名称、两件测量工具和学生说的自变量 / 因变量名称（目标由提示唯一确定）；目标里不写任何具体设定值。
- 注意：程序化定律只有十几种，提示只有约 50 种不同写法；generate 会删掉与验证集同提示的训练样本，
  --n-card 取 100–150 就够，再多只是重复。
- 物理组件层的组合实验没有用：镜像组合与精选实验重复，新组合全部是留出实验。
种子：任务码 7，与 splits.example_seed 同一做法。
"""

from __future__ import annotations

import json

import jsonschema
import numpy as np

from ..knowledge.schema import CARD_SCHEMA
from ..labsim.generic import procedural_task_family
from ..tutor.context import SYSTEM_PROMPT_ZH
from .format import Example
from .splits import BASE_SEED, SPLIT_CODES

TASK_CODE = 7
FIELDS = ("name_zh", "goal", "formula", "relation_family", "variables", "apparatus", "steps", "data_table")
OUTPUT_SCHEMA = {"type": "object", "required": list(FIELDS), "additionalProperties": False,
                 "properties": {k: CARD_SCHEMA["properties"][k] for k in FIELDS}}
FAMILY_FORMULA = {
    "proportional": "{y} = a·{x}", "linear": "{y} = a·{x} + b", "inverse": "{y} = a/{x}",
    "square": "{y} = a·{x}²", "sqrt": "{y} = a·√{x}", "power": "{y} = a·{x}^q",
}
FAMILY_ZH = {"proportional": "正比", "linear": "一次函数（有截距）", "inverse": "反比", "square": "与平方成正比",
             "sqrt": "与平方根成正比", "power": "幂函数"}

CARD_EXAMPLE = json.dumps({
    "name_zh": "实验名称", "goal": "探究……与……的关系。", "formula": "y = a·x", "relation_family": "proportional",
    "variables": [{"symbol": "x", "name": "要改变的量", "unit": "单位", "role": "control"},
                  {"symbol": "y", "name": "要测的量", "unit": "单位", "role": "measured"}],
    "apparatus": ["器材"], "steps": ["步骤"], "data_table": {"columns": ["列名"]}}, ensure_ascii=False)
CARD_RULES = (
    "【要求】写出这个实验的实验卡。relation_family 只能是 " + "、".join(FAMILY_FORMULA) +
    " 之一；formula 用变量符号写，比例系数写 a（有截距写 b）；variables 先写控制量再写测量量；"
    "steps 不超过 5 步、不写具体数值；data_table.columns 先写控制量列再写测量量列（带单位）。"
    "只输出一个 JSON 对象，字段顺序固定：\n" + CARD_EXAMPLE
)


def card_seed(split: str, index: int, base: int = BASE_SEED) -> int:
    ss = np.random.SeedSequence([int(base), TASK_CODE, SPLIT_CODES[split], int(index)])
    return int(ss.generate_state(1, dtype=np.uint64)[0] >> np.uint64(1))


def card_messages(name: str, tools: list[str], x_name: str = "", y_name: str = "") -> list[dict]:
    var = f"\n【学生说】要改变的是{x_name}，要测的是{y_name}" if x_name and y_name else ""
    user = (f"【任务】实验卡：学生要做一个清单之外的实验。\n【实验】{name}\n【学生手上的测量工具】{'、'.join(tools)}"
            f"{var}\n\n{CARD_RULES}")
    return [{"role": "system", "content": SYSTEM_PROMPT_ZH}, {"role": "user", "content": user}]


def card_target(exp) -> dict:
    xv, yv = exp.xv, exp.yv
    fam = exp.family
    formula = FAMILY_FORMULA.get(fam, "{y} = a·{x}").format(y=yv["symbol"], x=xv["symbol"])
    u = lambda v: f"（{v['unit']}）" if v.get("unit") else ""  # noqa: E731
    tools = list(dict.fromkeys([xv["tool"], yv["tool"]]))
    return {
        "name_zh": exp.name_zh,
        "goal": f"探究{yv['name']}与{xv['name']}的关系，判断是否满足{FAMILY_ZH.get(fam, fam)}关系。",
        "formula": formula, "relation_family": fam,
        "variables": [{"symbol": xv["symbol"], "name": xv["name"], "unit": xv.get("unit", ""), "role": "control"},
                      {"symbol": yv["symbol"], "name": yv["name"], "unit": yv.get("unit", ""), "role": "measured"}],
        "apparatus": tools,
        "steps": [f"按要求组装器材，检查{'、'.join(tools)}能正常读数。",
                  f"用{xv['tool']}把{xv['name']}调到第一个设定值，用{yv['tool']}读出{yv['name']}，记入表格。",
                  f"依次改变{xv['name']}，每次只改这一个量，读出对应的{yv['name']}，至少测 6 组。",
                  f"以{xv['name']}为横轴、{yv['name']}为纵轴作图，按 {formula} 判断关系并求出 a。"],
        "data_table": {"columns": ["序号", f"{xv['name']} {xv['symbol']}{u(xv)}", f"{yv['name']} {yv['symbol']}{u(yv)}"]},
    }


def validate_card_output(obj) -> list[str]:
    if isinstance(obj, str):
        try:
            obj = json.loads(obj)
        except json.JSONDecodeError as e:
            return [f"不是合法 JSON：{e}"]
    v = jsonschema.Draft7Validator(OUTPUT_SCHEMA)
    errs = [f"{'/'.join(map(str, e.path)) or '(根)'}: {e.message}" for e in v.iter_errors(obj)]
    if not errs and obj["relation_family"] not in FAMILY_FORMULA:
        errs.append(f"未知关系族 {obj['relation_family']}")
    return errs


def build_example(index: int, split: str, base_seed: int | None = None) -> Example:
    seed = card_seed(split, index, BASE_SEED if base_seed is None else base_seed)
    rng = np.random.default_rng(seed)
    exp, _ = next(procedural_task_family(rng, 1))
    target = card_target(exp)
    errs = validate_card_output(target)
    assert not errs, errs
    tools = list(target["apparatus"])
    family = exp.id.rsplit("_", 1)[0]
    meta = {"experiment_id": family, "instance_id": exp.id, "split": split, "seed": seed,
            "relation_family": exp.family, "generic": True}
    ex = Example(f"card-{family}-{split}-{index:06d}", "card", "engine",
                 card_messages(exp.name_zh, tools, exp.xv["name"], exp.yv["name"]) + [{"role": "assistant",
                                                       "content": json.dumps(target, ensure_ascii=False)}], meta)
    ex.validate()
    return ex
