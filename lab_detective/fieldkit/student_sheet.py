"""生成 docs/student/01_手推清单.md：精选课内训练实验的每个原因，只给物理过程的描述，留空让学生自己推。

只用原因的 name_zh 和 description_zh（描述按约定只写物理过程）；不放预期特征、修正建议（常常点明了分开它的测量）、
幅度标签（有些含效应公式）。原因有改动时重新生成：
    python -m lab_detective.fieldkit.student_sheet > docs/student/01_手推清单.md
"""

from __future__ import annotations

import warnings

import numpy as np

from ..labsim.core import Dataset, Observation
from ..labsim.registry import training_experiments

BLANK = "＿" * 28

INTRO = """# 01 手推清单：每个原因的数据特征和能分开它的测量

> 这份清单由 `lab_detective/fieldkit/student_sheet.py` 从代码生成，原因有改动时会重新生成。
> 下面是目前精选的课内训练实验。它们只是起点，其他装置由物理组件拼出来，不在这份清单里。

## 怎么用

1. **先推，再看。** 在读代码、物理卡、实验卡、密封卡之前，自己推出每个原因的两件事：
   - 按课本标准方案做实验时，数据（作图后）会是什么样子；
   - 哪一次追加测量能把它和同一实验里的其他原因分开，为什么。
2. 每个原因下面只写了**发生了什么**（物理过程）。特征、公式、修正方法都故意没写。
3. 推导写在空白处。可以打印出来手写，也可以复制一份在副本上写。每页写上日期。
4. 描述特征时用这套词：截距偏正、截距偏负、斜率偏大、斜率偏小、向上弯、向下弯、散布偏大、散布偏小、
   随测量顺序漂移、看不出来。可以同时有几个，也可以说"取决于幅度或方向"。
5. 全部推完以后，再去对照物理卡和代码。你推的和代码不一致的地方单独记下来（写清是哪个原因、你的推导、代码的说法），
   交给开发者。这些不一致本身就是检验孪生物理的一部分，不要为了"对上"去改你的推导。
6. 每个实验的第一个原因"只有随机误差"是参照，不用推。

"""


def _axes(exp) -> tuple[str, str, str]:
    """标准方案作图的横轴、纵轴，以及理想关系里斜率、截距是否已知。"""
    ds = exp.simulate(np.random.default_rng(0), exp.random_only_id, "standard", noise_scale=1e-9)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lin = exp.linearise(Dataset(exp.id, [Observation(o.kind, o.setting, o.values) for o in ds.observations]))
    slope = "斜率按理论已知" if lin.ideal_slope is not None else "斜率未知（要从数据里求）"
    if lin.ideal_intercept is None or not np.isfinite(lin.ideal_intercept):
        icpt = "截距未知"
    else:
        icpt = f"截距理论上为 {lin.ideal_intercept:g}"
    return lin.x_label, lin.y_label, f"{slope}，{icpt}"


def _plan_text(exp) -> str:
    plan = exp.protocols["standard"].plan
    kind = plan[0][0]
    ok = exp.obs_kinds[kind]
    vals = "、".join(f"{s:g}" for _, s in plan)
    unit = ok.setting_unit
    reading = "、".join(f"{r.label_zh}（{r.unit}）" for r in ok.readings)
    return f"{exp.protocols['standard'].label_zh}：{ok.setting_label_zh} 取 {vals} {unit}，每次{ok.label_zh}，记录{reading}。"


def render() -> str:
    parts = [INTRO]
    for exp in sorted(training_experiments(), key=lambda e: e.id):
        x, y, rel = _axes(exp)
        parts.append(f"## {exp.name_zh}（`{exp.id}`）\n")
        parts.append(f"- 标准方案：{_plan_text(exp)}")
        parts.append(f"- 作图：横轴 {x}，纵轴 {y}；{rel}。")
        menu = "；".join(f"{m.label_zh}" for m in exp.menu)
        parts.append(f"- 产品里可选的追加测量（你也可以提出别的）：{menu}。")
        parts.append(f"- 参照：只有随机误差（`{exp.random_only_id}`）。\n")
        n = 0
        for c in exp.causes:
            if c.id == exp.random_only_id:
                continue
            n += 1
            parts.append(f"### {n}. {c.name_zh}（`{c.id}`）\n")
            parts.append(f"{c.description_zh}\n")
            parts.append(f"- 标准方案下数据会怎样：\n\n  {BLANK}\n\n  {BLANK}\n")
            parts.append(f"- 推导要点：\n\n  {BLANK}\n\n  {BLANK}\n")
            parts.append(f"- 能把它分开的追加测量，以及为什么：\n\n  {BLANK}\n\n  {BLANK}\n")
    return "\n".join(parts).rstrip() + "\n"


if __name__ == "__main__":
    print(render(), end="")
