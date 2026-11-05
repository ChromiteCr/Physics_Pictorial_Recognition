"""识别字母表：所有模型（本地、API、规则）共用的同一份单字母代码。

A–F 是 6 个课内（训练）实验，G–L 是 6 个留出实验，Z = 其他（不在表里）。
字母一旦定下就不改：训练样本、bench 冻结集和推理都按它读写。
experiment_id 与 labsim 里 ``Experiment.id`` 一致；Z 的 experiment_id 为 None。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LetterEntry:
    letter: str
    experiment_id: str | None
    name_zh: str
    short_zh: str
    held_out: bool
    look_zh: str  # 照片里一般长什么样（写进提示词，帮模型对照）


OTHER_LETTER = "Z"

TABLE: tuple[LetterEntry, ...] = (
    LetterEntry("A", "hooke", "探究弹簧弹力与形变量的关系（胡克定律）", "胡克定律", False,
                "弹簧竖直悬挂在铁架台上，下端挂钩码，旁边竖直放刻度尺"),
    LetterEntry("B", "friction", "探究滑动摩擦力大小的影响因素", "滑动摩擦力", False,
                "弹簧测力计水平拉着木块（或小车）在木板/桌面上匀速运动"),
    LetterEntry("C", "average_speed", "测量平均速度", "平均速度", False,
                "小车从斜面上滑下，斜面上有刻度或金属片，用秒表计时"),
    LetterEntry("D", "newton2", "探究加速度与力、质量的关系（牛顿第二定律）", "牛顿第二定律", False,
                "水平轨道上的小车用细线跨过滑轮连着钩码，配打点计时器或光电门"),
    LetterEntry("E", "momentum", "验证动量守恒定律", "动量守恒", False,
                "同一条轨道上两辆小车（滑块）相向或追碰，配光电门或打点计时器"),
    LetterEntry("F", "energy", "验证机械能守恒定律", "机械能守恒", False,
                "小车/物体从倾斜轨道上由静止滑下，测下落高度和末速度（光电门）"),
    LetterEntry("G", "incline_friction", "测量斜面上的动摩擦因数", "斜面摩擦", True,
                "物块放在倾角可调的斜面上，用测力计沿斜面拉或让它滑下"),
    LetterEntry("H", "free_fall", "研究自由落体运动（测重力加速度）", "自由落体", True,
                "重物（重锤）竖直下落，连着纸带穿过竖直的打点计时器，或穿过光电门"),
    LetterEntry("I", "atwood", "阿特伍德机（绳跨定滑轮两端挂重物）", "阿特伍德机", True,
                "一根细线跨过高处的定滑轮，两端各挂一个重物"),
    LetterEntry("J", "pendulum", "用单摆测量重力加速度", "单摆", True,
                "小球用细线挂在铁架台上，来回摆动，旁边有秒表或刻度尺"),
    LetterEntry("K", "projectile", "研究平抛运动", "平抛运动", True,
                "小球从斜槽末端水平飞出，落在竖直的白纸/坐标板或地面上"),
    LetterEntry("L", "lever", "探究杠杆的平衡条件", "杠杆平衡", True,
                "带刻度的杠杆架在支点上，两侧挂钩码"),
    LetterEntry("Z", None, "其他实验（不在上表中）", "其他", False, ""),
)

LETTERS: tuple[str, ...] = tuple(e.letter for e in TABLE)
EXPERIMENT_IDS: tuple[str, ...] = tuple(e.experiment_id for e in TABLE if e.experiment_id)
_BY_LETTER = {e.letter: e for e in TABLE}
_BY_ID = {e.experiment_id: e for e in TABLE if e.experiment_id}


def entry(letter: str) -> LetterEntry:
    return _BY_LETTER[letter]


def experiment_for(letter: str) -> str | None:
    """字母 → experiment_id（Z → None）。未知字母抛 KeyError。"""
    return _BY_LETTER[letter].experiment_id


def letter_for(experiment_id: str | None) -> str:
    """experiment_id → 字母；None 或不在表里 → Z。"""
    if experiment_id is None:
        return OTHER_LETTER
    e = _BY_ID.get(experiment_id)
    return e.letter if e else OTHER_LETTER


def name_zh(letter: str) -> str:
    return _BY_LETTER[letter].name_zh


def normalize_letter(raw: object) -> str | None:
    """容错读取字母：'a'、'A.'、'(A)'、'A：胡克' → 'A'；认不出返回 None。"""
    if not isinstance(raw, str):
        return None
    s = raw.strip().strip("（）()[]【】\"'“”").strip()
    if not s:
        return None
    c = s[0].upper()
    if c in _BY_LETTER and (len(s) == 1 or not s[1].isalpha()):
        return c
    return None


def table_text(with_look: bool = True) -> str:
    """提示词里用的字母表（每行一个实验，顺序固定）。"""
    lines = []
    for e in TABLE:
        line = f"{e.letter} = {e.name_zh}"
        if with_look and e.look_zh:
            line += f"；典型装置：{e.look_zh}"
        lines.append(line)
    return "\n".join(lines)


# 旧版规则表（physics.EXPERIMENT_TYPES）的 id → 新 id
OLD_ID_MAP: dict[str, str] = {
    "friction": "friction",
    "average_speed": "average_speed",
    "hooke_law": "hooke",
    "newton_second_law": "newton2",
    "momentum_conservation": "momentum",
    "energy_conservation_input": "energy",
}
