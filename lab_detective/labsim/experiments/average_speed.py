"""测量平均速度：小车沿斜面由静止滑下，测不同路程所用的时间（初中，训练实验）。

学生的做法（人教版八年级上册第一章第 4 节"测量平均速度"，扩展成多点）：
斜面保持很小的坡度，小车放在斜面顶端，金属片放在斜面下方；用刻度尺从小车车头量到金属片，
记路程 s；松手让小车由静止滑下，用停表测从松手到小车撞击金属片（听到"啪"一声）的时间 t。
课本只测全程和半程两段；这里把金属片依次放在 25、40、60、80、100、120 cm 处，每个位置测 3 次，
用 v = s/t 算各段平均速度。"single" 方案只在 100 cm 处测 3 次。

物理：小车沿斜面做匀加速运动，a = g(sinθ − μ_r cosθ) / (1 + ΣI/(m R²))，
最后一项是车轮转动惯量（转动动能分走一部分重力势能）。a 由坡度、滚动摩擦、车轮共同决定，
学生事先不知道，所以 β = √(2/a) 作为自由的线性参数：s = ½ a t² ⇒ t = β·√s。

面板坐标：横轴 = √s（√m，s 由刻度尺测得，视为准确），纵轴 = 每个位置 3 次计时的平均值 t（s）。
理想直线 t = β·√s，过原点，斜率自由（β 未知）。

计时器可选（构造参数 timer）：
- "stopwatch"（默认）：停表分辨率 0.01 s，人工计时的随机误差取 0.08 s（受训计时员的文献量级，假设）；
- "photogate"：放车挡板联动的数字计时器 + 金属片处的光电门，分辨率 0.001 s；随机误差主要来自
  小车每次滚动的差异，取 0.01 s。电子计时没有人的反应问题，计时类原因和对应的追加测量都去掉。
"""

from __future__ import annotations

import numpy as np

from ..core import (
    Cause,
    Dataset,
    Experiment,
    LinearData,
    LinearParam,
    MenuItem,
    ObsKind,
    Prior,
    Protocol,
    Reading,
)

N_REP = 3  # 每个位置计时 3 次

TIMERS = {
    "stopwatch": dict(resolution=0.01, sigma=0.08, label="停表"),
    "photogate": dict(resolution=0.001, sigma=0.01, label="数字计时器（光电门）"),
}

# 人工计时才有的原因 / 追加测量（换成光电门时去掉）
_HUMAN_CAUSES = ("stop_lag", "start_late", "start_early", "rushed_timing")
_HUMAN_KINDS = ("video", "solo")

# 原因 id
RANDOM = "average_speed.random_only"
STOP_LAG = "average_speed.stop_lag"
START_LATE = "average_speed.start_late"
START_EARLY = "average_speed.start_early"
CART_LEN = "average_speed.cart_length"
PUSHED = "average_speed.pushed"
RUSHED = "average_speed.rushed_timing"

# 每种计时偏差在各类观测里是否出现（+1 让时间变长，−1 变短，0 不出现）
#   main/release：照常两人配合计时；solo：放车的同学自己同时开表（启动同步，停表照旧听声音）；
#   video：秒表读数 − 慢动作视频读数（同一次下滑）
_OFFSET_SIGN = {
    STOP_LAG: {"main": 1.0, "release": 1.0, "solo": 1.0, "video": 1.0},
    START_LATE: {"main": -1.0, "release": -1.0, "solo": 0.0, "video": -1.0},
    START_EARLY: {"main": 1.0, "release": 1.0, "solo": 0.0, "video": 1.0},
}

TIMED_KINDS = ("main", "release", "solo", "video")


def _times(prefix_zh: str, timer: str) -> tuple[Reading, ...]:
    t = TIMERS[timer]
    return tuple(
        Reading(f"t{i + 1}", f"{prefix_zh}第 {i + 1} 次", f"{prefix_zh} trial {i + 1}", "s",
                resolution=t["resolution"], sigma=t["sigma"])
        for i in range(N_REP)
    )


def _obs_kinds(timer: str) -> dict[str, ObsKind]:
    kinds = {
        "main": ObsKind(
            "main", "金属片放在路程 s 处，计时 3 次", "time the run to the stopper, 3 trials",
            _times("时间", timer),
            setting_label_zh="路程 s（车头到金属片）", setting_label_en="distance (cart front to stopper)",
            setting_unit="cm",
        ),
        "release": ObsKind(
            "release", "用挡板挡住小车，竖直抽走挡板放车，计时 3 次",
            "release by lifting a barrier straight up, 3 trials",
            _times("挡板放车时间", timer), setting_label_zh="路程 s", setting_label_en="distance",
            setting_unit="cm",
        ),
        "solo": ObsKind(
            "solo", "放车的同学自己一手松车、一手同时按下停表，计时 3 次",
            "the releaser starts the watch with the other hand at release, 3 trials",
            _times("单人计时", timer), setting_label_zh="路程 s", setting_label_en="distance",
            setting_unit="cm",
        ),
        "video": ObsKind(
            "video", "照常计时，同时用手机慢动作（240 帧/秒）拍下同一次下滑，记"
            "（停表读数 − 视频读数），共 3 次",
            "time as usual while filming the same run in slow motion; record watch − video, 3 trials",
            tuple(
                Reading(f"dt{i + 1}", f"停表 − 视频，第 {i + 1} 次", f"watch − video, trial {i + 1}", "s",
                        resolution=TIMERS[timer]["resolution"], sigma=TIMERS[timer]["sigma"])
                for i in range(N_REP)
            ),
            setting_label_zh="路程 s", setting_label_en="distance", setting_unit="cm",
        ),
        "distance_check": ObsKind(
            "distance_check", "从小车车头重新量到金属片，与表中记录的路程比较",
            "re-measure from the cart's front to the stopper and compare",
            (Reading("s_diff", "原记录路程 − 重量路程", "recorded − re-measured distance", "cm",
                     resolution=0.1, sigma=0.05),),
            setting_label_zh="路程 s", setting_label_en="distance", setting_unit="cm",
        ),
    }
    if timer != "stopwatch":
        for k in _HUMAN_KINDS:
            kinds.pop(k)
    return kinds


class AverageSpeed(Experiment):
    id = "average_speed"
    name_zh = "测量平均速度（小车沿斜面下滑）"
    name_en = "Measuring average speed: cart rolling down a slope"

    obs_kinds = _obs_kinds("stopwatch")
    nuisance = {}

    # β = √(2/a)（s/√m）：a 约 0.17~2 m/s²（±3σ），对应很缓到较陡的斜面
    linear = (LinearParam("beta", 2.2, 0.4, "√(2/a)", "s/√m"),)

    causes = (
        Cause(
            RANDOM, "只有随机误差", "random error only",
            "放车、量路程、计时的步骤都正确，数据偏离只来自按表时机的随机快慢和读数取整。",
            "The procedure is correct; deviations come only from random timing jitter and rounding.",
            "none", ("none",),
            fix_zh="每个位置多测几次取平均。", fix_en="Repeat each run and average.",
        ),
        # 推导：计时员看到放车时同步开表，但要听到撞击声后才按停，停表晚 τ（听觉反应约 0.15~0.25 s）。
        #   t_记 = t_真 + τ = β√s + τ，每一组时间都多出同一个常数 τ。
        Cause(
            STOP_LAG, "听到撞击声才按停表", "watch stopped only after hearing the impact",
            "计时的同学放车时同步按下停表，但要等听到小车撞上金属片的声音以后才去按停，"
            "手指的动作总比撞击晚一点。",
            "The timer starts the watch together with the release, but presses stop only after "
            "hearing the cart hit the stopper, so the stop always comes a little late.",
            "y_offset", ("intercept_pos",),
            linear=(LinearParam("stop_lag", 0.20, 0.06, "停表滞后时间", "s"),),
            fix_zh="盯住小车，在它快到金属片时提前准备、看到撞上就按停；或用光电门/手机慢动作计时。",
            fix_en="Watch the cart and anticipate the impact, or use a photogate / slow-motion video.",
        ),
        # 推导：一人放车，另一人看到小车动起来才开表，开表晚 τ（视觉反应约 0.2 s）。
        #   t_记 = t_真 − τ = β√s − τ。
        Cause(
            START_LATE, "看到小车动了才开表", "watch started after the cart was already moving",
            "一个同学放车，另一个同学看到小车开始运动以后才按下停表，开表总比松手晚一点。",
            "One student releases the cart; the other starts the watch only after seeing it move, "
            "so the start is always a little late.",
            "y_offset", ("intercept_neg",),
            linear=(LinearParam("start_lag", 0.20, 0.06, "开表滞后时间", "s"),),
            fix_zh="由放车的同学喊口令、计时员按表的同时松手，或放车和开表由同一人完成。",
            fix_en="Release on the timer's count, or let one student release and start the watch together.",
        ),
        # 推导：计时员喊"开始"的同时按表，放车的同学听到口令后才松手，小车晚 τ 才出发。
        #   t_记 = t_真 + τ。和"停表滞后"在标准数据里完全同形，只能靠改变分工的追加测量分开。
        Cause(
            START_EARLY, "先按表、后松手", "watch started before the cart was released",
            "计时的同学喊\"开始\"的同时按下停表，放车的同学听到口令后才松手，小车总比停表晚一点出发。",
            "The timer calls \"go\" and starts the watch at the same moment; the other student "
            "releases the cart only after hearing the call.",
            "y_offset", ("intercept_pos",),
            linear=(LinearParam("start_lead", 0.18, 0.06, "开表提前时间", "s"),),
            fix_zh="由放车的同学自己同时松手和开表，或改为放车的同学喊口令、计时员同步按表。",
            fix_en="Let the releaser start the watch, or have the releaser give the count.",
        ),
        # 推导：路程从车尾量到金属片，但小车是车头撞上金属片，实际只跑了 s − L（L = 车长）。
        #   t = β√(s − L)；在 √s 坐标上 √(x² − L) 是上凸的，短路程处时间偏小得最多。
        Cause(
            CART_LEN, "路程从车尾量起", "distance measured from the cart's rear",
            "量路程时从小车车尾量到金属片，而小车是车头撞到金属片停下的，"
            "小车实际跑过的路程比记录的少了一个车身长。",
            "The distance was measured from the cart's rear to the stopper, but the cart stops when "
            "its front hits the stopper, so it actually travels one cart length less.",
            "x_offset", ("curve_down", "intercept_neg"),
            magnitude=Prior.uniform(4.0, 15.0, 111),
            magnitude_label_zh="车身长", magnitude_unit="cm",
            fix_zh="路程从车头量到金属片（起点和终点都对准车头）。",
            fix_en="Measure from the cart's front at the start to the stopper.",
        ),
        # 推导：松手时推了一下，初速度 v0。等效于小车从更高处 d0 = v0²/(2a) 由静止出发，
        #   经过起点时已有速度 v0：t = β(√(s + d0) − √d0)。小路程时 t ≈ β s/(2√d0)，
        #   大路程时 t ≈ β(√s − √d0)，在 √s 坐标上是下凸的。
        Cause(
            PUSHED, "松手时推了一下", "cart pushed at release",
            "放车时手顺势往前推了一下，小车一出发就带着一定的速度，而不是从静止开始运动。",
            "The cart was given a small push at release, so it starts with some speed instead of "
            "from rest.",
            "bend", ("curve_up", "intercept_neg"),
            magnitude=Prior.loguniform(1.0, 60.0, 161),
            magnitude_label_zh="等效助跑距离 v0²/(2a)", magnitude_unit="cm",
            fix_zh="用挡板挡住小车，竖直抽走挡板放车，保证从静止出发。",
            fix_en="Hold the cart with a barrier and lift the barrier straight up to release it.",
        ),
        # 推导：坡度太大时全程不到 1 s，计时员没法预判撞击时刻，每次按表的提前/滞后都不一样，
        #   相当于每个读数额外加了一个零均值、标准差 σ' 的误差：σ_总 = √(σ² + σ'²)。
        Cause(
            RUSHED, "坡度太大，来不及准确计时", "slope too steep for hand timing",
            "斜面垫得太高，小车一下子就冲到底，计时的同学来不及反应，每次按表的时机忽早忽晚。",
            "The slope is too steep; the cart reaches the stopper so quickly that the timer cannot "
            "follow it, and each press comes randomly early or late.",
            "scatter", ("scatter_excess",),
            magnitude=Prior.uniform(0.08, 0.4, 65),
            magnitude_label_zh="额外计时误差", magnitude_unit="s",
            fix_zh="减小斜面坡度，让小车运动时间长一些再计时。",
            fix_en="Lower the slope so the run takes longer.",
        ),
    )

    protocols = {
        "single": Protocol("single", "只测全程（100 cm，3 次）", "single distance (100 cm, 3 trials)",
                           (("main", 100.0),)),
        "standard": Protocol(
            "standard", "课本方案扩展：6 个位置，每处 3 次", "textbook method at 6 distances, 3 trials each",
            tuple(("main", s) for s in (25.0, 40.0, 60.0, 80.0, 100.0, 120.0)),
        ),
    }

    menu = (
        MenuItem("video_check", "照常计时，同时用手机慢动作拍下同一次下滑（100 cm，3 次）",
                 "time a 100 cm run while filming it in slow motion (3 trials)", 4.0, (("video", 100.0),)),
        MenuItem("solo_timing", "放车的同学自己同时松手和开表，在 100 cm 处计时 3 次",
                 "the releaser also starts the watch; time the 100 cm run 3 times", 3.0, (("solo", 100.0),)),
        MenuItem("distance_check", "从车头重新量一次 60 cm 位置的路程",
                 "re-measure the 60 cm distance from the cart's front", 1.0, (("distance_check", 60.0),)),
        MenuItem("barrier_release", "用挡板放车，在 100 cm 处计时 3 次",
                 "release with a barrier and time the 100 cm run 3 times", 3.0, (("release", 100.0),)),
        MenuItem("repeat_40", "在 40 cm 处再计时 3 次", "time the 40 cm run 3 more times", 2.0,
                 (("main", 40.0),)),
    )

    def __init__(self, timer: str = "stopwatch"):
        if timer not in TIMERS:
            raise ValueError(f"timer 只能是 {tuple(TIMERS)}")
        self.timer = timer
        if timer != "stopwatch":
            human = {f"{self.id}.{c}" for c in _HUMAN_CAUSES}
            self.obs_kinds = _obs_kinds(timer)
            self.causes = tuple(c for c in type(self).causes if c.id not in human)
            self.menu = tuple(m for m in type(self).menu
                              if all(k in self.obs_kinds for k, _ in m.plan))

    def relevant_nuisance(self, cause_id, kinds):
        return ()

    def predict(self, kind, settings, order, params, cause_id):
        P = params["mag"].shape[0]
        n = settings.size
        r = len(self.obs_kinds[kind].readings)
        L = len(self.linear_params(cause_id))
        mag = params["mag"]  # (P, 1)
        c = np.zeros((P, n, r))
        U = np.zeros((P, n, r, L))
        if kind in ("main", "release", "solo"):
            s = np.broadcast_to(settings[None, :] / 100.0, (P, n))  # m
            if cause_id == CART_LEN:
                s = np.maximum(s - mag / 100.0, 1e-6)
            if cause_id == PUSHED and kind != "release":
                d0 = mag / 100.0
                g = np.sqrt(s + d0) - np.sqrt(d0)
            else:
                g = np.sqrt(s)
            U[:, :, :, 0] = g[:, :, None]
        if kind in TIMED_KINDS and cause_id in _OFFSET_SIGN:
            U[:, :, :, 1] = _OFFSET_SIGN[cause_id][kind]
        if kind == "distance_check" and cause_id == CART_LEN:
            c[:] = mag[:, :, None]
        return c, U

    def reading_sigma(self, kind, params, cause_id):
        base = super().reading_sigma(kind, params, cause_id)
        if kind in TIMED_KINDS and cause_id == RUSHED:
            return np.sqrt(base**2 + params["mag"][:, :, None] ** 2)
        return base

    def linearise(self, ds: Dataset) -> LinearData:
        idx = [i for i, o in enumerate(ds.observations) if o.kind == "main"]
        s = np.array([ds.observations[i].setting for i in idx]) / 100.0
        y = np.array([np.mean(ds.observations[i].values) for i in idx])
        t = TIMERS[self.timer]
        sd1 = float(np.sqrt(t["sigma"] ** 2 + t["resolution"] ** 2 / 12))
        sy = np.full(y.shape, sd1 / np.sqrt(N_REP))
        return LinearData(np.sqrt(s), y, sy, np.array(idx), "√s (√m)", "平均时间 t (s)", None, 0.0)
