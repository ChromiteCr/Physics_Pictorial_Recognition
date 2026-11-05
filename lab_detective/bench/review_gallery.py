"""识别照片审核图册：生成一个离线、自包含的 review.html，让学生快速核实每张照片的实验标签。

    python -m lab_detective.bench.review_gallery [--manifest results/recognition/manifest.jsonl]
                                                 [--out results/recognition/review.html]
                                                 [--split results/bench/recognition/split.json]

标签必须由学生本人亲手确认（设计文档 §16.5）。这个页面因此刻意做成"没点就不算"：
- 页面里**没有任何预选**：规则给的暂定字母只作为提示文字，不会变成已选状态；
  规则没有候选时（暂定 Z 只是"规则没候选"，不是"其他实验"）连提示都不给字母；
- 导出的 review.json 只含学生点过字母（或按 y 采纳"规则有把握"提示）的照片，格式恰好是
  ``{"<照片id>": "<字母>"}``；没点的照片不出现，bench 里就保持"暂定"并带标记；
- 可以隐藏全部提示做盲审，避免暂定标签带偏判断。

操作：点字母按钮，或直接按键盘 A–L / Z；确认后自动跳到下一张未确认的；
←/→ 切换照片，Enter 或空格放大，y 采纳提示（仅"规则有把握"的单字母提示），Backspace 清除，Esc 关闭放大图。
进度自动存在浏览器 localStorage（读写失败不影响使用），也可随时"导入 review.json"接着做。
下载的文件通常落在 ~/Downloads，生成 bench 时用 ``--review`` 指给它。

缩略图以 base64 嵌进页面（页面可单独拷走）；点击放大时读相对路径 images/<id>.jpg，
页面与 images/ 目录分开后自动退回缩略图。
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import io
import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

from ..vision.cues import YOLO_CLASS_ZH
from ..vision.letters import OTHER_LETTER, TABLE, entry
from ..vision.som import load_image, resize_long_side

REPO = Path(__file__).resolve().parents[2]
MANIFEST_DEFAULT = REPO / "results" / "recognition" / "manifest.jsonl"
OUT_DEFAULT = REPO / "results" / "recognition" / "review.html"
SPLIT_DEFAULT = REPO / "results" / "bench" / "recognition" / "split.json"
THUMB_SIDE = 420
THUMB_QUALITY = 72


# ── 数据准备 ───────────────────────────────────────────────────────────


def read_manifest(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def thumbnail_uri(path: str | Path, side: int = THUMB_SIDE, quality: int = THUMB_QUALITY) -> str:
    """缩略图 → base64 data URI（JPEG）。"""
    im, _ = resize_long_side(load_image(path), side)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def hint_for(row: dict) -> dict:
    """规则给学生的提示（不是预选）。

    confident：规则有把握的单个字母，可以按 y 采纳；
    tied：规则分不开的字母集合；
    none：规则一个候选也没有（暂定 Z 不代表"其他实验"）。
    """
    prov = row.get("provisional_letter")
    if row.get("rule_confident") and prov and prov != OTHER_LETTER:
        return {"kind": "confident", "letters": [prov],
                "text": f"规则把握较大：{prov} {entry(prov).short_zh}（仍需你确认）"}
    tied = list(row.get("rule_tied") or []) or [c["letter"] for c in row.get("rule_candidates", [])]
    tied = [c for c in tied if c != OTHER_LETTER]
    if tied:
        return {"kind": "tied", "letters": tied, "text": "规则分不开：" + "、".join(tied) + " 之一"}
    return {"kind": "none", "letters": [],
            "text": "规则没有候选（不代表是“其他实验”，请看图判断）"}


def boxes_text(row: dict) -> str:
    """真值框里的器材清单，如 '小车×2、轨道'（给学生看图时对照；盲审时一并隐藏）。"""
    n = Counter(b.get("class_name", "?") for b in row.get("gt_boxes", []))
    parts = [f"{YOLO_CLASS_ZH.get(k, k)}×{v}" if v > 1 else YOLO_CLASS_ZH.get(k, k) for k, v in sorted(n.items())]
    return "、".join(parts)


def load_split_info(path: str | Path | None) -> dict[str, dict]:
    """读 split.json 里的 photos 字段 {id: {split, session, taken_at}}；文件不存在返回空。"""
    if path is None or not Path(path).exists():
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8")).get("photos", {})


def gallery_id(ids: list[str]) -> str:
    """图册指纹（localStorage 的键）：照片集合变了，旧进度不会串进来。"""
    return hashlib.sha1("\n".join(sorted(ids)).encode("utf-8")).hexdigest()[:10]


def build_payload(rows: list[dict], image_root: Path, split: dict[str, dict], thumb_side: int) -> dict:
    photos = []
    for r in rows:
        img = image_root / r["file"]
        if not img.exists():
            raise FileNotFoundError(f"manifest 里的图片不存在：{img}")
        sp = split.get(r["id"], {})
        photos.append({
            "id": r["id"],
            "file": r["file"],
            "thumb": thumbnail_uri(img, thumb_side),
            "hint": hint_for(r),
            "boxes": boxes_text(r),
            "test": sp.get("split") == "test",
            "session": sp.get("session"),
            "taken_at": sp.get("taken_at"),
        })
    # 有拍摄时间就按时间排（同一场景的照片挨在一起，便于连续标注），否则按 id
    photos.sort(key=lambda p: (p["taken_at"] or "", p["id"]))
    letters = [{"letter": e.letter, "short": e.short_zh, "name": e.name_zh, "look": e.look_zh,
                "held_out": e.held_out} for e in TABLE]
    return {"gid": gallery_id([p["id"] for p in photos]), "photos": photos, "letters": letters}


def render(payload: dict, title: str = "识别照片审核") -> str:
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    # 防止数据里的 "</script>" 提前结束脚本块；U+2028/2029 在旧浏览器里是行终止符
    data = data.replace("</", "<\\/").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    return TEMPLATE.replace("__TITLE__", html.escape(title)).replace("__DATA__", data)


def build_gallery(
    manifest: str | Path = MANIFEST_DEFAULT,
    out: str | Path = OUT_DEFAULT,
    image_root: str | Path | None = None,
    split: str | Path | None = SPLIT_DEFAULT,
    thumb_side: int = THUMB_SIDE,
) -> dict:
    """生成 review.html，返回摘要。image_root 默认是 manifest 所在目录（file 字段相对它）。"""
    manifest, out = Path(manifest), Path(out)
    rows = read_manifest(manifest)
    payload = build_payload(rows, Path(image_root) if image_root else manifest.parent,
                            load_split_info(split), thumb_side)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = render(payload)
    out.write_text(text, encoding="utf-8")
    kinds = Counter(p["hint"]["kind"] for p in payload["photos"])
    return {"out": str(out), "n_photos": len(rows), "n_test": sum(p["test"] for p in payload["photos"]),
            "hint_kinds": dict(kinds), "bytes": len(text.encode("utf-8")), "gallery_id": payload["gid"]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="生成识别照片审核图册 review.html")
    ap.add_argument("--manifest", type=Path, default=MANIFEST_DEFAULT)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--image-root", type=Path, default=None, help="默认 manifest 所在目录")
    ap.add_argument("--split", type=Path, default=SPLIT_DEFAULT, help="split.json（可选，用来标出测试集）")
    ap.add_argument("--thumb-side", type=int, default=THUMB_SIDE)
    a = ap.parse_args(argv)
    s = build_gallery(a.manifest, a.out, a.image_root, a.split, a.thumb_side)
    print(json.dumps(s, ensure_ascii=False, indent=2))
    print(f"用浏览器打开 {a.out}；点完后点“下载 review.json”，再用 --review 把它交给 recognition_bench。")
    return 0


# ── 页面模板 ───────────────────────────────────────────────────────────
# 逻辑函数放在 LOGIC-BEGIN/END 之间，不碰 DOM，测试里用 node 单独跑这一段。

TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root {
  --bg: #f6f5f1; --panel: #ffffff; --ink: #1d2433; --muted: #5d6678; --line: #d9d6cc;
  --accent: #1f5fbf; --accent-ink: #ffffff; --ok: #1b7a43; --ok-bg: #e3f3e8;
  --warn: #8a5a00; --warn-bg: #fbf0d6; --test: #7a2bbf; --focus: #e8590c;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #15181f; --panel: #1f2430; --ink: #e8ebf2; --muted: #9aa3b5; --line: #343b4c;
    --accent: #6ea2f2; --accent-ink: #0b1220; --ok: #6fd39b; --ok-bg: #173325;
    --warn: #f0c36a; --warn-bg: #3a2f14; --test: #c79bf2; --focus: #ff9a4d;
  }
}
:root[data-theme="dark"] {
  --bg: #15181f; --panel: #1f2430; --ink: #e8ebf2; --muted: #9aa3b5; --line: #343b4c;
  --accent: #6ea2f2; --accent-ink: #0b1220; --ok: #6fd39b; --ok-bg: #173325;
  --warn: #f0c36a; --warn-bg: #3a2f14; --test: #c79bf2; --focus: #ff9a4d;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink);
  font: 15px/1.5 system-ui, -apple-system, "PingFang SC", "Microsoft YaHei", "Noto Sans CJK SC", sans-serif; }
header { position: sticky; top: 0; z-index: 5; background: var(--panel); border-bottom: 1px solid var(--line);
  padding: 10px 16px; display: flex; flex-wrap: wrap; gap: 8px 14px; align-items: center; }
header h1 { font-size: 17px; margin: 0 6px 0 0; }
.stat { font-variant-numeric: tabular-nums; color: var(--muted); }
.stat b { color: var(--ink); }
.bar { flex: 1 1 120px; min-width: 100px; height: 8px; border-radius: 4px; background: var(--line); overflow: hidden; }
.bar > i { display: block; height: 100%; width: 0; background: var(--ok); transition: width .2s; }
button, select { font: inherit; color: var(--ink); background: var(--panel); border: 1px solid var(--line);
  border-radius: 6px; padding: 5px 10px; cursor: pointer; }
button:hover { border-color: var(--accent); }
button.primary { background: var(--accent); color: var(--accent-ink); border-color: var(--accent); }
button:focus-visible, select:focus-visible, input:focus-visible { outline: 2px solid var(--focus); outline-offset: 2px; }
label.chk { display: inline-flex; gap: 6px; align-items: center; color: var(--muted); cursor: pointer; }
details.ref { padding: 6px 16px 10px; background: var(--panel); border-bottom: 1px solid var(--line); }
details.ref summary { cursor: pointer; color: var(--muted); }
details.ref table { border-collapse: collapse; margin-top: 8px; width: 100%; font-size: 13px; }
details.ref td { padding: 3px 8px; border-top: 1px solid var(--line); vertical-align: top; }
details.ref td:first-child { font-weight: 700; white-space: nowrap; }
main { padding: 16px; display: grid; gap: 14px; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); }
.card { background: var(--panel); border: 2px solid var(--line); border-radius: 10px; overflow: hidden;
  display: flex; flex-direction: column; }
.card.hidden { display: none; }
.card.cur { border-color: var(--focus); }
.card.done { border-color: var(--ok); }
.card .pic { display: block; width: 100%; aspect-ratio: 4 / 3; object-fit: contain; background: #0003; cursor: zoom-in; }
.meta { padding: 6px 10px 0; display: flex; gap: 6px; flex-wrap: wrap; align-items: center; font-size: 12px; color: var(--muted); }
.meta .id { color: var(--ink); font-weight: 600; }
.tag { padding: 0 6px; border-radius: 9px; border: 1px solid currentColor; font-size: 11px; }
.tag.test { color: var(--test); }
.tag.ok { color: var(--ok); background: var(--ok-bg); border-color: transparent; }
.hintline { padding: 2px 10px 0; font-size: 12px; color: var(--warn); min-height: 20px; }
.hintline .gt { color: var(--muted); }
body.blind .hintline { visibility: hidden; }
body.blind .accept { display: none; }
.letters { padding: 8px 10px 10px; display: grid; grid-template-columns: repeat(3, 1fr); gap: 4px; margin-top: auto; }
.letters button { padding: 3px 2px; font-size: 12px; line-height: 1.25; text-align: center; }
.letters button b { display: block; font-size: 14px; }
.letters button[aria-pressed="true"] { background: var(--ok); color: var(--bg); border-color: var(--ok); }
.letters button.held { border-style: dashed; }
.accept { margin: 0 10px 8px; }
.empty { grid-column: 1 / -1; color: var(--muted); padding: 40px 0; text-align: center; }
#lb { position: fixed; inset: 0; background: #000d; z-index: 20; display: none; flex-direction: column;
  align-items: center; justify-content: center; padding: 12px; gap: 8px; }
#lb.on { display: flex; }
#lb img { max-width: 96vw; max-height: 66vh; object-fit: contain; background: #111; }
#lb .info { color: #fff; font-size: 14px; text-align: center; max-width: 96vw; }
#lb .letters { width: min(720px, 96vw); grid-template-columns: repeat(7, 1fr); background: var(--panel); border-radius: 8px; }
#toast { position: fixed; left: 50%; bottom: 18px; transform: translateX(-50%); background: var(--ink); color: var(--bg);
  padding: 8px 14px; border-radius: 8px; display: none; z-index: 30; max-width: 92vw; }
@media (max-width: 520px) { main { grid-template-columns: 1fr 1fr; padding: 10px; gap: 8px; } header { position: static; padding: 8px 10px; } }
</style>
</head>
<body>
<header>
  <h1>__TITLE__</h1>
  <span class="stat">已确认 <b id="nDone">0</b> / <b id="nAll">0</b></span>
  <span class="stat" id="testStat" hidden>测试集 <b id="nTestDone">0</b> / <b id="nTest">0</b></span>
  <div class="bar" aria-hidden="true"><i id="barFill"></i></div>
  <select id="filter" aria-label="筛选">
    <option value="all">全部</option>
    <option value="todo">只看未确认</option>
    <option value="done">只看已确认</option>
    <option value="test">只看测试集</option>
    <option value="test_todo">测试集里未确认的</option>
  </select>
  <select id="letterFilter" aria-label="按已选字母筛选"><option value="">任意字母</option></select>
  <label class="chk"><input type="checkbox" id="blind"> 盲审（隐藏提示）</label>
  <label class="chk"><input type="checkbox" id="auto" checked> 确认后跳到下一张</label>
  <button id="btnExport" class="primary">下载 review.json</button>
  <button id="btnCopy">复制 JSON</button>
  <button id="btnImport">导入 review.json</button>
  <input type="file" id="fileImport" accept=".json,application/json" hidden>
  <button id="btnClear">清空全部</button>
</header>
<details class="ref">
  <summary>字母表与快捷键（点开查看）</summary>
  <table id="refTable"></table>
  <p class="stat">快捷键：A–L / Z 选择字母；← → 切换照片；Enter 或空格放大；y 采纳“规则把握较大”的提示；Backspace 清除当前照片的选择；Esc 关闭放大图。<br>
  虚线边框的字母是留出实验（G–L）。照片不像表里任何一个实验时选 Z。只有你点过的照片才会进 review.json。</p>
</details>
<main id="grid"></main>
<div id="lb" role="dialog" aria-modal="true" aria-label="放大查看">
  <img id="lbImg" alt="">
  <div class="info" id="lbInfo"></div>
  <div class="letters" id="lbLetters"></div>
  <button id="lbClose">关闭（Esc）</button>
</div>
<div id="toast" role="status"></div>
<script type="application/json" id="payload">__DATA__</script>
<script>
/*LOGIC-BEGIN*/
// 纯逻辑，不碰 DOM
function isDone(labels, id) { return Object.prototype.hasOwnProperty.call(labels, id); }
function visibleIds(photos, labels, mode, letter) {
  var out = [];
  for (var i = 0; i < photos.length; i++) {
    var p = photos[i], d = isDone(labels, p.id);
    var ok = mode === "all" || (mode === "todo" && !d) || (mode === "done" && d) ||
             (mode === "test" && p.test) || (mode === "test_todo" && p.test && !d);
    if (ok && letter) ok = labels[p.id] === letter;
    if (ok) out.push(p.id);
  }
  return out;
}
// 当前之后的下一张未确认（到头回绕）；全部确认返回 null
function nextTodo(order, labels, curId) {
  var n = order.length, start = order.indexOf(curId);
  for (var k = 1; k <= n; k++) {
    var id = order[(start + k + n) % n];
    if (!isDone(labels, id)) return id;
  }
  return null;
}
// 导出：只含学生确认过的、且在照片集里的 id；键排序，格式 {id: letter}
function buildExport(labels, validIds) {
  var ok = {}; validIds.forEach(function (i) { ok[i] = 1; });
  var out = {};
  Object.keys(labels).sort().forEach(function (id) { if (ok[id]) out[id] = labels[id]; });
  return out;
}
// 导入：未知 id、非法字母分别记录，不抛异常
function mergeImport(obj, validIds, letters) {
  var okId = {}, okLetter = {};
  validIds.forEach(function (i) { okId[i] = 1; });
  letters.forEach(function (l) { okLetter[l] = 1; });
  var res = { labels: {}, unknown: [], bad: [] };
  if (obj === null || typeof obj !== "object" || Array.isArray(obj)) { res.bad.push("<根不是对象>"); return res; }
  Object.keys(obj).forEach(function (id) {
    if (!okId[id]) res.unknown.push(id);
    else if (!okLetter[obj[id]]) res.bad.push(id);
    else res.labels[id] = obj[id];
  });
  return res;
}
/*LOGIC-END*/

(function () {
  var D = JSON.parse(document.getElementById("payload").textContent);
  var photos = D.photos, LET = D.letters, LETTER_IDS = LET.map(function (l) { return l.letter; });
  var ids = photos.map(function (p) { return p.id; });
  var byId = {}; photos.forEach(function (p) { byId[p.id] = p; });
  var KEY = "ld-review-" + D.gid, KEY_UI = "ld-review-ui";
  var labels = {}, order = ids.slice(), cur = ids[0] || null, lbOn = false;

  function $(s) { return document.getElementById(s); }
  function lsGet(k) { try { return window.localStorage.getItem(k); } catch (e) { return null; } }
  function lsSet(k, v) { try { window.localStorage.setItem(k, v); } catch (e) { /* 存储不可用时照常工作 */ } }
  function toast(msg) {
    var t = $("toast"); t.textContent = msg; t.style.display = "block";
    clearTimeout(toast.h); toast.h = setTimeout(function () { t.style.display = "none"; }, 3500);
  }
  function save() { lsSet(KEY, JSON.stringify(labels)); }

  // 字母按钮（卡片与放大图共用）
  function letterButtons(getId) {
    var frag = document.createDocumentFragment();
    LET.forEach(function (l) {
      var b = document.createElement("button");
      b.type = "button"; b.dataset.letter = l.letter; b.title = l.name;
      b.className = l.held_out ? "held" : "";
      b.innerHTML = "<b>" + l.letter + "</b>" + l.short;
      b.addEventListener("click", function (ev) {
        ev.stopPropagation(); var id = getId(); if (id) setLabel(id, l.letter, true);
      });
      frag.appendChild(b);
    });
    return frag;
  }

  function buildCards() {
    var grid = $("grid"); grid.textContent = "";
    photos.forEach(function (p) {
      var c = document.createElement("div"); c.className = "card"; c.id = "c-" + p.id; c.dataset.id = p.id;
      var img = document.createElement("img"); img.className = "pic"; img.loading = "lazy";
      img.src = p.thumb; img.alt = p.id; img.addEventListener("click", function () { cur = p.id; openLb(p.id); });
      c.appendChild(img);
      var m = document.createElement("div"); m.className = "meta";
      m.innerHTML = '<span class="id"></span>' + (p.test ? '<span class="tag test">测试集</span>' : "") +
        (p.session != null ? '<span>时段 ' + p.session + "</span>" : "") + '<span class="tag ok" hidden>已确认</span>';
      m.querySelector(".id").textContent = p.id.replace(/^sc-/, "");
      c.appendChild(m);
      var h = document.createElement("div"); h.className = "hintline";
      h.textContent = p.hint.text;
      if (p.boxes) { var g = document.createElement("span"); g.className = "gt"; g.textContent = "　标注：" + p.boxes; h.appendChild(g); }
      c.appendChild(h);
      if (p.hint.kind === "confident") {
        var a = document.createElement("button"); a.type = "button"; a.className = "accept";
        a.textContent = "采纳提示 " + p.hint.letters[0] + "（y）";
        a.addEventListener("click", function (ev) { ev.stopPropagation(); setLabel(p.id, p.hint.letters[0], true); });
        c.appendChild(a);
      }
      var lw = document.createElement("div"); lw.className = "letters"; lw.appendChild(letterButtons(function () { return p.id; }));
      c.appendChild(lw);
      c.addEventListener("click", function () { setCur(p.id, false); });
      grid.appendChild(c);
    });
    var e = document.createElement("div"); e.className = "empty"; e.id = "empty"; e.textContent = "当前筛选下没有照片"; e.hidden = true;
    grid.appendChild(e);
  }

  function paintCard(id) {
    var c = $("c-" + id), l = labels[id];
    c.classList.toggle("done", isDone(labels, id));
    c.querySelector(".tag.ok").hidden = !isDone(labels, id);
    c.querySelectorAll(".letters button").forEach(function (b) { b.setAttribute("aria-pressed", String(b.dataset.letter === l)); });
  }

  function refresh() {
    var mode = $("filter").value, lf = $("letterFilter").value;
    order = visibleIds(photos, labels, mode, lf);
    var show = {}; order.forEach(function (i) { show[i] = 1; });
    ids.forEach(function (i) { $("c-" + i).classList.toggle("hidden", !show[i]); });
    $("empty").hidden = order.length > 0;
    var done = ids.filter(function (i) { return isDone(labels, i); }).length;
    var tests = photos.filter(function (p) { return p.test; });
    $("nDone").textContent = done; $("nAll").textContent = ids.length;
    $("barFill").style.width = (ids.length ? 100 * done / ids.length : 0) + "%";
    $("testStat").hidden = tests.length === 0;
    $("nTest").textContent = tests.length;
    $("nTestDone").textContent = tests.filter(function (p) { return isDone(labels, p.id); }).length;
    if (cur && !show[cur] && order.length) cur = order[0];
    markCur();
  }

  function markCur() {
    document.querySelectorAll(".card.cur").forEach(function (c) { c.classList.remove("cur"); });
    if (cur && $("c-" + cur)) $("c-" + cur).classList.add("cur");
  }
  function setCur(id, scroll) {
    cur = id; markCur();
    if (scroll && $("c-" + id)) $("c-" + id).scrollIntoView({ block: "nearest", behavior: "smooth" });
    if (lbOn) fillLb(id);
  }

  function setLabel(id, letter, advance) {
    if (letter === null) delete labels[id]; else labels[id] = letter;
    paintCard(id); save();
    var nxt = null;
    if (letter !== null && advance && $("auto").checked) nxt = nextTodo(order, labels, id);
    refresh();
    if (lbOn) fillLb(id);
    if (nxt) setCur(nxt, true);
    else if (letter !== null && advance && order.indexOf(id) < 0 && order.length) setCur(order[0], true);
  }

  function step(delta) {
    if (!order.length) return;
    var i = order.indexOf(cur); if (i < 0) i = 0;
    setCur(order[(i + delta + order.length) % order.length], true);
  }

  // 放大图
  function fillLb(id) {
    var p = byId[id], img = $("lbImg");
    img.onerror = function () { img.onerror = null; img.src = p.thumb; };
    img.src = p.file; img.alt = id;
    var l = labels[id], nm = l ? LET.filter(function (x) { return x.letter === l; })[0] : null;
    var blind = document.body.classList.contains("blind");
    $("lbInfo").textContent = id + (nm ? "　已选：" + l + " " + nm.short : "　（未确认）") + (blind ? "" : "　" + p.hint.text);
    $("lbLetters").querySelectorAll("button").forEach(function (b) { b.setAttribute("aria-pressed", String(b.dataset.letter === l)); });
  }
  function openLb(id) { lbOn = true; $("lb").classList.add("on"); fillLb(id); }
  function closeLb() { lbOn = false; $("lb").classList.remove("on"); }

  // 导出 / 导入
  function exportText() { return JSON.stringify(buildExport(labels, ids), null, 2) + "\n"; }
  function doExport() {
    var n = Object.keys(buildExport(labels, ids)).length;
    if (!n) { toast("还没有确认任何照片"); return; }
    var blob = new Blob([exportText()], { type: "application/json" });
    var a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "review.json";
    document.body.appendChild(a); a.click(); document.body.removeChild(a);
    setTimeout(function () { URL.revokeObjectURL(a.href); }, 2000);
    toast("已导出 " + n + " 张（文件一般在“下载”文件夹）");
  }
  function doCopy() {
    var t = exportText();
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(t).then(function () { toast("已复制到剪贴板"); },
        function () { toast("浏览器不允许复制，请用“下载”"); });
    } else toast("浏览器不支持复制，请用“下载”");
  }
  function doImport(file) {
    var r = new FileReader();
    r.onload = function () {
      var obj; try { obj = JSON.parse(r.result); } catch (e) { toast("不是合法的 JSON"); return; }
      var m = mergeImport(obj, ids, LETTER_IDS), n = Object.keys(m.labels).length;
      Object.keys(m.labels).forEach(function (id) { labels[id] = m.labels[id]; paintCard(id); });
      save(); refresh();
      toast("导入 " + n + " 张" + (m.unknown.length ? "；忽略不在本图册的 id " + m.unknown.length + " 个" : "") +
            (m.bad.length ? "；忽略非法字母 " + m.bad.length + " 个" : ""));
    };
    r.readAsText(file);
  }

  // 初始化
  function init() {
    var ref = $("refTable");
    LET.forEach(function (l) {
      var tr = document.createElement("tr");
      tr.innerHTML = "<td></td><td></td><td></td>";
      tr.children[0].textContent = l.letter;
      tr.children[1].textContent = l.name + (l.held_out ? "（留出）" : "");
      tr.children[2].textContent = l.look || "不像上面任何一个实验，或者看不清";
      ref.appendChild(tr);
    });
    var lf = $("letterFilter");
    LET.forEach(function (l) { var o = document.createElement("option"); o.value = l.letter; o.textContent = "已选 " + l.letter + " " + l.short; lf.appendChild(o); });
    $("lbLetters").appendChild(letterButtons(function () { return cur; }));
    buildCards();
    try {
      var raw = lsGet(KEY);
      if (raw) { var m = mergeImport(JSON.parse(raw), ids, LETTER_IDS); labels = m.labels; }
      var ui = JSON.parse(lsGet(KEY_UI) || "{}");
      if (ui.blind) { $("blind").checked = true; document.body.classList.add("blind"); }
      if (ui.auto === false) $("auto").checked = false;
    } catch (e) { labels = {}; }
    ids.forEach(paintCard);
    var first = photos.filter(function (p) { return p.test && !isDone(labels, p.id); })[0];
    if (first) $("filter").value = "test_todo";
    refresh();
    if (!cur || !byId[cur]) cur = order[0] || null;
    markCur();
  }

  $("filter").addEventListener("change", refresh);
  $("letterFilter").addEventListener("change", refresh);
  $("blind").addEventListener("change", function () {
    document.body.classList.toggle("blind", this.checked);
    if (lbOn && cur) fillLb(cur);
    lsSet(KEY_UI, JSON.stringify({ blind: this.checked, auto: $("auto").checked }));
  });
  $("auto").addEventListener("change", function () {
    lsSet(KEY_UI, JSON.stringify({ blind: $("blind").checked, auto: this.checked }));
  });
  $("btnExport").addEventListener("click", doExport);
  $("btnCopy").addEventListener("click", doCopy);
  $("btnImport").addEventListener("click", function () { $("fileImport").click(); });
  $("fileImport").addEventListener("change", function () { if (this.files[0]) doImport(this.files[0]); this.value = ""; });
  $("btnClear").addEventListener("click", function () {
    if (!Object.keys(labels).length) return;
    if (window.confirm("清空所有已确认的标签？（建议先下载 review.json 备份）")) { labels = {}; save(); ids.forEach(paintCard); refresh(); }
  });
  $("lbClose").addEventListener("click", closeLb);
  $("lb").addEventListener("click", function (ev) { if (ev.target === this) closeLb(); });

  document.addEventListener("keydown", function (ev) {
    var t = ev.target && ev.target.tagName;
    if (t === "INPUT" || t === "SELECT" || t === "TEXTAREA" || ev.ctrlKey || ev.metaKey || ev.altKey) return;
    var k = ev.key;
    if (k === "Escape") { closeLb(); return; }
    if (k === "ArrowRight") { ev.preventDefault(); step(1); return; }
    if (k === "ArrowLeft") { ev.preventDefault(); step(-1); return; }
    if ((k === "Enter" || k === " ") && t !== "BUTTON") { ev.preventDefault(); if (lbOn) closeLb(); else if (cur) openLb(cur); return; }
    if (!cur) return;
    if (k === "Backspace" || k === "Delete") { ev.preventDefault(); setLabel(cur, null, false); return; }
    if ((k === "y" || k === "Y") && byId[cur].hint.kind === "confident" && !document.body.classList.contains("blind")) { setLabel(cur, byId[cur].hint.letters[0], true); return; }
    var up = k.length === 1 ? k.toUpperCase() : "";
    if (up && LETTER_IDS.indexOf(up) >= 0) { setLabel(cur, up, true); }
  });

  init();
})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    sys.exit(main())
