"""
生成一个轻量的静态网页：缩略图网格浏览 data/self_captured 下每张图的标注框，
支持点开单张图完全清空旧框、手动拖框重画、选类别，然后导出 YOLO 格式 .txt。

**重要限制**：这是纯静态 HTML（Artifact 沙盒页面），没有服务器，不能直接写回本地
文件系统。"导出"是浏览器下载一个和图片同名的 .txt，你需要手动把下载的文件拖到
data/self_captured/labels/ 里覆盖旧的。适合"这张图自动标注完全不能用，整张重画"
的场景；只是小修小补（改个别框的类别/边界）用 labelImg/Roboflow 效率更高。

用法：
    python gen_label_gallery.py
    再用浏览器直接打开 output/label_gallery.html（或让 Claude 发布成 Artifact）。

autolabel.py 还在跑的时候可以随时重新执行这个脚本刷新进度。
"""

import base64
import json
import cv2
from pathlib import Path

PROJ = Path(__file__).parent
IMG_DIR = PROJ / "data" / "self_captured" / "images"
LBL_DIR = PROJ / "data" / "self_captured" / "labels"
OUT_HTML = PROJ / "output" / "label_gallery.html"

NAMES = {0: "cart", 1: "track", 2: "spring", 3: "string", 4: "ruler", 5: "dynamometer"}
COLORS_HEX = {0: "#ff8000", 1: "#808080", 2: "#0078ff",
              3: "#c800c8", 4: "#00c8c8", 5: "#0000ff"}
IMG_MAX = 620  # 兼顾编辑精度和文件体积；YOLO坐标是归一化的，缩放不影响导出精度
                # （Artifact 单文件上限16MB，266张图要控制在这个尺寸内）
JPEG_QUALITY = 65


def render_image_and_boxes(img_path: Path, lbl_path: Path):
    """
    返回 (base64 jpg, 缩放后宽高, 已有框列表[[cid,cx,cy,bw,bh],...])。

    注意：这里不把框烧进图片像素——框数据是归一化坐标，网格缩略图用 CSS 绝对定位
    叠一层 div 画框（跟 JS 编辑器共用同一份坐标数据），编辑器里用 canvas 画。
    只嵌入一份干净图片，避免同一张图存两份 base64 把文件体积翻倍。
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return "", 0, 0, []
    h, w = img.shape[:2]
    boxes = []
    if lbl_path.exists() and lbl_path.stat().st_size:
        for line in lbl_path.read_text().strip().split("\n"):
            if not line.strip():
                continue
            parts = line.split()
            cid = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:5])
            boxes.append([cid, cx, cy, bw, bh])

    scale = IMG_MAX / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    img = cv2.resize(img, (new_w, new_h))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return b64, new_w, new_h, boxes


def main():
    img_paths = sorted(IMG_DIR.glob("*.jpg"))
    cards = []
    images_data = []
    n_labeled = n_empty = n_missing = 0

    for idx, img_path in enumerate(img_paths):
        lbl_path = LBL_DIR / f"{img_path.stem}.txt"
        b64, w, h, boxes = render_image_and_boxes(img_path, lbl_path)
        n_boxes = len(boxes)
        if not lbl_path.exists():
            status = "missing"; n_missing += 1
        elif n_boxes == 0:
            status = "empty"; n_empty += 1
        else:
            status = "ok"; n_labeled += 1

        images_data.append({
            "stem": img_path.stem, "b64": b64, "w": w, "h": h, "boxes": boxes,
        })
        overlay_boxes = "".join(
            f'<div class="obox" style="left:{(cx - bw / 2) * 100:.2f}%;'
            f'top:{(cy - bh / 2) * 100:.2f}%;width:{bw * 100:.2f}%;height:{bh * 100:.2f}%;'
            f'border-color:{COLORS_HEX.get(cid, "#fff")}"></div>'
            for cid, cx, cy, bw, bh in boxes
        )
        cards.append(f"""
        <div class="card status-{status}" onclick="openModal({idx})">
          <div class="thumb-wrap">
            <img src="data:image/jpeg;base64,{b64}" loading="lazy" />
            {overlay_boxes}
          </div>
          <div class="meta">
            <span class="fname">{img_path.stem}</span>
            <span class="badge badge-{status}">{n_boxes} 框</span>
          </div>
        </div>""")

    class_options = "".join(
        f'<option value="{cid}">{name}</option>' for cid, name in NAMES.items()
    )
    class_colors_js = json.dumps(COLORS_HEX)
    class_names_js = json.dumps(NAMES)
    images_json = json.dumps(images_data)

    html = f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>标注框预览与重标</title>
<style>
:root {{ color-scheme: light dark; }}
body {{ font-family: -apple-system, "PingFang SC", sans-serif; margin: 0; padding: 16px;
        background: #fff; color: #111; }}
@media (prefers-color-scheme: dark) {{ body {{ background: #15161a; color: #eee; }} }}
h1 {{ font-size: 18px; margin: 0 0 4px; }}
.stats {{ font-size: 13px; opacity: .75; margin-bottom: 14px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 10px; }}
.card {{ border-radius: 8px; overflow: hidden; border: 1px solid rgba(128,128,128,.25); cursor: pointer; }}
.card:hover {{ outline: 2px solid #4a90d9; }}
.thumb-wrap {{ position: relative; }}
.card img {{ width: 100%; display: block; }}
.obox {{ position: absolute; border: 2px solid; pointer-events: none; }}
.meta {{ display: flex; justify-content: space-between; align-items: center;
         padding: 5px 8px; font-size: 11px; }}
.fname {{ opacity: .7; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.badge {{ padding: 1px 6px; border-radius: 10px; font-weight: 600; flex-shrink: 0; margin-left: 6px; }}
.badge-ok {{ background: #d7f5dd; color: #1a7a34; }}
.badge-empty {{ background: #f5e4d7; color: #a35a1a; }}
.badge-missing {{ background: #f5d7d7; color: #a31a1a; }}
@media (prefers-color-scheme: dark) {{
  .badge-ok {{ background: #123b1e; color: #7be89b; }}
  .badge-empty {{ background: #3b2812; color: #e8b47b; }}
  .badge-missing {{ background: #3b1212; color: #e87b7b; }}
}}
.filter {{ margin-bottom: 10px; }}
.filter label {{ margin-right: 12px; font-size: 13px; cursor: pointer; }}

.modal {{ position: fixed; inset: 0; background: rgba(0,0,0,.75); z-index: 50;
          display: flex; align-items: center; justify-content: center; }}
.modal.hidden {{ display: none; }}
.modal-box {{ background: #fff; color: #111; border-radius: 10px; padding: 14px;
              max-width: 96vw; max-height: 96vh; overflow: auto; }}
@media (prefers-color-scheme: dark) {{ .modal-box {{ background: #202127; color: #eee; }} }}
.modal-toolbar {{ display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin-bottom: 8px; }}
.modal-toolbar button, .modal-toolbar select {{
  font-size: 13px; padding: 5px 10px; border-radius: 6px; border: 1px solid rgba(128,128,128,.4);
  background: transparent; color: inherit; cursor: pointer;
}}
.canvas-wrap {{ position: relative; line-height: 0; }}
.canvas-wrap canvas {{ max-width: 90vw; max-height: 70vh; cursor: crosshair; }}
#box-list {{ margin-top: 8px; font-size: 12px; max-height: 120px; overflow-y: auto; }}
#box-list .box-row {{ display: flex; align-items: center; gap: 6px; padding: 2px 0; }}
#box-list select {{ font-size: 12px; }}
#box-list button {{ font-size: 11px; cursor: pointer; }}
.hint {{ font-size: 11px; opacity: .65; margin-top: 4px; }}
</style></head>
<body>
<h1>data/self_captured 标注框预览与重标</h1>
<div class="stats">共 {len(img_paths)} 张 &middot; 已标注 {n_labeled} &middot; 0框 {n_empty} &middot; 缺标注文件 {n_missing}
&middot; 点击任意缩略图可清空重画 &middot; 导出的 .txt 需手动放入 data/self_captured/labels/</div>
<div class="filter">
  <label><input type="checkbox" checked onchange="toggle('ok', this.checked)"> 有框({n_labeled})</label>
  <label><input type="checkbox" checked onchange="toggle('empty', this.checked)"> 0框({n_empty})</label>
  <label><input type="checkbox" checked onchange="toggle('missing', this.checked)"> 未处理({n_missing})</label>
</div>
<div class="grid" id="grid">
{''.join(cards)}
</div>

<div class="modal hidden" id="modal">
  <div class="modal-box">
    <div class="modal-toolbar">
      <strong id="modal-title"></strong>
      <select id="class-select">{class_options}</select>
      <button onclick="clearBoxes()">清空所有框</button>
      <button onclick="undoBox()">撤销上一个</button>
      <button onclick="exportTxt()">导出 .txt</button>
      <span style="flex:1"></span>
      <button onclick="navImage(-1)">← 上一张</button>
      <button onclick="navImage(1)">下一张 →</button>
      <button onclick="closeModal()">关闭 ✕</button>
    </div>
    <div class="canvas-wrap">
      <canvas id="canvas"></canvas>
    </div>
    <div id="box-list"></div>
    <div class="hint">拖拽画框；框的类别取自左上角下拉框（画完后也能在下面列表里改）；切换图片前记得导出，否则修改会丢失。</div>
  </div>
</div>

<script>
const IMAGES = {images_json};
const CLASS_NAMES = {class_names_js};
const CLASS_COLORS = {class_colors_js};

let curIdx = -1;
let curBoxes = [];   // [{{cid, cx, cy, bw, bh}}] 归一化坐标
let drawing = false;
let startX = 0, startY = 0;

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const img = new Image();

function toggle(status, show) {{
  document.querySelectorAll('.status-' + status).forEach(el => {{
    el.style.display = show ? '' : 'none';
  }});
}}

function openModal(idx) {{
  curIdx = idx;
  const d = IMAGES[idx];
  curBoxes = d.boxes.map(b => ({{cid: b[0], cx: b[1], cy: b[2], bw: b[3], bh: b[4]}}));
  document.getElementById('modal-title').textContent = d.stem;
  img.onload = () => {{
    canvas.width = d.w;
    canvas.height = d.h;
    redraw();
  }};
  img.src = 'data:image/jpeg;base64,' + d.b64;
  document.getElementById('modal').classList.remove('hidden');
}}

function closeModal() {{
  document.getElementById('modal').classList.add('hidden');
  curIdx = -1;
}}

function navImage(delta) {{
  if (curIdx < 0) return;
  const next = curIdx + delta;
  if (next < 0 || next >= IMAGES.length) return;
  openModal(next);
}}

function redraw() {{
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  curBoxes.forEach(b => drawBox(b));
  renderBoxList();
}}

function drawBox(b) {{
  const x = (b.cx - b.bw / 2) * canvas.width;
  const y = (b.cy - b.bh / 2) * canvas.height;
  const w = b.bw * canvas.width;
  const h = b.bh * canvas.height;
  ctx.strokeStyle = CLASS_COLORS[b.cid] || '#fff';
  ctx.lineWidth = 3;
  ctx.strokeRect(x, y, w, h);
  ctx.fillStyle = CLASS_COLORS[b.cid] || '#fff';
  ctx.font = '16px sans-serif';
  ctx.fillText(CLASS_NAMES[b.cid] || b.cid, x + 2, Math.max(14, y - 4));
}}

function getPos(e) {{
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  return {{ x: (e.clientX - rect.left) * scaleX, y: (e.clientY - rect.top) * scaleY }};
}}

canvas.addEventListener('mousedown', e => {{
  const p = getPos(e);
  startX = p.x; startY = p.y; drawing = true;
}});

canvas.addEventListener('mousemove', e => {{
  if (!drawing) return;
  const p = getPos(e);
  redraw();
  ctx.strokeStyle = '#ffff00';
  ctx.lineWidth = 2;
  ctx.setLineDash([5, 3]);
  ctx.strokeRect(Math.min(startX, p.x), Math.min(startY, p.y),
                 Math.abs(p.x - startX), Math.abs(p.y - startY));
  ctx.setLineDash([]);
}});

canvas.addEventListener('mouseup', e => {{
  if (!drawing) return;
  drawing = false;
  const p = getPos(e);
  const x1 = Math.min(startX, p.x), x2 = Math.max(startX, p.x);
  const y1 = Math.min(startY, p.y), y2 = Math.max(startY, p.y);
  if (x2 - x1 < 4 || y2 - y1 < 4) {{ redraw(); return; }}  // 太小的忽略，防止误触
  const cid = parseInt(document.getElementById('class-select').value);
  curBoxes.push({{
    cid,
    cx: (x1 + x2) / 2 / canvas.width,
    cy: (y1 + y2) / 2 / canvas.height,
    bw: (x2 - x1) / canvas.width,
    bh: (y2 - y1) / canvas.height,
  }});
  redraw();
}});

function clearBoxes() {{
  curBoxes = [];
  redraw();
}}

function undoBox() {{
  curBoxes.pop();
  redraw();
}}

function renderBoxList() {{
  const list = document.getElementById('box-list');
  list.innerHTML = curBoxes.map((b, i) => {{
    const opts = Object.entries(CLASS_NAMES).map(([cid, name]) =>
      `<option value="${{cid}}" ${{parseInt(cid) === b.cid ? 'selected' : ''}}>${{name}}</option>`
    ).join('');
    return `<div class="box-row">
      #${{i + 1}}
      <select onchange="changeBoxClass(${{i}}, this.value)">${{opts}}</select>
      <span style="opacity:.6">(${{b.cx.toFixed(2)}}, ${{b.cy.toFixed(2)}})</span>
      <button onclick="deleteBox(${{i}})">删除</button>
    </div>`;
  }}).join('') || '<div style="opacity:.5">（无框）</div>';
}}

function changeBoxClass(i, val) {{
  curBoxes[i].cid = parseInt(val);
  redraw();
}}

function deleteBox(i) {{
  curBoxes.splice(i, 1);
  redraw();
}}

function exportTxt() {{
  if (curIdx < 0) return;
  const stem = IMAGES[curIdx].stem;
  const lines = curBoxes.map(b =>
    `${{b.cid}} ${{b.cx.toFixed(6)}} ${{b.cy.toFixed(6)}} ${{b.bw.toFixed(6)}} ${{b.bh.toFixed(6)}}`
  ).join('\\n');
  const blob = new Blob([lines], {{type: 'text/plain'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = stem + '.txt';
  a.click();
  URL.revokeObjectURL(a.href);
}}

document.getElementById('modal').addEventListener('click', e => {{
  if (e.target.id === 'modal') closeModal();
}});
document.addEventListener('keydown', e => {{
  if (curIdx < 0) return;
  if (e.key === 'Escape') closeModal();
  if (e.key === 'ArrowLeft') navImage(-1);
  if (e.key === 'ArrowRight') navImage(1);
}});
</script>
</body></html>"""
    OUT_HTML.write_text(html)
    print(f"生成完成: {OUT_HTML}")
    print(f"共 {len(img_paths)} 张，已标注 {n_labeled}，0框 {n_empty}，未处理 {n_missing}")


if __name__ == "__main__":
    main()
