"""密封卡：可打印的离线 HTML（不引用任何外部文件），每张卡有剪切线和对折线。

外页只有运行编号、代码、实验名和实验课次；内页是成年人照做的动作（每张卡都有动作，没有故障的那几张是不改变装置的假动作）、
需要时交给学生的一句话、安全提示。卡片上不出现条件名、原因名或任何能直接看出条件的词。
"""

from __future__ import annotations

import html

STYLE = """
@page { size: A4; margin: 12mm; }
body { font-family: "PingFang SC", "Noto Sans CJK SC", "Microsoft YaHei", sans-serif; color: #111; background: #fff;
       margin: 0; font-size: 12pt; line-height: 1.5; }
.cover { padding: 8mm 4mm; page-break-after: always; }
.cover h1 { font-size: 18pt; margin: 0 0 4mm; }
.cover li { margin: 1.5mm 0; }
.card { border: 1.5px dashed #444; margin: 0 0 8mm; padding: 0; page-break-inside: avoid; break-inside: avoid;
        position: relative; }
.card::before { content: "✂"; position: absolute; top: -11px; left: -4px; font-size: 14pt; background: #fff; }
.outer, .inner { padding: 5mm 6mm; }
.outer { min-height: 34mm; }
.code { font-size: 26pt; font-weight: 700; letter-spacing: 3px; font-family: "Menlo", "Consolas", monospace; }
.fold { border-top: 1px dotted #888; text-align: center; font-size: 9pt; color: #555; padding: 1mm 0; }
.inner h3 { margin: 0 0 2mm; font-size: 12pt; }
.inner ol { margin: 0 0 3mm 5mm; padding-left: 4mm; }
.say { border: 1px solid #999; padding: 2mm 3mm; margin: 2mm 0; min-height: 10mm; }
.small { font-size: 9.5pt; color: #333; }
.tick { margin-top: 3mm; }
@media print { .card:nth-of-type(2n) { page-break-after: always; } }
"""

COVER = """
<section class="cover">
<h1>成年人须知（真实实验的密封卡）</h1>
<ol>
<li>你在这次实验里只做两件事：负责安全监督；按密封卡做一项准备动作。所有测量和记录都由学生本人完成。</li>
<li>每次运行开始前，学生把写着本次代码的信封交给你。请避开学生拆开，照内页的步骤做；写着“学生读完原长后”
“学生平衡好摩擦力之后”这类时机的步骤，到那时再做（请学生先转过身或离开桌子一会儿）。</li>
<li>每张卡都有动作。有的动作会改变装置，有的不会，请一律照做，动作的快慢、时长尽量一样。</li>
<li>内页“交给学生的话”一栏有内容时，只把那一句念给（或给）学生看；没有内容就什么也不说。</li>
<li>不要告诉学生你做了什么，也不要做卡片以外的任何调整。学生问起时，只回答“照常做就好”。</li>
<li>任何时候觉得不安全就立刻停止：砝码、钢球、铁架台要放稳，人不站在摆动或滑动的路线上。</li>
<li>不记录任何关于人的信息：不写名字，不拍照，不录像。卡片上的勾选框只表示动作做完了。</li>
<li>每次运行结束后，把卡片放回原信封封好，由你保管；等学生把数据冻结（生成哈希清单）后再全部交还。</li>
</ol>
<p class="small">剪切：沿虚线剪下每张卡；沿点线对折，外页朝外装进信封，信封上抄写外页的代码。</p>
</section>
"""


def _esc(s) -> str:
    return html.escape(str(s), quote=True)


def render_cards(public: dict, sealed: dict) -> str:
    runs = {r["code"]: r for r in public["runs"]}
    parts = ["<!doctype html>", '<html lang="zh-CN"><head><meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width, initial-scale=1">',
             "<title>密封卡</title>", f"<style>{STYLE}</style></head><body>", COVER]
    for k in sealed["key"]:
        run = runs[k["code"]]
        name = public["experiments"][run["experiment"]]["name_zh"]
        steps = "".join(f"<li>{_esc(s)}</li>" for s in k["card"]["steps"])
        say = _esc(k["card"]["student_text"]) if k["card"]["student_text"] else "（本次没有）"
        safety = _esc(k["card"]["safety"]) if k["card"]["safety"] else "照常注意安全。"
        parts.append(
            '<div class="card">'
            '<div class="outer">'
            f'<div class="small">运行 {_esc(run["run_id"])} · 第 {_esc(run["session"])} 次实验课</div>'
            f'<div class="code">{_esc(k["code"])}</div>'
            f"<div>{_esc(name)}</div>"
            '<div class="small">由在场成年人在本次运行开始前拆开，学生不要看内页。</div>'
            "</div>"
            '<div class="fold">沿此线对折（内页朝里）</div>'
            '<div class="inner">'
            f"<h3>代码 {_esc(k['code'])}：请你照做</h3>"
            f"<ol>{steps}</ol>"
            f'<div>交给学生的话：</div><div class="say">{say}</div>'
            f'<div class="small">安全：{safety}</div>'
            '<div class="tick">□ 动作已做完　□ 卡片已放回信封</div>'
            "</div></div>"
        )
    parts.append("</body></html>")
    return "\n".join(parts)
