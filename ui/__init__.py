"""实验侦探的 Streamlit 界面（app.py 是入口）。

模块分两类：
  纯逻辑（不 import streamlit，可单独测试）：catalog（实验目录、"其他实验"组合）、lab（数据表、实时对照、
  模拟数据、事实单/幽灵曲线封装、知识点）、llm（本地模型服务探测）、charts（Altair 图）。
  界面（每一步一个文件）：step_identify ① 拍照识别、step_design ② 实验设计、step_measure ③ 测量、
  step_diagnose ④ 诊断、step_explain ⑤ 讲解与知识点；theme（黑板主题，系统字体，完全离线）、state（会话状态）。
所有数字都来自 lab_detective 的物理代码；语言模型只在"模型模式"下写讲解、帮忙认图，从不算物理。
"""
