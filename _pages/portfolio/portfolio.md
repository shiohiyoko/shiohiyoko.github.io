---
title: Portfolio
layout: splash
permalink: /portfolio/
collection: portfolio
entries_layout: grid
header:
  overlay_color: "#000"
  overlay_filter: "0.5"
  overlay_image: /assets/images/portfolio-header.jpg

intro:
  - excerpt: '鈴鹿工業高等専門学校 電子情報工学科'

feature_row1:
  - image_path: /assets/images/portfolio/ooedo/ooedo.png
    alt: "placeholder image 1"
    title: "大江戸ロボット忍法帖"
    excerpt: "2017年度に行われた高専ロボコンの競技"
    url: "/portfolio/ooedo/"
    btn_label: "Read More"
    btn_class: "btn--primary"
feature_row2:
  - image_path: /assets/images/portfolio/newfuro.jpg
    alt: "placeholder image 2"
    title: "ロボット・ニューフロンティア"
    excerpt: "2016年度に行われた高専ロボコンの競技"
    url: "/portfolio/newfront/"
    btn_label: "Read More"
    btn_class: "btn--primary"
feature_row3:
  - image_path: /assets/images/portfolio/landdisk.jpg
    alt: "placeholder image 3"
    title: "LANDING DISK"
    excerpt: "2016年度に行われた学生ロボコンの競技"
    url: "/portfolio/landdisk/"
    btn_label: "Read More"
    btn_class: "btn--primary"
---

{% include feature_row id="intro" type="center" %}

{% include feature_row id="feature_row1" type="left" %}

{% include feature_row id="feature_row2" type="right" %}

{% include feature_row id="feature_row3" type="left" %}
