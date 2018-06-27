---
title: ロボット・ニューフロンティア
layout: single
classes: wide
permalink: /portfolio/newfront/



contest_rule:
  - image_path: /assets/images/portfolio/newfront/rule_pic.jpg
    alt: "palceholder image 1"
    title: "競技内容"
    excerpt: "港や途中の島に設置された箱を使い，新大陸でその箱をタワー型に積み上げて高さを競う競技です."

achievements:
  - excerpt: "組み込み系のプログラムの書き方,  
18v駆動モーターの制御,  
回路のデバッグ方法,  
シリアル通信,  
PID制御"

#担当箇所
wheels:
  - image_path:  /assets/images/portfolio/ooedo/wheel.png
    alt: "placehoder image 3"
    title: "走行系"
    excerpt: "このロボットにはメカな無ホイールという全方位移動できるタイヤを使用しました．モーターはブラシレスモータを用い，高速移動が可能になっています．"

jackup:
  - image_path: /assets/images/portfolio/ooedo/dsdd.png
    alt: "placehoder image 4"
    title: "ジャッキアップ"
    excerpt: "このロボットは機体がサイズ制限ギリギリに設計されたため，スタート時はタイヤをロボットの内側に折りたたむ機能を取り付けました．これは角度センサを取り付けて，９０度毎で保持するように制御しました．"
---


{% include  feature_row id="contest_rule" type = "left" %}
{% include  feature_row id="achievements" type = "right" %}
{% include  feature_row id="wheels" type = "left" %}
{% include  feature_row id="jackup" type = "left" %}
