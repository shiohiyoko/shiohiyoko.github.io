---
title: 大江戸ロボット忍法帖
layout: single
classes: wide
permalink: /portfolio/ooedo/

contest_rule:
  - image_path: /assets/images/portfolio/ooedo/rule_pic.jpg
    alt: "palceholder image 1"
    title: "競技内容"
    excerpt: "相手の本陣についた風船を全部割るか，相手についた風船をより多く割った方の勝利"

creation:
  - image_path: /assets/images/portfolio/ooedo/mekaron.JPG
    alt: "placeholder image 2"
    title: "作成したロボット"
    excerpt: "私が担当した機体はこちら．自由度が２つある腕を２本動かしながら，先端についたひみつ道具を使い，風船を割ります．"

weapon:
  - image_path:  /assets/images/portfolio/ooedo/spikes.JPG
    alt: "placehoder image 3"
    title: "ひみつ道具"
    excerpt: "ルール上，ひみつ道具はヤスリを使う必要がありましが，使えるヤスリの量に指定がありました．なので，使うヤスリは少量で風船を割りやすいようなひみつ道具を作る必要がありました．そこで私はモーニングスターをイメージした腕の先端につける部品を作成しました．"

controller:
  - image_path: /assets/images/portfolio/ooedo/left_mouse.jpg
    alt: "placehoder image 4"
    title: "コントローラ"
    excerpt: "移動しながら２つの腕を動かすのはps3のコントローラーでは無理がありました．そこで，私達はマウスの横にジョイスティックを埋め込んだマウスとボタンを埋め込んだマウスを自作し，それらを使いました．"

three_omni:
  - image_path: /assets/images/portfolio/ooedo/three_omni.png
    alt: "placehoder image 5"
    title: "３輪オムニホイール"
    excerpt: "私が担当したロボットの一つの走行系はオムニホイールを３つ使用したものです．これにより，全方位移動できるようになり，いろんな方向から攻撃ができるようになりました．"
double_wheel:
  - image_path:  /assets/images/portfolio/ooedo/double_wheel.png
    alt: "placehoder image 6"
    title: "2輪駆動"
    excerpt: "私が担当したもうひとつの機体の走行系は，ラジコンに使われるタイヤを走行系に取り付け，２輪駆動にしたことでスピードが上がり，相手の本陣に早くたどり着くことが可能になりました．"


---


{% include  feature_row id="contest_rule" type = "left" %}
{% include  feature_row id="creation" type = "right" %}
{% include  feature_row id="weapon" type = "left" %}
{% include  feature_row %}
{% include  feature_row id="controller" type = "left" %}
{% include  feature_row id="double_wheel" type = "right" %}
{% include  feature_row id="three_omni" type = "center" %}
