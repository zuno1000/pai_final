# PAI最終課題_言葉で指示すると学習した動作で応える挨拶ロボット

Physical AI 応用編1 最終課題の提出コードです。
Gazebo上のCRANE+V2に, 人間のテレオペデモから模倣学習した3つの動作（お辞儀 `bow` / 手を振る `wave` / うなずく `nod`）を覚えさせ, 文字列コマンドで切り替えて実行します。 さらに, 方策への観測入力を改ざんしたときの頑健性を検証します。

- 動画：https://youtu.be/oGRt4kKm6fg
- 環境：`airobotbook/ros2-desktop-ai-robot-book-humble` コンテナ（ROS 2 Humble + Gazebo）,   [matsuolab/pai_training](https://github.com/matsuolab/pai_training),   [matsuolab/physicalai_teleop](https://github.com/matsuolab/physicalai_teleop), CPU のみ

## 構成

```
nodes/
  gesture_dispatcher.py   /gesture_cmdを受けてACT方策を選択, 実行, 直立復帰するROS2ノード
  state_tamper.py         /joint_statesを改ざんして/joint_states_tamperedに再発行するノード
  joint3_monitor.py       動作実行中の関節ピーク値を記録する評価用ノード
configs/{bow,wave,nod}/   pai_training用の変換, 学習設定
```

## 再現手順（pai_trainingがセットアップ済みのコンテナ内）

1. データ収集：`ros2 launch crane_plus_gazebo crane_plus_with_table.launch.py`と`ros2 run physicalai_teleop teleop_keyboard`を起動し、動作ごとに`ros2 bag record -a -o ~/dataset/rosbags/<gesture>/epN`で10エピソードを記録
2. 変換・学習
   ```bash
   cp configs/wave/* ~/pai_training/configs/
   cd ~/pai_training && pixi run convert && WANDB_MODE=offline pixi run train
   ```
3. 実行
   ```bash
   cd ~/pai_training
   pixi run python /workspace/pai_final/nodes/gesture_dispatcher.py
   ros2 topic pub --once /gesture_cmd std_msgs/msg/String "{data: 'wave'}"
   ```
4. 改ざんテスト
   ```bash
   python3 nodes/state_tamper.py --mode bias --bias 0.2
   pixi run python nodes/gesture_dispatcher.py --state-topic /joint_states_tampered
   ```

学習済みモデル・データセットは容量の都合でリポジトリに含めていません。
