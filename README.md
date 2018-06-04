# retrival-chatbot
a simple retrieval-based chatbot.

# 环境
- pytorch: 0.4.0
- numpy: 1.14.0
- jieba: 0.39

# 数据集
- data 文件夹下包含训练测试所用数据（已预处理）
- process_data.ipynb,vocab_build.ipynb为预处理数据代码。
# 训练
```
python cli.py data save
# data is data dir, save is save dir.
```
# 问答
```
python inference.py "小米空气净化器2效果如何"
# 输出: 您好，小米空气净化器2颗粒物CADR高达310m^3/h
```
