# Video Comprehension and Retrieval

## Usage
1. 包含音频转文字，视频图像转文字，文字embeddding过程。

- `export OPENAI_API_KEY='your-api-key-here'`
- `python main.py -v video_path`

2. video_embedding.json内容存入elasticsearch
- `python create_database.py -d data_path`

3. 自然语言query
- `python query.py -q question`

4. model下载
包含sbert-chinese和vosk-cn模型
```
链接：https://rec.ustc.edu.cn/share/98f237b0-3112-11ef-8840-898423b2c5db
密码：rcdc
```

## TODO

- [x] 音频、图像采样
- [x] 音频、图像理解，转文字
- [x] 文字embedding
- [x] 全文检索&向量检索demo
- [ ] 标签设计（可选）
- [ ] 检索实现
- [ ] 效果评估



