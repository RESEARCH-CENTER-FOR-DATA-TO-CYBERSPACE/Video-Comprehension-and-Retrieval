# Video Comprehension and Retrieval

## Usage
1. 包含音频转文字，视频图像转文字，文字embedding过程。

- `export OPENAI_API_KEY='your-api-key-here'`
- `python main.py -v video_path`

2. video_embedding.json内容存入elasticsearch
- `python create_database.py -d data_path --host es_host -p es_port`

3. 自然语言query
- `python query.py -q '朝代' --host '211.86.152.66' -p 9200`

将embedding结果和全文搜索结果视为两路召回，目前将得分采用归一化方法，消除量纲影响（后续需要改进），取得分排名前三的结果呈现给用户

4. model下载
包含sbert-chinese和vosk-cn模型
- https://huggingface.co/uer/sbert-base-chinese-nli
- https://alphacephei.com/vosk/models/vosk-model-cn-0.22.zip
``` 
zip：https://rec.ustc.edu.cn/share/98f237b0-3112-11ef-8840-898423b2c5db
passwd：rcdc
```

## TODO

- [x] 音频、图像采样
- [x] 音频、图像理解，转文字
- [x] 文字embedding
- [x] 全文检索&向量检索demo
- [ ] 标签设计（可选）
- [ ] 检索实现
- [ ] 效果评估



