# Video Comprehension and Retrieval

## Usage
1. 包含音频转文字，视频图像转文字，文字embedding过程。

- `export OPENAI_API_KEY='your-api-key-here'`
- `python main.py -v video_path`

2. video_embedding.json内容存入elasticsearch
- `python create_database.py -d data_path --host es_host -p es_port`

3. 自然语言query
- `python query.py -q query --host es_host -p es_port`
- 将embedding结果和全文搜索结果视为两路召回，将召回结果用于排序。排序部分目前是将得分采用归一化方法消除量纲影响（后续需要改进），最后取得分排名前三的结果呈现给用户

4. model下载
包含sbert-chinese和vosk-cn模型
- https://huggingface.co/uer/sbert-base-chinese-nli
- https://alphacephei.com/vosk/models/vosk-model-cn-0.22.zip
``` 
zip：https://rec.ustc.edu.cn/share/98f237b0-3112-11ef-8840-898423b2c5db
passwd：rcdc
```

5. 前端展示与视频服务器的搭建
- 前端使用了gradio作为展示界面，需要安装gradio库。输入相关信息，查询elasticsearch，返回查询结果与视频的链接
``` 
gradio test_demo.py
```
- 视频服务器使用了flask搭建，需要安装flask库，请求访问某个视频特定时间帧 (http://127.0.0.1:5000/?video=10.mp4&time=32)，返回页面展示该视频并从特定时间帧开始播放。视频播放界面前端html存于./templates/index.html，视频文件放在./static
``` 
python test_video_server.py
```

## TODO

- [x] 音频、图像采样
- [x] 音频、图像理解，转文字
- [x] 文字embedding
- [x] 全文检索&向量检索demo
- [ ] 标签设计（可选）
- [x] 混合检索实现
- [x] 真实视频测试
- [ ] 效果评估



