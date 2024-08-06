import gradio as gr
import pandas as pd
from elasticsearch import Elasticsearch
import os
import math
os.environ['GRADIO_TEMP_DIR'] = './tmp'
video_path = './static/hash_folders/'
es_indices = {'gpt4o': 'video_gpt4o_demo', 'claude3.5': 'video_claude3dot5_demo', 'glm4v': 'video_glm4v_demo'}

# Get info about the retrieved documents from the response object
def parse_response(response):
	similar_docs = []
	for hit in response.body['hits']['hits']:
		similar_doc = {
			"videoID": hit['_source']['video_id'],
			"docID": hit['_source']['time'],
			"image_text": hit['_source']['image_text'],
			"similarity": hit['_score']
		}
		similar_docs.append(similar_doc)
	return similar_docs

def get_embedding(text, mdoel_name='./model/uer_sbert-base-chinese-nli'): # uer/sbert-base-chinese-nli
    # 加载预训练的模型和分词器
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(mdoel_name, trust_remote_code=True)
    model.max_seq_length = 512

    return model.encode(text)

def query(host, port, model_colomn, query_text):
    # Establish connection with the elasticsearch server
    es = Elasticsearch([{ 'host':host, 'port': int(port), 'scheme':'http'}])

    results = {}
    query_text = query_text if query_text else '与疫情有关片段在哪里'
    # First retrieve the documents based on the default TF/IDF metric:
    body = {
    "query": {
        "match": {
        "image_text": {
            "query": query_text
        }
        }
    }
    }
    
    response = es.search(index=es_indices[model_colomn], body=body)
    # response = es.search(index='video_gpt4o', body=body)
    # response = es.search(index='video_claude3dot5', body=body)

    results['tf_idf'] = parse_response(response)

    embedding_type = 'image_emb'

    # Get the embedding of the query text
    query_embedding = get_embedding(query_text)

    # Create the body of the search request
    body = {
    "query": {
        "script_score": {
        "query": {
            "match_all": {}
        },
        "script": {
            # Take the cosine similarity between the query vector and the vectors of the indexed documents
            "source": f"cosineSimilarity(params.queryVector, '{embedding_type}')+1.0",
            "params": {
            "queryVector": query_embedding
                    }
                }
            }
        }
    }
    # Perform the request
    response = es.search(index=es_indices[model_colomn], body=body)
    # response = es.search(index='video_gpt4o', body=body)
    # response = es.search(index='video_claude3dot5', body=body)


    similar_docs = parse_response(response)

    # Add the info about the 10 most similar documents into the dictonary entry for each embedding type
    results[str(embedding_type)] = similar_docs

    tf_idf_min = results['tf_idf'][-1]['similarity']
    tf_idf_max = results['tf_idf'][0]['similarity']
    emb_min = results[embedding_type][-1]['similarity']
    emb_max = results[embedding_type][0]['similarity']

    for item in results['tf_idf']:
        item['similarity'] = (item['similarity']-tf_idf_min)/(tf_idf_max-tf_idf_min)
    for item in results[embedding_type]:
        item['similarity'] = (item['similarity']-emb_min)/(emb_max-emb_min)

    total = results['tf_idf']+results[embedding_type]
    sorted_by_similarity_desc = sorted(total, key=lambda x: x['similarity'], reverse=True)

    # print('**************************')
    # print('以下是最有可能的前三个视频片段')
    # for i in range(0,3):
    #     tmp = sorted_by_similarity_desc[i]
    #     print('您检索的信息存在于视频"'+tmp['videoID']+'"，且其时间为：'+tmp['docID'][0:2]+'小时'+''+tmp['docID'][3:5]+'分'+tmp['docID'][6:8]+'秒'+tmp['docID'][-3:]+'毫秒')
    # print('**************************')

    tmp = sorted_by_similarity_desc
    link0 = "http://127.0.0.1:5000/?video="+tmp[0]['videoID'].split('/')[-1]+'&time='+str(int(tmp[0]['docID'][0:2])*60*60+int(tmp[0]['docID'][3:5])*60+int(tmp[0]['docID'][6:8]))
    link1 = "http://127.0.0.1:5000/?video="+tmp[1]['videoID'].split('/')[-1]+'&time='+str(int(tmp[1]['docID'][0:2])*60*60+int(tmp[1]['docID'][3:5])*60+int(tmp[1]['docID'][6:8]))
    link2 = "http://127.0.0.1:5000/?video="+tmp[2]['videoID'].split('/')[-1]+'&time='+str(int(tmp[2]['docID'][0:2])*60*60+int(tmp[2]['docID'][3:5])*60+int(tmp[2]['docID'][6:8]))

    return '您检索的信息存在于视频"'+tmp[0]['videoID']+'"，且其时间为：'+tmp[0]['docID'][0:2]+'小时'+''+tmp[0]['docID'][3:5]+'分'+tmp[0]['docID'][6:8]+'秒'+tmp[0]['docID'][-3:]+'毫秒'+'\n'\
        +f'视频链接：{link0}'\
        ,f'<a href={link0} target="_blank">点击链接跳转至视频特定帧的位置</a>'\
        ,'您检索的信息存在于视频"'+tmp[1]['videoID']+'"，且其时间为：'+tmp[1]['docID'][0:2]+'小时'+''+tmp[1]['docID'][3:5]+'分'+tmp[1]['docID'][6:8]+'秒'+tmp[1]['docID'][-3:]+'毫秒'+'\n'\
        +f'视频链接：{link1}'\
        ,f'<a href={link1} target="_blank">点击链接跳转至视频特定帧的位置</a>'\
        ,'您检索的信息存在于视频"'+tmp[2]['videoID']+'"，且其时间为：'+tmp[2]['docID'][0:2]+'小时'+''+tmp[2]['docID'][3:5]+'分'+tmp[2]['docID'][6:8]+'秒'+tmp[2]['docID'][-3:]+'毫秒'+'\n'\
        +f'视频链接：{link2}'\
        ,f'<a href={link2} target="_blank">点击链接跳转至视频特定帧的位置</a>'



def get_video_list():
    videos = []
    # 按顺序遍历video_path下的所有mp4文件
    # 返回一个列表，列表中每个元素是一个字典，字典包含视频的标题和路径
    for root, dirs, files in os.walk(video_path):
        for file in files:
            if file.endswith('.mp4'):
                if file.split('_')[0] not in ['13', '17', '20', '7', '10']:
                    continue
                videos.append({
                    "title": file.split('.')[0],
                    "path": os.path.join(root, file)
                })

    return videos

with gr.Blocks(title="Video Comprehension and Retrieval") as demo:
    gr.Markdown("# <center> Video Comprehension and Retrieval")
    gr.Markdown("输入自然语言文本和Elasticsearch服务器的ip与端口号。返回经过计算最有可能的视频所处的位置，并显示出该视频对应时间的链接。")
    
    with gr.Row():
        host = gr.Textbox(label="Server Host", value='127.0.0.1')
        port = gr.Textbox(label="Port", value='9200')
        model_column = gr.Dropdown(["claude3.5", "gpt4o", "glm4v"], label="大模型选项")

    query_text = gr.Dropdown(
        choices=["出现大量军用飞机的画面",
                 "请定位大规模轰炸、战斗的场面",
                 "请定位韩国总统尹锡悦和美国总统拜登会面的场景",
                 "我需要查找一辆印有Verizon的白色货车",
                 "请帮我定位一群骑着自行车的人，他们领头的穿着黑色短袖",
                 "视频的何处出现了无人机视角拍摄的画面？",
                "我需要定位粉红色碎花连衣裙的女士，年龄应该大于四十岁，挎着黑色背包，手里拎着白色的购物袋",
                "定位一个戴着鸭舌帽、穿着蓝色上衣和卡其色短裤，背着绿色背包的人。年龄应该大于四十岁",],
        label="请输入要检索的视频描述",
        allow_custom_value=True,
    )
    
    submit_btn = gr.Button("Submit")
    
    output0 = gr.Textbox(label="Output 1")
    link_output0 = gr.HTML()
    output1 = gr.Textbox(label="Output 2")
    link_output1 = gr.HTML()
    output2 = gr.Textbox(label="Output 3")
    link_output2 = gr.HTML()
    
    gr.Markdown("## Available Videos")
    
    videos = get_video_list()
    num_videos = len(videos)
    num_rows = math.ceil(num_videos / 5)
    
    video_components = []
    for i in range(num_rows):
        with gr.Row():
            for j in range(5):
                index = i * 5 + j
                if index < num_videos:
                    video = gr.Video(label=videos[index]["title"], height=225, width=300, include_audio=False)
                    video_components.append(video)
                else:
                    # Add an empty column to maintain the 5-column layout
                    gr.Column(scale=1, min_width=300)
    
    def populate_videos():
        return [video["path"] for video in videos]
    
    demo.load(fn=populate_videos, outputs=video_components)
    
    submit_btn.click(
        fn=query,
        inputs=[host, port, model_column, query_text],
        outputs=[output0, link_output0, output1, link_output1, output2, link_output2]
    )

demo.launch(server_port=8000, share=False)
# gradio test_demo.py

# http://127.0.0.1:5000/?video=6_.mp4&time=30