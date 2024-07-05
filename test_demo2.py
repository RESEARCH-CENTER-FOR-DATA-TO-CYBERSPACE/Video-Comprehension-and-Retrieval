import gradio as gr
import pandas as pd
from elasticsearch import Elasticsearch
import os

# Get info about the retrieved documents from the response object
def parse_response(response):
	similar_docs = []
	for hit in response.body['hits']['hits']:
		similar_doc = {
			"videoID": hit['_source']['video_id'],
			"docID": hit['_source']['time'],
			"image_text": hit['_source']['image_text'],
			# "audio_text": hit['_source']['audio_text'],
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
    
    response = es.search(index=model_colomn, body=body)
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
    response = es.search(index=model_colomn, body=body)
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
    link0 = "http://127.0.0.1:5000/?video="+str(tmp[0]['videoID'][15:])+'&time='+str(int(tmp[0]['docID'][0:2])*60*60+int(tmp[0]['docID'][3:5])*60+int(tmp[0]['docID'][6:8]))
    link1 = "http://127.0.0.1:5000/?video="+str(tmp[1]['videoID'][15:])+'&time='+str(int(tmp[1]['docID'][0:2])*60*60+int(tmp[1]['docID'][3:5])*60+int(tmp[1]['docID'][6:8]))
    link2 = "http://127.0.0.1:5000/?video="+str(tmp[2]['videoID'][15:])+'&time='+str(int(tmp[2]['docID'][0:2])*60*60+int(tmp[2]['docID'][3:5])*60+int(tmp[2]['docID'][6:8]))

    return '您检索的信息存在于视频"'+tmp[0]['videoID']+'"，且其位于的时间为：'+tmp[0]['docID'][0:2]+'小时'+''+tmp[0]['docID'][3:5]+'分'+tmp[0]['docID'][6:8]+'秒'+tmp[0]['docID'][-3:]+'毫秒'+'\n'\
        +'视频链接：http://127.0.0.1:5000/?video='+str(tmp[0]['videoID'][15:])+'&time='+str(int(tmp[0]['docID'][0:2])*60*60+int(tmp[0]['docID'][3:5])*60+int(tmp[0]['docID'][6:8]))\
        ,f'<a href={link0} target="_blank">点击链接跳转至视频特定帧的位置</a>'\
        ,'您检索的信息存在于视频"'+tmp[1]['videoID']+'"，且其位于的时间为：'+tmp[1]['docID'][0:2]+'小时'+''+tmp[1]['docID'][3:5]+'分'+tmp[1]['docID'][6:8]+'秒'+tmp[1]['docID'][-3:]+'毫秒'+'\n'\
        +'视频链接：http://127.0.0.1:5000/?video='+str(tmp[1]['videoID'][15:])+'&time='+str(int(tmp[1]['docID'][0:2])*60*60+int(tmp[1]['docID'][3:5])*60+int(tmp[1]['docID'][6:8]))\
        ,f'<a href={link1} target="_blank">点击链接跳转至视频特定帧的位置</a>'\
        ,'您检索的信息存在于视频"'+tmp[2]['videoID']+'"，且其位于的时间为：'+tmp[2]['docID'][0:2]+'小时'+''+tmp[2]['docID'][3:5]+'分'+tmp[2]['docID'][6:8]+'秒'+tmp[2]['docID'][-3:]+'毫秒'+'\n'\
        +'视频链接：http://127.0.0.1:5000/?video='+str(tmp[2]['videoID'][15:])+'&time='+str(int(tmp[2]['docID'][0:2])*60*60+int(tmp[2]['docID'][3:5])*60+int(tmp[2]['docID'][6:8]))\
        ,f'<a href={link2} target="_blank">点击链接跳转至视频特定帧的位置</a>'


#接口创建函数
#fn设置处理函数，inputs设置输入接口组件，outputs设置输出接口组件
#fn,inputs,outputs都是必填函数
model_colomn = gr.Dropdown(["video_gpt4o", "video_claude3dot5_test"], label="大模型选项")
link_output0 = gr.HTML()
link_output1 = gr.HTML()
link_output2 = gr.HTML()
demo = gr.Interface(fn=query, inputs=["text","text", model_colomn, "text", ], outputs=["text", link_output0, "text", link_output1, "text", link_output2], title="Video Comprehension and Retrieval", description="输入自然语言文本和Elasticsearch服务器的ip与端口号。返回经过计算最有可能的视频所处的位置，并显示出该视频对应时间的链接。")
demo.launch(share = False)

# gradio test_demo.py
# python query.py -q '普京在哪里' --host '210.45.76.45' --port 9200

# http://127.0.0.1:5000/?video=6_.mp4&time=30