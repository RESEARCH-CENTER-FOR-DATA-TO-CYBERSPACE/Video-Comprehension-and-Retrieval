import pandas as pd
from elasticsearch import Elasticsearch
import argparse

# Get info about the retrieved documents from the response object
def parse_response(response):
	similar_docs = []
	for hit in response.body['hits']['hits']:
		similar_doc = {
			"videoID": hit['_source']['video_id'],
			"docID": hit['_source']['time'],
			"image_text": hit['_source']['image_text'],
			"audio_text": hit['_source']['audio_text'],
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



# Parse the arguments
parser = argparse.ArgumentParser(
                    prog='retrieveDocuments.py',
                    description='make queries to the server')

#parser.add_argument('-e', '--embeddings', help='provide the type of embeddings you want to use. The Valid types are : word2vec, doc2vec, USE, SBERT', nargs='+', default=['word2vec', 'doc2vec', 'USE', 'SBERT'])
parser.add_argument('-q', '--query_text', help='provide the the text to be queried', required=False)
parser.add_argument('--host', default='210.45.76.45', help='enter the host which the elasticsearch server listens to')
parser.add_argument('-p', '--port', default=9200, help='enter the port which the elasticsearch server listens to')
#parser.add_argument('-s', '--save_path', help='provide the path to the file you want to save the json file that contains the results', required=True)
args = parser.parse_args()


# Establish connection with the elasticsearch server
es = Elasticsearch([{'host': args.host, 'port': args.port, 'scheme':'http'}])

# Read the text in the query file
# with open(args.query_path, "r") as f:
#     query_text = f.read()

results = {}
query_text = args.query_text if args.query_text else '与疫情有关片段在哪里'
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
response = es.search(index='video_cases', body=body)

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
response = es.search(index='video_cases', body=body)

similar_docs = parse_response(response)

# Add the info about the 10 most similar documents into the dictonary entry for each embedding type
results[str(embedding_type)] = similar_docs


print(results.keys())



# print(results['tf_idf'])
# print(results[embedding_type])
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

print('**************************')
print('以下是最有可能的前三个视频片段')
for i in range(0,3):
  tmp = sorted_by_similarity_desc[i]
  print('您检索的信息存在于视频"'+tmp['videoID']+'"，且其时间为：'+tmp['docID'][0:2]+'小时'+''+tmp['docID'][3:5]+'分'+tmp['docID'][6:8]+'秒'+tmp['docID'][-3:]+'毫秒')
print('**************************')