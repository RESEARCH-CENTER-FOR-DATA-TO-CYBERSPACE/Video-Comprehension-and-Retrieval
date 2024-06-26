import pandas as pd
from elasticsearch import Elasticsearch
import argparse
from pathlib import Path
parser = argparse.ArgumentParser(
                    prog='createDatabase',
                    description='create the elasticsearch database and add the documents along with their embeddings')

parser.add_argument('-d', '--data_path', help='enter the path to the .csv file that contains the embedding vectors')
parser.add_argument('-p', '--port', default=9200, help='enter the port which the elasticsearch server listens to')

# Parse the arguments
args = parser.parse_args()
import os

# Establish connection with the elasticsearch server
es = Elasticsearch([{'host': '211.86.152.66', 'port': 9200, 'scheme': 'http'}])
# Delete the Database if it exists
es.options(ignore_status=[400, 404]).indices.delete(index='video_cases')

# Specify the mapping, which will describe the structure of each document
mappings = {
	"properties": {
		"video_id": {
			"type": "text",
		},
		"time": {
			"type": "text",
		},
		"image_text": {
			"type": "text"
		},
		"image_emb": {
			"type": "dense_vector",
			"dims": 768
		},
		"audio_text": {
			"type": "text"
		},
		"audio_emb": {
			"type": "dense_vector",
			"dims": 768
		}
	}
}

# Create the index
result = es.indices.create(index="video_cases", mappings=mappings)

items = Path("./hash_folders").iterdir()
folder_names = [item.name for item in items if item.is_dir()]
for folder_name in folder_names:

	args.data_path = './hash_folders/' + folder_name + '/video_embedding.json'

	# Access the data that are going to be uploaded on the server
	embeddings_df = pd.read_json(str(args.data_path))

	# Iterate over the rows of the csv and add each document on the database
	for index, row in embeddings_df.iterrows():
		doc = {
			'video_id': str(row['video_id']),
			'time': str(row['time']),
			'image_text' : str(row['image_text']),
			"image_emb": row['image_emb'],
			'audio_text': str(row['audio_text']),
			"audio_emb": row['audio_emb'],
		}
		# 生成一个唯一的文档 ID
		doc_id = f"{row['video_id']}_{row['time']}"
		# 使用文档 ID 添加文档到索引中，确保不会覆盖
		res = es.index(index="video_cases", id=doc_id, document=doc)
