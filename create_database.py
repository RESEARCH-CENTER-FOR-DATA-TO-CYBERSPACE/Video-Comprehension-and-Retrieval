import pandas as pd
from elasticsearch import Elasticsearch
import argparse

parser = argparse.ArgumentParser(
                    prog='createDatabase',
                    description='create the elasticsearch database and add the documents along with their embeddings')

parser.add_argument('-d', '--data_path', help='enter the path to the .csv file that contains the embedding vectors')
parser.add_argument('-p', '--port', default=9200, help='enter the port which the elasticsearch server listens to')

# Parse the arguments
args = parser.parse_args()

# args.data_path = './6a215328-9bf3-4525-bb3c-124dd4efca66/video_embedding.json'

# Establish connection with the elasticsearch server
es = Elasticsearch([{'host': '211.86.152.66', 'port':9200, 'scheme':'http'}])

# Delete the Database if it exists
es.options(ignore_status=[400,404]).indices.delete(index='video_cases')

# Specify the mapping, which will describe the structure of each document
mappings = {
	"properties":{
		"time":{
            "type": "text", 
		},
		"image_text":{
				"type":"text"
		},
		"image_emb":{
				"type": "dense_vector",
				"dims": 768
		},
		"audio_text":{
				"type":"text"
		},
		"audio_emb":{
				"type": "dense_vector",
				"dims": 768
		}    
    }
}

# Create the index
result = es.indices.create(index="video_cases", mappings=mappings)

# Access the data that are going to be uploaded on the server
embeddings_df = pd.read_json(str(args.data_path))

# Iterate over the rows of the csv and add each document on the database
for index, row in embeddings_df.iterrows():
	doc = {
		'time': str(row['time']),
		'image_text' : str(row['image_text']),
		"image_emb": row['image_emb'],
		'audio_text': str(row['audio_text']),
		"audio_emb": row['audio_emb'],
	}

	res = es.index(index="video_cases", id=index, document=doc)