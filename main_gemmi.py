# -*- coding: utf-8 -*-
import argparse
import logging
import os
import time
import shutil
import uuid
from sentence_transformers import SentenceTransformer
import torch
from vosk import Model, SetLogLevel
import pandas as pd
from sharetape import Sharetape
import base64
import re
from datetime import timedelta
import json

import httpx
import argparse
from openai import OpenAI
from tqdm import tqdm
from openai import AzureOpenAI
import google.generativeai as genai
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def parse_time(time_str):
    """Convert time string to a timedelta object, handling milliseconds correctly."""
    time_parts = re.split('[:.,_]', time_str)
    hours, minutes, seconds = map(int, time_parts[:3])
    milliseconds = int(time_parts[3]) if len(time_parts) > 3 else 0
    return timedelta(hours=hours, minutes=minutes, seconds=seconds, milliseconds=milliseconds)


def format_time(td):
    """将timedelta对象格式化为SRT时间格式"""
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = td.microseconds // 1000
    return f"{hours:02}_{minutes:02}_{seconds:02}_{milliseconds:03}"


def read_and_extract_srt(file_path, start_time_str='00_00_00_000', end_time_str='100_00_00_000', interval_seconds=60,
                         padding_seconds=5):
    """Reads SRT content from file and extracts subtitles within specified time range split by intervals with padding."""
    start_time = parse_time(start_time_str)
    end_time = parse_time(end_time_str)
    interval = timedelta(seconds=interval_seconds)
    padding = timedelta(seconds=padding_seconds)
    subtitles = {}

    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    current_segment_start = start_time - padding
    segment_texts = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if re.match(r'^\d+$', line):  # Skip sequence numbers
            i += 1
            continue
        if '-->' in line:
            start, end = line.split(' --> ')
            start_td = parse_time(start)
            end_td = parse_time(end)

            # Skip subtitles outside the specified time range
            if start_td < start_time or end_td > end_time:
                i += 2  # Skip this entry (time line + text line)
                continue

            # Start a new segment if necessary
            if start_td >= current_segment_start + interval + padding:
                if segment_texts:
                    subtitles[format_time(current_segment_start + padding)] = ' '.join(segment_texts).strip()
                current_segment_start = start_td - padding
                segment_texts = []

            i += 1
            continue

        # Collect text from the following lines until another timestamp or sequence number
        if lines[i].strip() and not re.match(r'^\d+$', lines[i].strip()):
            segment_texts.append(lines[i].strip())

        i += 1

    # Capture the final segment if within end time
    if segment_texts and current_segment_start <= end_time:
        subtitles[format_time(current_segment_start + padding)] = ' '.join(segment_texts).strip()

    return subtitles


def save_to_json(data, output_path):
    """Saves subtitle data to a JSON file."""
    with open(output_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)


def encode_text(video_id, image2text_path, audio2text_path, output_path, mdoel_name='uer/sbert-base-chinese-nli'):
    # 加载预训练的模型和分词器
    model = SentenceTransformer(mdoel_name, trust_remote_code=True)
    model.max_seq_length = 512
    df = pd.DataFrame(columns=['video_id', 'time', 'image_text', 'image_emb', 'audio_text', 'audio_emb'])
    data2 = None
    if os.path.exists(audio2text_path):
        with open(audio2text_path, 'r', encoding='utf-8') as f2:
            data2 = json.load(f2)
    with open(image2text_path, 'r', encoding='utf-8') as f1:
        data = json.load(f1)
        for time, text1 in data.items():
            text1 = text1.replace(' ', '')
            embedding1 = model.encode(text1)
            if data2:
                text2 = data2[time].replace(' ', '')
                embedding2 = model.encode(text2)
                df = pd.concat([df, pd.DataFrame(
                    {'video_id': video_id, 'time': time, 'image_text': text1, 'image_emb': [embedding1], 'audio_text': text2,
                     'audio_emb': [embedding2]})], ignore_index=True)
            else:
                df = pd.concat([df, pd.DataFrame({'video_id': video_id, 'time': time, 'image_text': text1, 'image_emb': [embedding1]})], ignore_index=True)

    df.to_json(output_path, index=False)


# 将时间戳转换为秒
def time_to_seconds(time_str):
    h, m, s, ms = map(int, time_str.split('_'))
    return h * 3600 + m * 60 + s + ms / 1000

def video_split(video_name, output_path, timestamps=None, time_delta=60):
    import os
    import shutil
    from datetime import timedelta
    from moviepy.editor import VideoFileClip
    from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)

    with VideoFileClip(video_name) as video:
        video_length_seconds = video.duration

    if not timestamps:
        timestamps = [format_time(timedelta(seconds=x)) for x in range(0, int(video_length_seconds), time_delta)]
    timestamps.append(format_time(timedelta(seconds=video_length_seconds)))

    # 对于每对时间点
    for i in range(len(timestamps) - 1):
        start_td = time_to_seconds(timestamps[i])
        end_td = time_to_seconds(timestamps[i + 1])

        # 输出文件路径
        split_video_path = os.path.join(output_path, f'{(timestamps[i])}.mp4')

        # 提取子视频
        ffmpeg_extract_subclip(video_name, start_td, end_td, targetname=split_video_path)

        print(f'Saved video segment: {split_video_path}')


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def upload_to_gemini(path, mime_type=None):
  """Uploads the given file to Gemini.

  See https://ai.google.dev/gemini-api/docs/prompting_with_media
  """
  file = genai.upload_file(path, mime_type=mime_type)
  print(f"Uploaded file '{file.display_name}' as: {file.uri}")
  return file

def wait_for_files_active(files):
  """Waits for the given files to be active.

  Some files uploaded to the Gemini API need to be processed before they can be
  used as prompt inputs. The status can be seen by querying the file's "state"
  field.

  This implementation uses a simple blocking polling loop. Production code
  should probably employ a more sophisticated approach.
  """
  print("Waiting for file processing...")
  for name in (file.name for file in files):
    file = genai.get_file(name)
    while file.state.name == "PROCESSING":
      print(".", end="", flush=True)
      time.sleep(10)
      file = genai.get_file(name)
    if file.state.name != "ACTIVE":
      raise Exception(f"File {file.name} failed to process")
  print("...all files ready")
  print()

def convert_video2text(clips_path, model):
    contents = {}
    clips = sorted(os.listdir(clips_path))
    for clip in clips:
        clip_path = os.path.join(clips_path, clip)
        files = [
            upload_to_gemini(clip_path, mime_type="video/mp4"),
            ]
        # Some files have a processing delay. Wait for them to be ready.
        wait_for_files_active(files)
        chat_session = model.start_chat(
        history=[
            {
            "role": "user",
            "parts": [
                files[0],
            ],
            },
        ]
        )

        response = chat_session.send_message("总结视频内容。注意输出需要是标准的json形式，但是不要输出json这个词，且输出结果中只能有'content'字段。")

        print(response.text)

        clip_name = clip.split('.')[0]
        try:
            contents[f'{clip_name}'] = json.loads(response.text)['content']
        except:
            logging.error(f'{clip_name} response {response.text} parse error!')
            contents[f'{clip_name}'] = response.text

    return contents


def speech2text(video, mdoel_path, video_id):
    SetLogLevel(-1)
    model = Model(model_path=mdoel_path)
    logging.info("sp2t setup")

    os.makedirs(f"{video_id}/tmp")

    audio = f"{video_id}/audio.wav"

    shartape = Sharetape(
        video,
        audio,
        f"{video_id}/tmp/mono_audio.wav",
        f"{video_id}/tmp/transcript.txt",
        f"{video_id}/tmp/words.json",
        f"{video_id}/captions.srt",
        model,
    )
    shartape.extract_transcript()

    return video_id


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video", type=str, required=False, default="./videos")
    args = parser.parse_args()
    result_file_path = './hash_folders-gemmi/'
    # result_file_path = './hash_test/'

    speech_recognition_model = './model/vosk-model-cn-0.22'
    embedding_model = './model/uer_sbert-base-chinese-nli'

    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    # Create the model
    # See https://ai.google.dev/api/python/google/generativeai/GenerativeModel
    generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
    }
    model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    # safety_settings = Adjust safety settings
    # See https://ai.google.dev/gemini-api/docs/safety-settings
    system_instruction="You are a helpful assistant designed to output JSON. And please use the same language as the question.",
    )


    # 递归遍历文件夹及其子文件夹
    for root, dirs, files in os.walk(args.video):
        for filename in files:
            video_path = os.path.join(root, filename)
            logging.info('开始视频处理，结果文件夹：' + result_file_path)
            tmp_id = (result_file_path + filename).rsplit('.', 1)[0]
            try:
                os.makedirs(tmp_id)
            except FileExistsError:
                logging.info('已处理该视频文件，跳过')
                continue
            video_id = speech2text(video_path, speech_recognition_model, tmp_id)  # 'd7b24747-fade-4765-91e4-40ce04d7865d'
            time_delta = 30  # 每个切片time_delta(s)

            # 文件路径
            # video_id = 'hash_test/0_经济支教问题'
            speech_path = f"{video_id}/audio.wav"
            srt_path = f'{video_id}/captions.srt'
            audio2text_path = f'{video_id}/video_text.json'
            image2text_path = f'{video_id}/image_text.json'
            embedding_path = f'{video_id}/video_embedding.json'
            split_video_path = f'{video_id}/split_video'

            # 读取、切片和保存SRT文件
            subtitles = read_and_extract_srt(srt_path, interval_seconds=time_delta)
            save_to_json(subtitles, audio2text_path)

            logging.info(f'提取音频位置为：{speech_path}')
            logging.info(f'语音识别结果：{audio2text_path}')
            timestamps = None

            if os.path.exists(audio2text_path):
                with open(audio2text_path, 'r', encoding='utf-8') as file:
                    timestamps = list(json.load(file).keys())

            # 视频分割
            video_split(video_path, split_video_path, timestamps, time_delta=time_delta)
            # video to text using GPT
            img_contents = convert_video2text(split_video_path, model)
            save_to_json(img_contents, image2text_path)

            # encode text using sbert
            encode_text(video_id, image2text_path, audio2text_path, embedding_path, embedding_model)

            print(video_id, 'finish')
