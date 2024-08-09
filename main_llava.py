import argparse
import logging
import os
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

from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def parse_time(time_str):
    """Convert time string to a timedelta object, handling milliseconds correctly."""
    time_parts = re.split('[:,]', time_str)
    hours, minutes, seconds = map(int, time_parts[:3])
    milliseconds = int(time_parts[3]) if len(time_parts) > 3 else 0
    return timedelta(hours=hours, minutes=minutes, seconds=seconds, milliseconds=milliseconds)


def format_time(td):
    """将timedelta对象格式化为SRT时间格式"""
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = td.microseconds // 1000
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


def read_and_extract_srt(file_path, start_time_str='00:00:00,000', end_time_str='100:00:00,000', interval_seconds=60,
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


def video_downsample(video_name, output_path, timestamps=None, time_rate=1, time_delta=60):
    import cv2
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)

    cap = cv2.VideoCapture(video_name)
    fps = cap.get(5)  # 获取视频帧率
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取总帧数
    video_length_seconds = total_frames / fps  # 计算视频总长度（秒）
    if not timestamps:
        timestamps = [format_time(timedelta(seconds=x)) for x in range(0, int(video_length_seconds), time_delta)]
    timestamps.append(format_time(timedelta(seconds=video_length_seconds)))

    # 对于每对时间点
    for i in range(len(timestamps) - 1):
        start_td = parse_time(timestamps[i])
        end_td = parse_time(timestamps[i + 1])
        current_td = start_td
        frames = []
        clip_path = os.path.join(output_path, f'{format_time(start_td)}')
        os.mkdir(clip_path)
        while current_td < end_td:

            cap.set(cv2.CAP_PROP_POS_MSEC, min(current_td.total_seconds(), video_length_seconds) * 1000)  # 设置视频时间
            ret, frame = cap.read()  # 读取帧
            if not ret:
                break  # 如果读取失败，跳出循环
            frames.append(frame)

            current_td += timedelta(seconds=time_rate)  # 增加 time_rate 秒
        for j in range(len(frames)):
            # 保存帧到文件
            cv2.imwrite(os.path.join(clip_path, f'{j}.jpg'), frames[j])
        print(f'Saved clip in {clip_path}')

    # frameRate = int(FPS) * time_rate
    # c = 0
    # while(True):
    #     cap.set()
    #     ret, frame = cap.read()
    #     if ret:
    #         if(c % frameRate == 0):
    #             logging.info("开始截取视频第：" + str(c) + " 帧")
    #             # 这里就可以做一些操作了：显示截取的帧图片、保存截取帧到本地
    #             cv2.imwrite(f"{output_path}/{str(c)}.jpg", frame)  # 这里是将截取的图像保存在本地
    #         c += 1
    #         cv2.waitKey(0)
    #     else:
    #         logging.info("所有帧都已经保存完成")
    #         break
    # cap.release()


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def convert_img2text_raw(image_path):
    images = [os.path.join(image_path, file) for file in os.listdir(image_path)]
    for image in images:
        # Getting the base64 string
        base64_image = encode_image(image)

        response = client.chat.completions.create(
            model=gpt_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text",
                         "text": "What’s in this image? Make sure you use the same language as the image."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        },
                    ],
                }
            ],
            max_tokens=300,
        )

        print(response.choices[0])
        break


# 将时间转换为秒
def time_to_seconds(t):
    h, m, s = t.split(':')
    s, ms = s.split(',')
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def convert_img2text(clips_path):
    contents = {}
    clips = sorted(os.listdir(clips_path))
    for clip in clips:
        clip_path = os.path.join(clips_path, clip)
        images = [os.path.join(clip_path, img) for img in sorted(os.listdir(clip_path))]
        messages = [
            {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "下面一组图片是从一个视频中采样得到的，请根据这些帧总结视频内容。注意输出json中只需要包含'content'字段。"},
                ],
            }
        ]
        for image in images:
            # Getting the base64 string
            base64_image = encode_image(image)
            messages[1]['content'].append(
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "low"}})

        response = client.chat.completions.create(
            response_format={"type": "json_object"},
            model=gpt_model,
            messages=messages,
            temperature=0,
            timeout=300
        )
        try:
            contents[f'{clip}'] = json.loads(response.choices[0].message.content)['content']
        except:
            logging.error(f'{clip} response {response.choices[0].message.content} parse error!')
            contents[f'{clip}'] = response.choices[0].message.content

    return contents


def llava_convert_img2text(clips_path):
    processor = LlavaNextProcessor.from_pretrained("model/llava-hf/llava-v1.6-mistral-7b-hf")
    model = LlavaNextForConditionalGeneration.from_pretrained("model/llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True) 
    model.to("cuda:3")

    contents = {}
    clips = sorted(os.listdir(clips_path))
    for clip in clips:
        clip_path = os.path.join(clips_path, clip)
        images = [os.path.join(clip_path, img) for img in sorted(os.listdir(clip_path))]
        for image in images:

            image = Image.open(image)
            # prompt = "[INST] <image>\nWhat is shown in this image? [/INST]"
            prompt = "[INST] <image>\n图片里都是什么内容，请输出中文回答? [/INST]"

            inputs = processor(prompt, image, return_tensors="pt").to("cuda:3")

            # autoregressively complete prompt
            output = model.generate(**inputs, max_new_tokens=300)
            print(processor.decode(output[0], skip_special_tokens=True))
            try:
                contents[f'{clip}'] = json.loads(processor.decode(output[0], skip_special_tokens=True))['content']
            except:
                logging.error(f'{clip} response {processor.decode(output[0], skip_special_tokens=True)} parse error!')
                contents[f'{clip}'] = processor.decode(output[0], skip_special_tokens=True)

    return contents

def speech2text(video, mdoel_path, filename):
    SetLogLevel(-1)
    model = Model(model_path=mdoel_path)
    logging.info("sp2t setup")


    video_id = './hash_folders/' + filename
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
    # os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video", type=str, required=False, default="./test_video")
    args = parser.parse_args()


    speech_recognition_model = './model/vosk-model-cn-0.22'
    embedding_model = './model/uer_sbert-base-chinese-nli'

    # 递归遍历文件夹及其子文件夹
    for root, dirs, files in os.walk(args.video):
        for filename in files:
            video_path = os.path.join(root, filename)
            logging.info('开始视频处理')
            tmp_id = './hash_folders/' + filename
            try:
                os.makedirs(tmp_id)
            except FileExistsError:
                logging.info('已处理该视频文件，跳过')
                continue
            video_id = speech2text(video_path, speech_recognition_model, filename)  # 'd7b24747-fade-4765-91e4-40ce04d7865d'
            time_rate = 5  # 每隔timeRate(s)截取一帧
            time_delta = 30  # 每个切片time_delta(s)

            # 文件路径
            speech_path = f"{video_id}/audio.wav"
            srt_path = f'{video_id}/captions.srt'
            audio2text_path = f'{video_id}/video_text.json'
            image2text_path = f'{video_id}/image_text.json'
            embedding_path = f'{video_id}/video_embedding.json'
            capture_img_path = f"./{video_id}/capture_image"

            # 读取、切片和保存SRT文件
            subtitles = read_and_extract_srt(srt_path, interval_seconds=time_delta)
            save_to_json(subtitles, audio2text_path)

            logging.info(f'提取音频位置为：{speech_path}')
            logging.info(f'语音识别结果：{audio2text_path}')
            timestamps = None

            if os.path.exists(audio2text_path):
                with open(audio2text_path, 'r') as file:
                    timestamps = list(json.load(file).keys())

            # 视频帧采样
            video_downsample(video_path, capture_img_path, timestamps, time_rate=time_rate, time_delta=time_delta)

            # image to text using GPT
            img_contents = llava_convert_img2text(capture_img_path)


            save_to_json(img_contents, image2text_path)

            # encode text using sbert
            encode_text(video_id, image2text_path, audio2text_path, embedding_path, embedding_model)
