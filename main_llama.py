# -*- coding: utf-8 -*-
import argparse
import logging
import os
import shutil
from sentence_transformers import SentenceTransformer
from vosk import Model, SetLogLevel
import pandas as pd
from sharetape import Sharetape
import base64
import re
from datetime import timedelta
import json
import torch
from transformers import MllamaForConditionalGeneration, AutoProcessor
from peft import PeftModel
from PIL import Image
import os
import argparse
from tqdm import tqdm

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

            save_path = str(clip_path) + '/' + str(i)  + '.png'

            # 保存帧到文件
            cv2.imwrite(save_path, frames[j])

        print(f'Saved clip in {clip_path}')

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


# 将时间转换为秒
def time_to_seconds(t):
    h, m, s = t.split(':')
    s, ms = s.split(',')
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


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

def convert_img2text(clips_path, processor, llama_model):
    contents = {}
    clips = sorted(os.listdir(clips_path))
    for i, clip in enumerate(clips):
        clip_path = os.path.join(clips_path, clip)
        images = [os.path.join(clip_path, img) for img in sorted(os.listdir(clip_path))]

        user_prompt = "请详细描述这张图片你所看到的所有细节，要求包括：1. 对象包括人物、环境、情节等等，宏观和微观的对象都包括； 2. 你可以根据现在的知识进行一些合理的推导与猜测。3. 例如，若出现人物，请给出其身高、肤色、衣着、样貌等特征；如果是交通工具，请给出颜色、种类、品牌、标记、数量等特征。 4. 对于画面中出现的诸如水印、媒体符号等无关内容，可以进行省略。5.你的输出可以尽可能丰富。"
        
        for image_path in images:
            # Getting the base64 string
            image = Image.open(image_path)
            # Prepare the content with the image and text
            content = [
                {"type": "image", "image": image},
                {"type": "text", "text": user_prompt}
            ]

            # Apply the chat template to create the prompt
            prompt = processor.apply_chat_template(
                [{"role": "user", "content": content}],
                add_generation_prompt=True
            )

            # Prepare the inputs for the model
            inputs = processor(
                images=image,
                text=prompt,
                return_tensors="pt"
            ).to(llama_model.device)

            # Generate the model's response
            output = llama_model.generate(**inputs, max_new_tokens=512)

            # Decode the output to get the assistant's response
            response = processor.decode(output[0], skip_special_tokens=True)

            index = response.find('assistant\n\n')
            # 如果找到 'assistant\n\n'，返回其后的文本；否则返回全部response
            if index != -1:
                message = response[index + len('assistant\n\n'):].strip()
            else:
                message = response

        print('------------------------'+str(i)+'-----------------------------------')
        print(message)

        contents[f'{clip}'] = message

    return contents

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video", type=str, required=False, default="videos")
    args = parser.parse_args()
    result_file_path = './hash_folders-llama32/'

    base_model_id = "/home/ygao/models/Llama-3.2-11B-Vision-Instruct"
    lora_model_id = "/home/ygao/models/Llama-3.2-Vision-chinese-lora"

    speech_recognition_model = '/home/ygao/Video-Comprehension-and-Retrieval/model/vosk-model-cn-0.22'
    embedding_model = '/home/ygao/Video-Comprehension-and-Retrieval/model/uer_sbert-base-chinese-nli'

    # Load the processor
    processor = AutoProcessor.from_pretrained(base_model_id)
    # Load the base model
    base_model = MllamaForConditionalGeneration.from_pretrained(
        base_model_id,
        device_map="auto",
        torch_dtype=torch.float16  # Use torch.bfloat16 if your hardware supports it
    ).eval()
    # Load the LoRA model and apply it to the base model
    llama_model = PeftModel.from_pretrained(base_model, lora_model_id)
    # Optionally, merge the LoRA weights with the base model for faster inference
    llama_model = llama_model.merge_and_unload()

    # 递归遍历文件夹及其子文件夹
    for root, dirs, files in os.walk(args.video):
        for filename in files:
            video_path = os.path.join(root, filename)
            logging.info('开始视频处理，结果文件夹：' + result_file_path)
            tmp_id = (result_file_path + filename).rsplit('.', 1)[0]
            print(tmp_id)
            try:
                os.makedirs(tmp_id)
            except FileExistsError:
                logging.info('已处理该视频文件，跳过')
                continue
            # video_id = tmp_id
            video_id = speech2text(video_path, speech_recognition_model, tmp_id)  # 'd7b24747-fade-4765-91e4-40ce04d7865d'
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
                with open(audio2text_path, 'r', encoding='utf-8') as file:
                    timestamps = list(json.load(file).keys())

            # 视频帧采样
            video_downsample(video_path, capture_img_path, timestamps, time_rate=time_rate, time_delta=time_delta)


            # image to text
            img_contents = convert_img2text(capture_img_path, processor, llama_model)
            save_to_json(img_contents, image2text_path)

            # encode text using sbert
            encode_text(video_id, image2text_path, audio2text_path, embedding_path, embedding_model)


