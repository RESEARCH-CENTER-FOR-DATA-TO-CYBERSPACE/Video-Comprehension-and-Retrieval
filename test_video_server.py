from flask import Flask, render_template, request
import os

app = Flask(__name__)

# 视频文件的路径
VIDEO_PATH = 'static/hash_folders/'

@app.route('/')
def index():
    video_file = request.args.get('video', '6_.mp4')
    start_time = request.args.get('time', '0')
    video_file = os.path.join(VIDEO_PATH, video_file)
    # 检查视频文件是否存在
    if not os.path.exists(video_file):
        return "Video file not found", 404
    
    return render_template('index.html', video_file=video_file, start_time=start_time)

if __name__ == '__main__':
    app.run(debug=True)


# python test_video_server.py 