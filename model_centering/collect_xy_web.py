#!/usr/bin/env python3
"""Collect XY labels from a browser on the same network; no motor control."""
import argparse
import base64
import os
import threading
import uuid

import cv2
from flask import Flask, jsonify, request


PAGE = '''<!doctype html><html lang="zh-Hant"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>車道目標點標註</title>
<style>body{font-family:sans-serif;max-width:720px;margin:auto;padding:16px}
img{width:100%;max-width:640px;image-rendering:auto;cursor:crosshair}
button{font-size:20px;padding:10px}#message{margin:12px 0}</style>
<h2>車道目標點標註</h2><p>先把車子放好，按「拍一張」，
再點照片中車子應前往的車道中央。每張照片只點一次。</p>
<button onclick="snapshot()">拍一張</button><div id="message">等待拍照</div>
<img id="photo" alt="拍照後會顯示在此" onclick="save(event)">
<script>
let token=null,busy=false;
const photo=document.getElementById('photo');
const message=document.getElementById('message');
async function snapshot(){
 if(busy)return;busy=true;message.textContent='拍照中…';
 try{const r=await fetch('/snapshot');const d=await r.json();
  if(!r.ok)throw Error(d.error);token=d.token;photo.src='data:image/jpeg;base64,'+d.image;
  message.textContent='請點照片中的目標位置';
 }catch(e){message.textContent='拍照失敗：'+e.message;}finally{busy=false;}
}
async function save(e){
 if(!token||busy||!photo.naturalWidth)return;
 const rect=photo.getBoundingClientRect();
 const x=Math.floor((e.clientX-rect.left)*photo.naturalWidth/rect.width);
 const y=Math.floor((e.clientY-rect.top)*photo.naturalHeight/rect.height);
 busy=true;
 try{const r=await fetch('/save',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({token,x,y})});const d=await r.json();
  if(!r.ok)throw Error(d.error);token=null;
  message.textContent='已存第 '+d.count+' 張：'+d.filename+'。移動車子後再按「拍一張」。';
 }catch(e){message.textContent='儲存失敗：'+e.message;}finally{busy=false;}
}
</script></html>'''


def create_app(camera, output, width, height):
    app = Flask(__name__)
    lock = threading.Lock()
    pending = {}
    count = [0]
    os.makedirs(output, exist_ok=True)

    @app.route('/')
    def index():
        return PAGE

    @app.route('/snapshot')
    def snapshot():
        with lock:
            # Drain a few buffered frames after the vehicle is repositioned.
            ok, frame = False, None
            for _ in range(3):
                ok, frame = camera.read()
            if not ok or frame is None:
                return jsonify(error='Camera frame unavailable'), 503
            raw = cv2.resize(frame, (width, height))
            token = uuid.uuid4().hex
            pending.clear()
            pending[token] = raw
            encoded_ok, encoded = cv2.imencode('.jpg', raw)
            if not encoded_ok:
                pending.clear()
                return jsonify(error='JPEG encoding failed'), 500
        return jsonify(token=token, image=base64.b64encode(encoded.tobytes()).decode('ascii'))

    @app.route('/save', methods=['POST'])
    def save():
        data = request.get_json(silent=True) or {}
        with lock:
            raw = pending.get(data.get('token'))
            try:
                x, y = int(data['x']), int(data['y'])
            except (KeyError, TypeError, ValueError):
                return jsonify(error='Invalid coordinates'), 400
            if raw is None or not (0 <= x < width and 0 <= y < height):
                return jsonify(error='Expired snapshot or point outside image'), 400
            name = 'xy_{}_{}_{}.jpg'.format(x, y, uuid.uuid4().hex[:12])
            if not cv2.imwrite(os.path.join(output, name), raw):
                return jsonify(error='Cannot save image'), 500
            pending.clear()
            count[0] += 1
            print('Saved {}: {}'.format(count[0], name), flush=True)
            return jsonify(count=count[0], filename=name)

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--output', default='dataset_xy')
    parser.add_argument('--port', type=int, default=5001)
    args = parser.parse_args()
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        parser.error('Cannot open camera {}'.format(args.camera))
    try:
        create_app(camera, args.output, 320, 240).run(
            host='0.0.0.0', port=args.port, threaded=True, debug=False, use_reloader=False)
    finally:
        camera.release()


if __name__ == '__main__':
    main()
