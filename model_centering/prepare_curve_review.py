"""Prepare a self-contained, local target-label review from recorded driving."""
from pathlib import Path
import base64
import csv
import json
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'curve_review_20261025'
OUT.mkdir(exist_ok=True)
(OUT / 'images').mkdir(exist_ok=True)
groups = []
for video_name in ('bend_exit_v2.avi', 'bend_exit_v3.avi'):
    source = Path.home() / 'Downloads' / video_name
    rows = list(csv.DictReader(Path(str(source)+'.csv').open()))
    segments = []
    for i,row in enumerate(rows):
        if row['sent_command_at_result'].startswith('DRIVE'):
            if not segments or i-segments[-1][-1]>12:
                segments.append([])
            segments[-1].append(i)
    cap = cv2.VideoCapture(str(source))
    for run,ids in enumerate(segments,1):
        samples = []
        for index in np.linspace(0,len(ids)*.82-1,15).astype(int):
            frame = ids[index]
            cap.set(cv2.CAP_PROP_POS_FRAMES,frame)
            ok,image = cap.read()
            if not ok:
                continue
            sample_id = '{}_run{}_f{}'.format(source.stem,run,frame)
            data = cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,95])[1].tobytes()
            (OUT/'images'/(sample_id+'.jpg')).write_bytes(data)
            samples.append(dict(id=sample_id,source=video_name,run=run,frame=frame,
                                image='data:image/jpeg;base64,'+base64.b64encode(data).decode('ascii')))
        groups.append(samples)
    cap.release()
samples = [group[i] for i in range(15) for group in groups if i<len(group)]
(OUT/'manifest.json').write_text(json.dumps([{k:v for k,v in s.items() if k!='image'}
                                            for s in samples],ensure_ascii=False,indent=2),encoding='utf-8')
template = r'''<!doctype html><html lang="zh-Hant"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>彎道目標點校正</title><style>
body{font:18px system-ui;background:#f3f5f7;color:#162235;max-width:920px;margin:25px auto;padding:0 18px}
h1{font-size:27px}p{line-height:1.65}canvas{width:min(100%,800px);height:auto;border:2px solid #52677f;cursor:crosshair;background:#111;display:block}
button{font:inherit;padding:12px 18px;margin:12px 8px 8px 0;border:1px solid #a5b1be;border-radius:8px;background:white;cursor:pointer}
.primary{background:#145dcc;color:white}.note{font-size:15px;color:#48556b}#status{font-weight:600;min-height:30px}#meta{font-size:15px;margin:10px 0}
</style><h1>彎道目標點校正</h1>
<p>在<b>黃色橫線的高度</b>，點兩條黑色車道邊界之間的正中央。每張點一次，會自動換下一張。車子偏左、偏右時，也都點車道中央。</p>
<p class="note">若有一側邊界完全看不到、中央已在畫面外，或有人／物品擋住車道，請按「略過」。這是標註照片，不會控制車子。</p>
<div id="status"></div><div id="meta"></div><canvas id="canvas" width="320" height="240"></canvas>
<button onclick="move(-1)">上一張</button><button onclick="skip()">略過這張</button><button onclick="move(1)">下一張</button>
<button class="primary" onclick="download()">下載標註結果</button>
<p class="note">進度會存在這台電腦的瀏覽器；完成後下載 JSON 檔，傳回對話即可。綠點是你已標的中央位置。</p>
<script>
const items=__DATA__, key='curve-review-reset-after-rejection-20261025';let saved={labels:{},skipped:[],index:0};
try{Object.assign(saved,JSON.parse(localStorage.getItem(key)||'{}'))}catch(e){}
let index=Math.max(0,Math.min(items.length-1,saved.index||0));const canvas=document.getElementById('canvas'),ctx=canvas.getContext('2d');let photo=new Image();
function persist(){saved.index=index;try{localStorage.setItem(key,JSON.stringify(saved))}catch(e){}}
function draw(){ctx.drawImage(photo,0,0,320,240);ctx.strokeStyle='#ffe247';ctx.lineWidth=1;ctx.setLineDash([5,4]);ctx.beginPath();ctx.moveTo(0,120);ctx.lineTo(320,120);ctx.stroke();ctx.setLineDash([]);let p=saved.labels[items[index].id];if(p){ctx.strokeStyle='#00ff77';ctx.lineWidth=2;ctx.beginPath();ctx.arc(p.x,120,6,0,Math.PI*2);ctx.stroke()}}
function render(){let item=items[index];document.getElementById('status').textContent=`第 ${index+1} / ${items.length} 張｜已標 ${Object.keys(saved.labels).length} 張｜略過 ${saved.skipped.length} 張`;
document.getElementById('meta').textContent=`${item.source} · 第 ${item.run} 次行駛 · 畫面 ${item.frame}`;photo=new Image();photo.onload=draw;photo.src=item.image;persist()}
function move(n){index=Math.max(0,Math.min(items.length-1,index+n));render()}
function skip(){let id=items[index].id;delete saved.labels[id];if(!saved.skipped.includes(id))saved.skipped.push(id);if(index<items.length-1)index++;render()}
canvas.onclick=e=>{let r=canvas.getBoundingClientRect(),x=Math.max(0,Math.min(319,Math.round((e.clientX-r.left)*320/r.width)));let id=items[index].id;saved.labels[id]={x,y:120};saved.skipped=saved.skipped.filter(s=>s!==id);persist();draw();setTimeout(()=>{if(index<items.length-1){index++;render()}else{document.getElementById('status').textContent+='｜已到最後一張，請下載標註結果'}},180)};
function download(){persist();let data={schema:'car-target-review-v1',items:Object.entries(saved.labels).map(([id,p])=>({id,...p})),skipped:saved.skipped};let u=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));let a=document.createElement('a');a.href=u;a.download='curve_target_labels_20261025.json';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000)}
render();
</script></html>'''
(OUT/'label_curves.html').write_text(template.replace('__DATA__',json.dumps(samples)),encoding='utf-8')
print('Prepared',len(samples),'photos from',len(groups),'driving attempts')
print(OUT/'label_curves.html')
