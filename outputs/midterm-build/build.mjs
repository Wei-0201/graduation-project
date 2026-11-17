import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL, fileURLToPath } from 'node:url';
import { Presentation, PresentationFile } from '@oai/artifact-tool';

const dir=path.dirname(fileURLToPath(import.meta.url));
const root=path.resolve(dir,'../..');
const deck=Presentation.create({slideSize:{width:1280,height:720}});
const font='Microsoft JhengHei', ink='#222920', gray='#505951', paper='#F0EDE5', green='#24594B', pale='#E1E8DC', white='#FFFFFF';
const none={fill:'none',width:0};
function shape(s,x,y,w,h,fill='none',geometry='rect'){
  return s.shapes.add({geometry,position:{left:x,top:y,width:w,height:h},fill,line:none});
}
function t(s,txt,x,y,w,h,size=24,{color=ink,bold=false,align='left',fill='none'}={}){
  const a=shape(s,x,y,w,h,fill,'textbox');
  a.text=txt;
  a.text.style={typeface:font,fontSize:size,bold,color,alignment:align,verticalAlignment:'middle',insets:{left:0,right:0,top:0,bottom:0}};
  return a;
}
function slide(title, section, notes=''){
  const s=deck.slides.add();s.background.fill=paper;
  t(s,section,64,36,1120,24,15,{color:green,bold:true});
  t(s,title,64,69,1150,72,44,{bold:true});
  shape(s,64,155,1152,2,'#D1D3C7');
  t(s,String(deck.slides.items.length).padStart(2,'0'),1160,667,56,24,15,{align:'right',color:gray});
  if(notes) s.speakerNotes.text=notes;
  return s;
}
function list(s, rows, {x=64,y=192,w=1152,step=105}={}){
  rows.forEach(([head,body],i)=>{
    const yy=y+i*step;
    t(s,String(i+1).padStart(2,'0'),x,yy,56,35,19,{color:green,bold:true});
    t(s,head,x+65,yy,w-65,35,27,{bold:true});
    t(s,body,x+65,yy+39,w-65,51,21,{color:gray});
  });
}
async function photo(s,id,x,y,w,h){
  s.images.add({blob:new Uint8Array(await fs.readFile(path.join(dir,'frame'+id+'.jpg'))),contentType:'image/jpeg',fit:'cover',alt:'室內膠帶車道的車載攝影機影格 '+id,position:{left:x,top:y,width:w,height:h}});
}
function table(s,values,widths,y=190,height=376){
  const o=s.tables.add({rows:values.length,columns:values[0].length,left:64,top:y,width:1152,height,columnWidths:widths,values});
  values.forEach((row,i)=>row.forEach((_,j)=>{
    const cell=o.getCell(i,j);
    cell.fill=i===0?green:(i%2?'#FAF8F2':'#E6EBE1');
    cell.text.style={typeface:font,fontSize:21,color:i===0?white:ink,bold:i===0||j===0};
  }));
  return o;
}
const refs=[
 ['Bojarski 等人（2016）','End to End Learning for Self-Driving Cars','https://arxiv.org/abs/1604.07316'],
 ['He 等人（2016）','Deep Residual Learning for Image Recognition','https://arxiv.org/abs/1512.03385'],
 ['Smith（2007）','An Overview of the Tesseract OCR Engine','https://research.google/pubs/an-overview-of-the-tesseract-ocr-engine/'],
];
const csv='資料來源：entry_exit_test.avi.csv（1,035 筆）及 right_entry_v3.avi.csv（3,900 筆）。AUTO 時長以 traffic_state 非 MANUAL 的相鄰 result_mono 間隔加總。自動行駛按 DRIVE 指令估算。整段以末張 capture_mono 減首張。';

{
const s=deck.slides.add();s.background.fill=paper;
await photo(s,255,735,0,545,720);
shape(s,64,123,54,5,green);
t(s,'專題期中報告',64,71,610,35,20,{color:green,bold:true});
t(s,'基於 Jetson Nano 的\n視覺車道置中小車',64,158,650,185,58,{bold:true});
t(s,'車載影像推論、進出彎控制與實測紀錄',64,364,640,64,24,{color:gray});
t(s,'組員　＿＿＿＿＿＿　　指導老師　＿＿＿＿＿＿',64,600,650,36,19,{color:gray});
s.speakerNotes.text='封面影像來源：專題車載實錄 right_entry_v2.avi，影格 255。影像為攝影機視角。';
}
{
const s=slide('目錄','期中報告');
const names=['研究動機','研究目標','文獻回顧與探討','系統情境與架設','系統架構圖','硬體與軟體','實體展示與數據分析','未來展望'];
names.forEach((n,i)=>{const x=i<4?64:640,y=207+(i%4)*90;t(s,'0'+(i+1),x,y,61,40,22,{color:green,bold:true});t(s,n,x+70,y,495,40,30,{bold:true});});
}
{
const s=slide('研究動機','01 研究動機','從室內低速實驗出發，觀察提前右切、出彎回正和車道線遺失。圖為 right_entry_v2.avi 影格 20。');
list(s,[['從車載影像取得轉向','讓小車沿室內膠帶車道自主行駛。'],['彎道需要判斷轉向時機','提前右切可能壓線，出彎收方向過慢也可能偏離車道。'],['用紀錄追查停車原因','將影像、模型輸出與指令放在同一時間軸比對。']],{w:660,step:130});
await photo(s,20,773,194,443,332);
t(s,'室內測試場地的車載畫面',773,535,443,32,17,{color:gray});
}
{
const s=slide('研究目標','02 研究目標');
table(s,[['實驗問題','系統做法','驗證重點'],['小車在車道內置中','以影像模型預測目標點','直線與彎道的行駛軌跡'],['右彎過早切入','以近處車道線修正右轉幅度','修正何時啟用、是否壓線'],['出彎後仍持續轉向','依近處車道回直收小轉向','出彎狀態及回正時機'],['辨識遺失與路牌事件','設定停車條件並記錄狀態','車道遺失、STOP／P 行為']], [250,445,457],192,370);
}
{
const s=slide('文獻回顧與探討','03 文獻回顧與探討',refs.map(r=>r.join(' ')).join('\n'));
list(s,[['影像轉向｜Bojarski 等人（2016）','以攝影機影像學習轉向。本專題預測車道目標點，並保留控制規則。'],['殘差網路｜He 等人（2016）','以殘差連接訓練深層網路。本專題以 ResNet-18 作為目標點模型。'],['文字辨識｜Smith（2007）','介紹 Tesseract OCR 架構。本專題先篩選路牌候選區，再辨識 STOP 與 P。']],{step:132});
}
{
const s=slide('系統情境與架設','04 系統情境與架設','場地影像：right_entry_v2.avi，影格 20。實驗為室內低速膠帶車道。路牌功能須另以實體牌驗證。');
await photo(s,20,64,190,552,414);
list(s,[['室內低速車道','在地面以雙側黑色膠帶標出直線與彎道。'],['小車與操作端','前視攝影機接 Jetson，電腦透過瀏覽器選擇手動或 AUTO。'],['紀錄方式','測試時保留車載影像與逐影格 CSV。']],{x:654,y:193,w:562,step:128});
}
{
const s=slide('系統架構圖','05 系統架構圖','依專題 jetson_v4_traffic_signs.py、right_entry_guard.py、observed_exit_easing.py 與 sign_parking_control.py 整理。');
function block(x,y,w,h,head,body){shape(s,x,y,w,h,pale);t(s,head,x+12,y+9,w-24,36,26,{bold:true,align:'center'});t(s,body,x+12,y+46,w-24,h-51,19,{align:'center',color:gray});}
block(64,229,214,117,'前視攝影機','擷取車道與路牌');
block(328,203,360,170,'Jetson Nano','ResNet-18 目標點推論\nOpenCV 車道線／Tesseract OCR\n轉向修正與停車判斷');
block(738,229,214,117,'ESP32','接收串列指令');
block(1002,229,214,117,'馬達控制','執行速度與轉向');
for(const x of [280,690,954])t(s,'→',x,262,46,42,33,{align:'center',color:green});
block(328,464,360,101,'瀏覽器操作介面','AUTO／手動／STOP');
block(738,464,478,101,'實驗紀錄','AVI 影像與 CSV 控制資料');
t(s,'↕',480,383,55,60,32,{align:'center',color:green});
t(s,'↘',695,378,66,65,37,{align:'center',color:green});
}
{
const s=slide('硬體與軟體','06 硬體與軟體','Jetson Nano 官方文件：https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit');
shape(s,64,190,556,388,pale);shape(s,644,190,572,388,'#F8F6EF');
t(s,'硬體',90,213,495,50,32,{bold:true});t(s,'軟體',670,213,500,50,32,{bold:true});
const cols=[
 ['Jetson Nano｜車載運算','USB 攝影機｜取得前視影像','ESP32｜接收控制指令','馬達與編碼器｜驅動與輪速回報'],
 ['PyTorch／torchvision｜模型推論','OpenCV｜影像前處理與車道線','Tesseract｜STOP／P 文字辨識','Flask／pyserial｜控制頁與串列通訊']
];
cols.forEach((arr,j)=>arr.forEach((v,i)=>t(s,v,j?670:90,279+i*60,510,45,23,{color:gray})));
}
{
const s=slide('車載畫面與展示流程','07 實體展示','影格來源為較早的 right_entry_v2.avi（影格 20、255、495），與後兩頁的 CSV 統計分開。畫面展示場地與失線情境，不能據此推算兩份 CSV 的事件時間。');
for(const [i,id,label] of [[0,20,'直線接近右彎'],[1,255,'近處車道線仍可見'],[2,495,'車道偏到畫面左側']]){
  const x=64+i*391;
  await photo(s,id,x,195,370,278);
  t(s,label,x,482,370,40,22,{bold:true});
}
t(s,'現場展示　啟動預覽 → 切換 AUTO → 觀察轉向與停車狀態 → 儲存紀錄',64,548,1150,54,23,{color:green,bold:true});
t(s,'影像來源為較早的右彎試跑，供辨識場景與偏離現象。',64,614,1150,29,17,{color:gray});
}
{
const s=slide('兩次測試的執行紀錄','07 實體展示與數據分析',csv);
table(s,[['指標','進出彎整合測試','右轉進彎 V3'],['CSV 筆數','1,035','3,900'],['整段記錄','105.0 秒','368.2 秒'],['AUTO 測試段數','2 段','4 段'],['AUTO 模式時間','25.6 秒','45.7 秒'],['自動行駛指令時間','22.0 秒','31.6 秒'],['AUTO 影像處理時間中位數','131 毫秒','155 毫秒']], [440,356,356],188,373);
t(s,'自動行駛時間是指令估算值。兩批測試的時長與段數不同。',64,595,1152,36,19,{color:gray});
}
{
const s=slide('失線停車與進出彎修正','07 實體展示與數據分析',csv+' 停車累計採相鄰 result_mono 間隔。兩份紀錄均未確認 STOP 或 P。V3 有一次 OCR 記憶體配置錯誤，發生於手動停止期間。');
table(s,[['紀錄中的事件','整合測試','右轉進彎 V3'],['出現失線停車的 AUTO 測試段','1／2','2／4'],['累計失線停車時間','2.7 秒','11.5 秒'],['右轉進彎修正狀態筆數','34 筆','27 筆'],['專用出彎修正狀態','0 筆','0 筆']], [540,306,306],190,288);
t(s,'右轉進彎修正已有介入紀錄。是否減少壓線，須同步對照軌跡。',64,506,1152,42,23,{bold:true});
t(s,'本批沒有專用出彎修正觸發，也沒有確認到 STOP 或 P。',64,553,1152,42,21,{color:gray});
}
{
const s=slide('結論與未來展望','08 未來展望');
list(s,[['期中進度','已完成車載推論、控制介面與 CSV 紀錄。兩份測試共送出約 53.6 秒自動行駛指令。'],['提高車道線辨識穩定度','優先測試反光、陰影與單側線出畫等場景，量測失線次數。'],['量化進出彎與路牌表現','固定起點與速度重複測試，比對壓線、回正位置、STOP 停車與 P 靠右結果。'],['同步輪速與影格時間','把 ESP32 輪速回報加入紀錄，區分指令時間與車輪實際轉動時間。']],{step:106});
}
{
const s=slide('參考文獻與實測來源','期中報告',refs.map(r=>r.join(' ')).join('\n'));
refs.forEach((r,i)=>{
 const y=184+i*105;t(s,r[0]+'　'+r[1],64,y,1152,39,23,{bold:true});
 t(s,r[2],64,y+43,1152,31,17,{color:green});
});
t(s,'實測紀錄',64,515,1152,37,24,{bold:true});
t(s,'entry_exit_test.avi.csv、right_entry_v3.avi.csv、right_entry_v2.avi',64,557,1152,36,19,{color:gray});
}

for(const [i,s] of deck.slides.items.entries()){
  const img=await deck.export({slide:s,format:'png',scale:1});
  await fs.writeFile(path.join(dir,`slide-${i+1}.png`),new Uint8Array(await img.arrayBuffer()));
}
const candidatePath=path.join(dir,'candidate.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
const skill='C:/Users/USER/.codex/plugins/cache/openai-primary-runtime/presentations/26.1030.10813/skills/presentations';
const {finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
await finalizePresentation({
 workspaceDir:root,candidatePath,finalPath:path.join(root,'outputs/midterm-20261104/Jetson視覺車道置中_期中報告.pptx'),
 pythonExecutable:'C:/Program Files/WindowsApps/OpenAI.CodexPrimaryRuntime.v26-1030-493-0_26.1030.493.0_x64__3k8sg7r9htsxt/dependencies/python/python.exe',
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit',
 '--require-native-table-slide','4','--require-native-table-slide','10','--require-native-table-slide','11'],
 fontPolicy:{basis:'design',families:[font]},verifyArtifactToolImport:true,
 receiptPath:path.join(dir,'validation.json')
});
