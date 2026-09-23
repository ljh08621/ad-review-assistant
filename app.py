import hashlib
import hmac
import json
import os
import secrets
import threading
import time
import urllib.error
from collections import deque
from pathlib import Path
from flask import Flask, request, jsonify, Response
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from werkzeug.exceptions import HTTPException
from reviewer import review, RULEBOOK
from materials import extract, extract_bundle, MAX_BYTES

ROOT = Path(__file__).resolve().parent
app = Flask(__name__, static_folder=None)
app.config['MAX_CONTENT_LENGTH'] = MAX_BYTES + 1024 * 100
signer = URLSafeTimedSerializer(os.environ.get('APP_SIGNING_KEY') or secrets.token_hex(32), salt='extracted-material-v1')
slots = threading.BoundedSemaphore(2)
rate_lock = threading.Lock()
recent_calls = deque()


def configuration():
    config = json.loads((ROOT/'config.example.json').read_text(encoding='utf-8'))
    path = ROOT/'config.json'
    if path.exists():
        config.update(json.loads(path.read_text(encoding='utf-8-sig')))
    mapping = {'DASHSCOPE_API_KEY':'api_key','AD_REVIEW_API_KEY':'api_key',
               'DASHSCOPE_BASE_URL':'base_url','QWEN_MODEL':'model','QWEN_VISION_MODEL':'vision_model'}
    for env, key in mapping.items():
        if os.environ.get(env): config[key] = os.environ[env]
    return config


@app.before_request
def protect():
    if request.method != 'POST': return
    origin = request.headers.get('Origin')
    if origin and origin.rstrip('/') not in (request.host_url.rstrip('/'), 'https://' + request.host):
        return jsonify(error='请从本应用页面发起请求。'), 403
    code = os.environ.get('APP_ACCESS_CODE','')
    if code and not hmac.compare_digest(hashlib.sha256(request.headers.get('X-Access-Code','').encode()).digest(), hashlib.sha256(code.encode()).digest()):
        return jsonify(error='请输入正确的试用访问码。'), 401
    now = time.monotonic()
    with rate_lock:
        while recent_calls and now-recent_calls[0]>60: recent_calls.popleft()
        if len(recent_calls)>=10:
            return jsonify(error='请求较频繁，请一分钟后再试。'), 429
        recent_calls.append(now)


@app.after_request
def headers(response):
    response.headers['Cache-Control']='no-store'
    response.headers['X-Content-Type-Options']='nosniff'
    response.headers['Content-Security-Policy']="default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; frame-ancestors 'none'; connect-src 'self'"
    return response


@app.get('/')
def index():
    return Response((ROOT/'index.html').read_text(encoding='utf-8'),content_type='text/html; charset=utf-8')


@app.get('/api/rules')
def rules(): return jsonify(RULEBOOK)


@app.get('/api/status')
def status():
    cfg = configuration()
    return jsonify(ai_configured=bool(cfg.get('api_key','').strip()),
        vision_model=cfg.get('vision_model','qwen-vl-plus'),access_required=bool(os.environ.get('APP_ACCESS_CODE')))


def with_slot(fn):
    if not slots.acquire(blocking=False):
        return jsonify(error='当前已有审核任务，请稍后再试。'),429
    try: return fn()
    finally: slots.release()


@app.post('/api/extract')
def extract_file():
    if request.form.get('consent')!='true':
        raise ValueError('请确认允许将文件发送至配置的视觉模型。')
    files=request.files.getlist('file')
    if not 1 <= len(files) <= 4 or any(not f.filename for f in files):
        raise ValueError('请选择1～4个图片或PDF文件。')
    inputs=[(f.filename,f.read(MAX_BYTES+1)) for f in files]
    if sum(len(raw) for _,raw in inputs)>MAX_BYTES:
        raise ValueError('一组材料总大小不能超过8 MB。')
    def work():
        started=time.monotonic()
        material=extract_bundle(inputs,configuration())
        return jsonify(material=material, token=signer.dumps(material), extraction_seconds=round(time.monotonic()-started,1))
    return with_slot(work)


@app.post('/api/review')
def review_text():
    if not request.is_json: return jsonify(error='请求必须使用 JSON。'),415
    data=request.get_json()
    if not isinstance(data,dict):raise ValueError('请求格式错误。')
    if data.get('mode')=='ai' and data.get('consent') is not True:
        raise ValueError('请确认允许将文案和资料发送至 AI 服务。')
    if not isinstance(data.get('text',''),str) or len(data.get('text',''))>20000:
        raise ValueError('输入正文不能超过20000字。')
    cfg=configuration()
    token=data.get('material_token')
    material=None
    if token:
        if not isinstance(token,str):raise ValueError('材料凭据无效。')
        try: material=signer.loads(token,max_age=7200)
        except (BadSignature,SignatureExpired):raise ValueError('材料识别记录已失效，请重新上传识别。')
        corrections=data.get('page_corrections')
        if corrections is not None:
            if not isinstance(corrections,list) or len(corrections)!=len(material['pages']) or any(not isinstance(x,str) or len(x)>20000 for x in corrections) or sum(map(len,corrections))>40000:
                raise ValueError('逐页修订格式无效或超过字数限制。')
            for page,corrected in zip(material['pages'],corrections):
                if corrected!=page['text']:
                    page['original_ocr_text']=page['text']
                    page['text']=corrected
                    page.setdefault('review_notes', []).append('识别文字经过人工修订，请对照原图核对；已保留原始识别版本。')
        sections=[];quality=[];visual=[]
        for page in material['pages']:
            sections.append(f"\n\n【材料序号{page['page']} · 文件 {page.get('source_filename', material['filename'])} · 第{page.get('source_page',page['page'])}页】\n"+page['text'])
            quality.extend(f"第{page['page']}页：{q}" for q in page['quality_issues'])
            visual.append(dict(page=page['page'], observations=page['visual_notes']))
        data['text']=data.get('text','')+''.join(sections)
        cfg['_quality_issues']=quality;cfg['_visual_context']=visual
    def work():
        started=time.monotonic()
        result=review(data,cfg)
        result["review_seconds"]=round(time.monotonic()-started,1)
        if material:
            for item in result['findings']:
                if item['quote']:
                    pages=[f"{p.get('source_filename',material['filename'])} · 第{p.get('source_page',p['page'])}页" for p in material['pages'] if item['quote'] in p['text']]
                    if pages: item['location']='、'.join(pages)+'（识别文本匹配，需核对原图）'
            result['notices'] = [note for page in material['pages'] for note in page.get('review_notes', [])]
            if data.get('extraction_confirmed') is not True:
                result['notices'].append('识别文字尚未人工核对，请对照原图检查。此提醒不代表图片模糊或不完整。')
            result['material']=material
            result['completeness'] += '；已处理上传文件全部 '+str(material['page_count'])+' 页'
        return jsonify(result)
    return with_slot(work)


@app.errorhandler(Exception)
def errors(exc):
    if isinstance(exc,HTTPException):
        return jsonify(error='文件或请求过大。' if exc.code==413 else '请求无效或页面不存在。'),exc.code
    if isinstance(exc,urllib.error.HTTPError):
        return jsonify(error=f'百炼返回错误（{exc.code}），请核对密钥地域、模型权限、余额与配置。本次未完成处理。'),502
    if isinstance(exc,(urllib.error.URLError,TimeoutError)):
        return jsonify(error='模型请求超时或网络不可用。本次未完成处理，请稍后重试。'),502
    if isinstance(exc,ValueError):
        return jsonify(error='配置或模型返回的 JSON 格式无效。' if isinstance(exc,json.JSONDecodeError) else str(exc)),400
    return jsonify(error='处理未完成，请检查文件或模型配置；没有生成完整审核结论。'),500


if __name__=='__main__':
    from waitress import serve
    host=os.environ.get('HOST','127.0.0.1');port=int(os.environ.get('PORT','8765'))
    if host not in ('127.0.0.1','localhost') and len(os.environ.get('APP_ACCESS_CODE',''))<12:
        raise SystemExit('对外提供服务前，请设置至少12位 APP_ACCESS_CODE 试用访问码。')
    print(f'广告审核助手已启动：http://{host}:{port}，按 Ctrl+C 停止。',flush=True)
    serve(app,host=host,port=port,threads=4,max_request_body_size=MAX_BYTES+102400,channel_timeout=600)
