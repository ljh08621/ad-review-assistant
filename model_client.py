import json
import urllib.request
from urllib.parse import urlsplit


def normalize_base_url(value):
    base = value.rstrip('/')
    parsed = urlsplit(base)
    if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError('模型服务地址必须是有效的 https 基础地址。')
    if not parsed.path and (parsed.hostname.endswith('.maas.aliyuncs.com') or parsed.hostname in ('dashscope.aliyuncs.com','dashscope-intl.aliyuncs.com','dashscope-us.aliyuncs.com')):
        base += '/compatible-mode/v1'
    if base.endswith('/chat/completions'):
        base = base[:-len('/chat/completions')]
    return base


def call_json(messages, config, vision=False):
    key = config.get('api_key', '').strip()
    if not key or key == '在这里填写密钥':
        raise ValueError('尚未配置 API，请在 config.json 中填写百炼密钥。')
    base = normalize_base_url(config.get('base_url', ''))
    model = config.get('vision_model', 'qwen-vl-plus') if vision else config.get('model', 'qwen-vl-plus')
    body = dict(model=model, messages=messages, response_format={'type':'json_object'}, stream=False, temperature=0)
    if vision and model == 'qwen3.8-flash':
        body['enable_thinking'] = False
    if not vision and 'enable_thinking' in config:
        if not isinstance(config['enable_thinking'], bool):
            raise ValueError('enable_thinking 必须为 true 或 false。')
        body['enable_thinking'] = config['enable_thinking']
    for attempt in range(2):
        request = urllib.request.Request(base + '/chat/completions', data=json.dumps(body).encode(),
            headers={'Authorization':'Bearer ' + key, 'Content-Type':'application/json'}, method='POST')
        with urllib.request.urlopen(request, timeout=120) as response:
            raw = json.load(response)
        choice = raw['choices'][0]
        if choice.get('finish_reason') not in (None, 'stop'):
            raise ValueError('模型输出被截断或未正常结束，本次没有完整结论。')
        content = choice['message']['content']
        try:
            result = json.loads(content)
            if not isinstance(result, dict):
                raise ValueError('模型未返回有效 JSON 对象。')
            return result
        except json.JSONDecodeError:
            if attempt: raise ValueError('模型连续返回无效 JSON，本次识别或审核未完成。')
            # One bounded retry; never fabricate a result from malformed output.
            body['messages'] = messages + [{'role':'user','content':'上一次返回不是有效JSON。请重新执行原任务，严格按指定字段类型返回可解析的JSON对象。不要添加Markdown。'}]
