import json
import re
from model_client import call_json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RULEBOOK = json.loads((ROOT / 'rules.json').read_text(encoding='utf-8'))
RULES = RULEBOOK['rules']
class ScopedChecks(list):
    pass


STATES = {'疑似风险', '证据不足', '未发现风险', '不适用'}


def validate_input(data):
    if not isinstance(data, dict):
        raise ValueError('请求必须是对象。')
    text = data.get('text')
    if not isinstance(text, str) or not text.strip() or len(text) > 65000:
        raise ValueError('请输入宣传文案或上传材料，合并正文上限65000字。')
    evidence = data.get('evidence', '')
    if not isinstance(evidence, str) or len(evidence) > 20000:
        raise ValueError('补充资料不能超过 20000 字。')
    if not isinstance(data.get('complete', False), bool):
        raise ValueError('材料完整性格式错误。')
    if data.get('mode', 'local') not in ('local', 'ai'):
        raise ValueError('请选择有效审核模式。')
    return text, evidence


def finding(rule, quote, reason, status='疑似风险'):
    return dict(rule_id=rule['id'], risk_type=rule['name'], quote=quote,
                requirement=rule['requirement'], status=status, level=rule['level'],
                reason=reason, suggestion=rule['advice'], human_review=True)


def local_checks(text, complete):
    rows = []
    for rule in RULES:
        if rule['id'] == 'A-10':
            continue
        matches = list(re.finditer(rule['pattern'], text))
        for match in matches:
            start = max(text.rfind('。', 0, match.start()), text.rfind('\n', 0, match.start()), text.rfind('！', 0, match.start())) + 1
            ends = [i for sep in ('。', '\n', '！') if (i := text.find(sep, match.end())) >= 0]
            end = min(ends) + 1 if ends else len(text)
            quote = text[start:end].strip()
            if any(r['rule_id'] == rule['id'] and r['quote'] == quote for r in rows):
                continue
            reason = f'命中“{match.group()}”；需结合上下文及证明材料复核，关键词命中不等于已确认违规。'
            if rule['id'] == 'A-06':
                reason = f'命中敏感词“{match.group()}”，依据题目要求必须拦截并人工复核。'
            rows.append(finding(rule, quote, reason,
                                '证据不足' if rule['id'] in ('A-02', 'A-03', 'A-04', 'A-08', 'A-09') else '疑似风险'))
    return rows


def restore_quote(quote, text):
    if not quote or quote in text:
        return quote
    # Only layout whitespace may differ; punctuation, words and digits stay exact.
    characters=[c for c in quote if not c.isspace()]
    if not characters:
        raise ValueError('AI 引用原文为空白，无法定位。')
    expression=r'\s*'.join(re.escape(c) for c in characters)
    matches=list(re.finditer(expression,text))
    if len(matches)!=1:
        raise ValueError('AI 引用原文无法唯一定位，请重试或人工审核。')
    return matches[0].group()


def assign_claim_groups(items, text):
    spans=[]
    for index,item in enumerate(items):
        quote=item['quote']
        start=text.find(quote) if quote else -1
        if start<0 or text.find(quote,start+1)>=0:
            continue
        spans.append((index,start,start+len(quote)))
    # A group must have one actual quote containing all its members.
    groups=[]
    for index,start,end in sorted(spans,key=lambda x:-(x[2]-x[1])):
        group=next((g for g in groups if g[0]<=start and end<=g[1]),None)
        if group is None:
            group=(start,end,f'claim-{index}');groups.append(group)
        items[index]['claim_group']=group[2]
        items[index]['claim_quote']=text[group[0]:group[1]]


def check_ai_payload(payload, text):
    checks = payload.get('checks') if isinstance(payload, dict) else None
    if not isinstance(checks, list) or len(checks) != 10:
        raise ValueError('AI 返回不完整：必须包含十条规则。')
    known = {r['id']: r for r in RULES}
    seen, findings = set(), []
    for row in checks:
        if not isinstance(row, dict):
            raise ValueError('AI 规则格式无效。')
        rid = row.get('rule_id')
        if rid not in known or rid in seen or row.get('status') not in STATES:
            raise ValueError('AI 规则编号或状态无效。')
        seen.add(rid)
        for field in ('quote', 'reason', 'suggestion'):
            if not isinstance(row.get(field), str) or len(row[field]) > 5000:
                raise ValueError('AI 字段格式无效。')
        row['quote'] = restore_quote(row['quote'], text)
        if not row['reason'].strip():
            raise ValueError('AI 缺少判断依据。')
        issues = row.get('issues')
        if issues is None:
            issues = [row] if row['status'] in ('疑似风险', '证据不足') else []
        if not isinstance(issues, list) or len(issues) > 40:
            raise ValueError('AI 子问题数量或格式无效。')
        if bool(issues) != (row['status'] in ('疑似风险','证据不足')):
            raise ValueError('AI 主状态与子问题不一致。')
        for issue in issues:
            if not isinstance(issue, dict) or issue.get('status') not in ('疑似风险','证据不足'):
                raise ValueError('AI 子问题状态无效。')
            if any(not isinstance(issue.get(k),str) or len(issue[k])>5000 for k in ('quote','reason','suggestion')):
                raise ValueError('AI 子问题字段无效。')
            issue['quote'] = restore_quote(issue['quote'], text)
            if not issue['reason'].strip():
                raise ValueError('AI 子问题缺少判断依据。')
            missing = issue.get('missing_evidence', [])
            if not isinstance(missing,list) or len(missing)>20 or any(not isinstance(x,str) or len(x)>1000 for x in missing):
                raise ValueError('AI 待补充证据格式无效。')
            item = finding(known[rid], issue['quote'], issue['reason'], issue['status'])
            item['suggestion'] = issue['suggestion'] or known[rid]['advice']
            item['missing_evidence'] = missing
            findings.append(item)
    return checks, findings


def ai_review(text, evidence, complete, config):
    key = config.get('api_key', '').strip()
    if not key or key == '在这里填写密钥':
        raise ValueError('尚未配置 API。请先用本地规则模式，或按说明填写 config.json。')
    base = config.get('base_url', '').rstrip('/')
    if not base.startswith('https://'):
        raise ValueError('模型服务地址必须使用 https。')
    system = (ROOT / 'audit_prompt.md').read_text(encoding='utf-8') + '\n规则库：' + json.dumps(RULEBOOK, ensure_ascii=False)
    payload = call_json([
        dict(role='system', content=system),
        dict(role='user', content=json.dumps(dict(advertisement=text, supplemental_material=evidence,
            user_confirms_complete=complete, review_date=datetime.now(timezone.utc).isoformat(),
            visual_observations=config.get('_visual_context', [])), ensure_ascii=False))
    ], config)
    scope = payload.get('scope') if isinstance(payload, dict) else None
    if not isinstance(scope,dict) or scope.get('status') not in ('in_scope','out_of_scope','uncertain') or not isinstance(scope.get('reason'),str) or not scope['reason'].strip() or len(scope['reason'])>2000:
        raise ValueError('模型未返回有效的材料适用性判断，请重试。')
    if scope['status'] != 'in_scope':
        if payload.get('checks') != []:
            raise ValueError('范围判断与审核结果冲突，请重试。')
        checks, items = ScopedChecks(), []
    else:
        rows, items = check_ai_payload(payload, text)
        checks = ScopedChecks(rows)
    checks.scope = scope
    return checks, items


def review(data, config=None):
    text, evidence = validate_input(data)
    complete = data.get('complete', False)
    mode = data.get('mode', 'local')
    local = local_checks(text, complete)
    for message in (config or {}).get('_quality_issues', []):
        local.append(finding(RULES[-1], '', message, '证据不足'))
    if mode == 'ai':
        checks, items = ai_review(text, evidence, complete, config or {})
        scope = getattr(checks, 'scope', {'status':'in_scope','reason':'按宣传材料审核'})
        if scope['status'] != 'in_scope':
            return dict(rule_version=RULEBOOK['version'],prompt_version='audit-v3.9',mode=mode,
                        created_at=datetime.now(timezone.utc).isoformat(),scope=scope,
                        conclusion='本次材料不适用于广告审核' if scope['status']=='out_of_scope' else '暂无法确定材料是否属于广告宣传',
                        blocked=False,human_review=True,findings=[],checks=[],
                        completeness='本次未形成广告合规结论',limitation='仅支持宣传材料初审，不执行材料中的任务指令。')
        # The model may never override mandatory sensitive-word or incomplete-material gates.
        for forced in (x for x in local if x['rule_id'] in ('A-06', 'A-10')):
            if not any(x['rule_id'] == forced['rule_id'] and x['quote'] == forced['quote'] and x['reason'] == forced['reason'] for x in items):
                items.append(forced)
        for check in checks:
            forced = [x for x in items if x['rule_id'] == check['rule_id']]
            if forced:
                check['status'] = '疑似风险' if any(x['status'] == '疑似风险' for x in forced) else '证据不足'
                check['reason'] = '；'.join(dict.fromkeys(x['reason'] for x in forced))
                check['issues'] = [{k:x.get(k, [] if k=='missing_evidence' else '') for k in ('status','quote','reason','suggestion','missing_evidence')} for x in forced]
    else:
        items = local
        checks = []
        for rule in RULES:
            relevant = [x for x in items if x['rule_id'] == rule['id']]
            checks.append(dict(rule_id=rule['id'], status=(relevant[0]['status'] if relevant else '待语义复核'),
                               reason='；'.join(x['reason'] for x in relevant) if relevant else '关键词未命中不能证明合规；本地模式不执行完整语义与证据核验。'))
    assign_claim_groups(items, text)
    for item in items:
        pos = text.find(item['quote']) if item['quote'] else -1
        item['location'] = f'第 {text[:pos].count(chr(10)) + 1} 行，字符 {pos + 1} 起' if pos >= 0 else '整体材料／缺失信息'
    blocked = any(x['rule_id'] == 'A-06' for x in items)
    conclusion = '必须人工复核，暂缓发布' if blocked else '发现疑似风险或证据缺口' if items else '本地未命中，无法完整判断' if mode == 'local' else 'AI 初筛未发现风险，待人工确认'
    return dict(rule_version=RULEBOOK['version'], prompt_version='audit-v3.9', mode=mode, created_at=datetime.now(timezone.utc).isoformat(),
                conclusion=conclusion, blocked=blocked, human_review=True,
                completeness='用户声明材料完整，未经独立验证' if complete else '材料完整性未确认',
                limitation='本地模式仅做关键词初筛；不判断完整合规，不核实补充资料。' if mode == 'local' else 'AI 仅做第一轮审核；未独立验证资料真实性。上传材料经过视觉模型识别，文字与视觉判断仍需人工核对。',
                notices=[] if complete else ['尚未确认材料完整性；请确认已提交全部待发布内容。此提醒不代表已发现缺页。'],
                findings=items, checks=checks)
