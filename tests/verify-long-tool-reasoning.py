#!/usr/bin/env python3
"""Opt-in sequential long-context tool-reasoning check; Python standard library only.

Uses generated data, preserves genuine model reasoning, independently scores plans,
and compares reminder on/off. No deployment changes or model-generated commands.
"""
import argparse
import copy
import hashlib
import itertools
import json
import os
import random
import re
import time
import urllib.request
from pathlib import Path

RULES=('Choose exactly three items, with total cost at most the current budget. '
       'A and C are incompatible. E requires F. Include at least one B or G. '
       'Maximize total value, breaking ties by alphabetically smallest sorted ids. '
       'Submit sorted ids, total cost and value using submit_plan. ')
INSTRUCTION=('Process batches 0 through 5 in order, one submit_plan call per turn. '
             'Each accepted submission returns NEW inventory for the next batch. '
             'Use only that batch\'s inventory, not archived records or earlier prices. '
             'Recalculate and check all constraints after each tool result. '+RULES)
TOOLS=[{'type':'function','function':{'name':'submit_plan','description':'Validate one batch plan and return the next batch inventory.',
       'parameters':{'type':'object','properties':{'batch_id':{'type':'integer'},'ids':{'type':'array','items':{'type':'string'}},
       'total_cost':{'type':'integer'},'total_value':{'type':'integer'}},
       'required':['batch_id','ids','total_cost','total_value'],'additionalProperties':False}}}]
PARAMS={'temperature':0,'top_p':1,'seed':739,'max_tokens':8192,'reasoning_effort':'max',
        'chat_template_kwargs':{'enable_thinking':True,'clear_thinking':False,'reasoning_effort':'max'}}


def log(event, **kw):
    print(json.dumps({'time_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                      'event': event, **kw}), flush=True)

def post(path, payload, timeout=1800):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)

def make_archive(lines):
    return ''.join(
        f'Archived record {i:06d}: region={i % 17}; invoice={100000 + i}; '
        f'gross_eur={100 + (i * 37) % 1900}; credit_eur={(i * 13) % 90}; '
        f'discount_percent={i % 31}; status=closed; evidence=verified; '
        'this record is already processed and requires no further action.\n'
        for i in range(lines))

def run(label, messages, tools=None, expected='new-batch', parameters=None):
    saved = OUT / (label + '.result.json')
    payload = {'model': MODEL, 'messages': messages, 'temperature': 0,
               'max_tokens': 4096, 'stream': True,
               'stream_options': {'include_usage': True},
               'chat_template_kwargs': {'enable_thinking': True}}
    if tools is not None:
        payload['tools'] = tools
        if tools:
            payload['tool_choice'] = 'auto'
    if parameters:
        payload.update(parameters)
    token_payload = {'model': MODEL, 'messages': messages,
                     'chat_template_kwargs': payload['chat_template_kwargs']}
    if tools is not None:
        token_payload['tools'] = tools
    prompt_count = post('/tokenize', token_payload)['count']
    (OUT / (label + '.request-metadata.json')).write_text(json.dumps({
        'label': label, 'prompt_tokens_tokenize': prompt_count,
        'message_roles': [m['role'] for m in messages],
        'tool_names': [t['function']['name'] for t in tools] if tools else [],
        'payload_sha256': hashlib.sha256(json.dumps(payload).encode()).hexdigest(),
        'parameters': {k: v for k, v in payload.items() if k not in ('messages', 'tools')}}, indent=2))
    log('request_start', label=label, prompt_tokens=prompt_count, expected=expected)
    started = time.monotonic()
    first = None
    reason, content, tool_deltas = [], [], {}
    usage, finish = {}, None
    raw_path = OUT / (label + '.sse.jsonl')
    req = urllib.request.Request(BASE + '/v1/chat/completions',
                                 data=json.dumps(payload).encode(),
                                 headers=HEADERS)
    with raw_path.open('w') as raw, urllib.request.urlopen(req, timeout=1800) as response:
        for line in response:
            if not line.startswith(b'data: '):
                continue
            data = line[6:].strip()
            if data == b'[DONE]':
                break
            obj = json.loads(data)
            raw.write(json.dumps(obj) + '\n')
            raw.flush()
            if 'error' in obj:
                raise RuntimeError(obj['error'])
            if obj.get('usage'):
                usage = obj['usage']
            for choice in obj.get('choices', []):
                d = choice.get('delta', {})
                rt = d.get('reasoning') or d.get('reasoning_content') or ''
                ct = d.get('content') or ''
                if (rt or ct or d.get('tool_calls')) and first is None:
                    first = time.monotonic() - started
                    log('first_output', label=label, seconds=round(first, 2))
                reason.append(rt)
                content.append(ct)
                for tc in d.get('tool_calls') or []:
                    x = tool_deltas.setdefault(tc['index'], {'id': '', 'type': 'function',
                                                          'function': {'name': '', 'arguments': ''}})
                    if tc.get('id'):
                        x['id'] = tc['id']
                    f = tc.get('function') or {}
                    x['function']['name'] += f.get('name') or ''
                    x['function']['arguments'] += f.get('arguments') or ''
                if choice.get('finish_reason'):
                    finish = choice['finish_reason']
    reason, content = ''.join(reason), ''.join(content)
    calls = [tool_deltas[k] for k in sorted(tool_deltas)]
    output = {'label': label, 'prompt_tokens_tokenize': prompt_count,
              'duration_seconds': round(time.monotonic() - started, 3), 'ttft_seconds': first,
              'reasoning_chars': len(reason), 'content_chars': len(content),
              'reasoning': reason, 'content': content, 'tool_calls': calls,
              'finish_reason': finish, 'usage': usage,
              'raw_tool_tags': bool(re.search(r'<\/?(?:tool_call|arg_key|arg_value)>', content)),
              'raw_think_tags': '<think>' in content or '</think>' in content}
    (OUT / (label + '.result.json')).write_text(json.dumps(output, indent=2))
    concise = {k: v for k, v in output.items() if k not in ('reasoning', 'content')}
    with (OUT / 'results.jsonl').open('a') as r:
        r.write(json.dumps(concise) + '\n')
    log('request_done', **concise)
    if finish == 'length':
        log('warning_output_truncated', label=label)
    return output

def batches():
    rng=random.Random(20261009739)
    out=[]
    for n in range(6):
        items=[{'id':name,'cost':rng.randint(9,31),'value':rng.randint(17,77)} for name in 'ABCDEFGH']
        out.append({'batch_id':n,'budget':rng.randint(57,70),'items':items,'rules':RULES})
    return out

BATCHES=batches()


def optimum(b):
    valid=[]
    for xs in itertools.combinations(b['items'],3):
        ids=[x['id'] for x in xs];s=set(ids);cost=sum(x['cost'] for x in xs)
        if cost<=b['budget'] and not {'A','C'}<=s and ('E' not in s or 'F' in s) and {'B','G'}&s:
            valid.append({'batch_id':b['batch_id'],'ids':ids,'total_cost':cost,'total_value':sum(x['value'] for x in xs)})
    if not valid:raise RuntimeError('No feasible plan in fixture')
    return sorted(valid,key=lambda x:(-x['total_value'],x['ids']))[0]

def initial(archive):
    return [{'role':'system','content':'You are a careful procurement assistant. Validate current data before submitting.'},
            {'role':'user','content':'<archive>\n'+archive+'\n</archive>\n'+INSTRUCTION+'\nCurrent batch:\n'+json.dumps(BATCHES[0])}]

def tokenize(ms,reminder=True):
    kw=copy.deepcopy(PARAMS['chat_template_kwargs']);kw['tool_reasoning_reminder']=reminder
    return post('/tokenize',{'model':MODEL,'messages':ms,'tools':TOOLS,'chat_template_kwargs':kw})

def verify(label,ms,n,reminder=True):
    params=copy.deepcopy(PARAMS);params['chat_template_kwargs']['tool_reasoning_reminder']=reminder
    r=run(label,ms,tools=TOOLS,parameters=params,expected='new-batch')
    picked=None
    if len(r['tool_calls'])==1 and r['tool_calls'][0]['function']['name']=='submit_plan':
        try:picked=json.loads(r['tool_calls'][0]['function']['arguments'])
        except ValueError:pass
    expected=optimum(BATCHES[n])
    r.update(expected=expected,actual=picked,correct_plan=picked==expected,reminder_enabled=reminder,
             preserved_reasoning_blocks=sum(bool(m.get('reasoning')) for m in ms))
    r['count_matches_usage']=r['prompt_tokens_tokenize']==r['usage']['prompt_tokens']
    r['separate_reasoning']=bool(r['reasoning'].strip()) and not r['raw_think_tags'] and not r['raw_tool_tags']
    r['passed']=r['correct_plan'] and r['separate_reasoning'] and r['count_matches_usage'] and r['finish_reason']=='tool_calls'
    (OUT/(label+'.result.json')).write_text(json.dumps(r,indent=2))
    log('independent_score',label=label,passed=r['passed'],correct_plan=r['correct_plan'],expected=expected,actual=picked,
            reasoning_tokens=r['usage'].get('completion_tokens_details'),prompt_tokens=r['usage']['prompt_tokens'])
    if reminder and not r['passed']:
        raise RuntimeError('Reminder-on case failed; response retained in output directory')
    return r

def append(ms,r,n):
    call=copy.deepcopy(r['tool_calls'][0])
    ms += [{'role':'assistant','content':r['content'] or None,'reasoning':r['reasoning'],'tool_calls':[call]},
           {'role':'tool','tool_call_id':call['id'],'content':json.dumps({'accepted_batch':n,'status':'accepted','next_batch':BATCHES[n+1]})}]

def enlarge(history,target):
    overhead=tokenize(initial('')+history)['count'];lines=10800
    for _ in range(6):
        ms=initial(make_archive(lines))+copy.deepcopy(history)
        count=tokenize(ms)['count'];log('long_sizing',target=target,actual=count,archive_lines=lines)
        if abs(count-target)<500:return ms
        lines=max(1,round(lines*(target-overhead)/max(1,count-overhead)))
    raise RuntimeError('Prompt sizing failed')

def main():
    global BASE, MODEL, OUT, HEADERS
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url',default='http://127.0.0.1:8000')
    parser.add_argument('--model',default='glm53')
    parser.add_argument('--targets',type=int,nargs='+',required=True,
                        help='Approximate input lengths, e.g. 900000 1020000')
    parser.add_argument('--output-dir',type=Path,required=True,
                        help='A new/empty output directory; full generated responses are saved here')
    args=parser.parse_args()
    if any(t < 10000 for t in args.targets) or len(set(args.targets)) != len(args.targets):
        parser.error('Targets must be unique integers >= 10000')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('Output directory must be new or empty; results are never silently reused')
    BASE=args.base_url.rstrip('/');MODEL=args.model;OUT=args.output_dir
    OUT.mkdir(parents=True,exist_ok=True)
    HEADERS={'Content-Type':'application/json'}
    if os.environ.get('OPENAI_API_KEY'):
        HEADERS['Authorization']='Bearer '+os.environ['OPENAI_API_KEY']
    # Leave actual model/context compatibility to the server; reject errors explicitly.
    (OUT/'fixtures.json').write_text(json.dumps({'batches':BATCHES,
                         'expected':[optimum(b) for b in BATCHES]},indent=2))
    (OUT/'provenance.json').write_text(json.dumps({'targets':args.targets,
        'model':MODEL,'parameters':PARAMS,'deployment_changed':False,
        'method':'Three genuine short seed rounds, synthetic archive, fresh tool inventories',
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2))
    log('experiment_start',targets=args.targets,model=MODEL)
    ms=initial('')
    for batch_id in range(3):
        result=verify('short-batch'+str(batch_id),ms,batch_id)
        append(ms,result,batch_id)
    history=copy.deepcopy(ms[2:])
    (OUT/'genuine-short-history.json').write_text(json.dumps(history,indent=2))
    for index,target in enumerate(args.targets):
        ms=enlarge(history,target)
        pair={}
        # Reversed order at alternate sizes; on/off share complete input history.
        for reminder in ([False,True] if index%2==0 else [True,False]):
            label=str(target)+'-batch3-reminder-'+('on' if reminder else 'off')
            pair[reminder]=verify(label,ms,3,reminder)
        append(ms,pair[True],3)
        result=verify(str(target)+'-batch4-reminder-on',ms,4)
        append(ms,result,4)
        verify(str(target)+'-batch5-reminder-on',ms,5)
        log('target_done',target=target,reminder_off_control_passed=pair[False]['passed'])
    log('experiment_complete')


if __name__=='__main__':
    main()
