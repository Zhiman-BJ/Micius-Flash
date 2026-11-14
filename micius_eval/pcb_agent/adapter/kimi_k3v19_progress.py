"""Preserve thinking options/history and distinguish stream bytes from complete calls."""
import copy,json
from tools.kimi_k3v19_efficiency import digest

def prepare_payload(original,model,stage,max_tokens,effort,notice=None):
    payload=copy.deepcopy(original)
    if payload.get('model')!=model:raise ValueError('Model mismatch')
    payload.pop('max_completion_tokens',None)  # One controller-pinned legacy API limit.
    payload.update(enable_thinking=True,reasoning_effort=effort,max_tokens=max_tokens)
    payload.update(temperature=1.0,top_p=0.95,top_k=20)  # Restore model-default sampling for both phases.
    payload['chat_template_kwargs']=dict(payload.get('chat_template_kwargs') or {},enable_thinking=True)
    if payload.get('stream'):payload['stream_options']=dict(payload.get('stream_options') or {},include_usage=True)
    if notice:
        # No timestamps/budgets are injected into the stable system prefix.
        payload['messages']=[*payload.get('messages',[]),dict(role='user',content=notice)]
    audit=dict(temperature=payload['temperature'],top_p=payload['top_p'],top_k=payload['top_k'],enable_thinking=payload['enable_thinking'],reasoning_effort=payload['reasoning_effort'],max_tokens=max_tokens,
        system_sha256=digest([m for m in original.get('messages',[]) if m.get('role') in ('system','developer')]),
        tools_sha256=digest(payload.get('tools',[])),history_sha256=digest(original.get('messages',[])),
        assistant_reasoning_messages=sum(any(m.get(k) for k in ('reasoning_content','reasoning','thinking','reasoning_details')) for m in original.get('messages',[]) if m.get('role')=='assistant'),
        phase=stage,scope='Parameter/history integrity; exposed reasoning is measured from actual response, never inferred')
    return payload,audit

class SemanticStream:
    def __init__(self):
        self.choices={};self.done=False;self.reasoning_chars=0;self.content_chars=0;self.parameter_chars=0
    def event(self,event):
        if event=='[DONE]':self.done=True;return
        if not isinstance(event,dict):return
        for c in event.get('choices',[]):
            state=self.choices.setdefault(c.get('index',0),dict(finish=None,tools={}))
            if c.get('finish_reason'):state['finish']=c['finish_reason']
            m=c.get('delta') or c.get('message') or {}
            self.reasoning_chars+=sum(len(v) if isinstance(v,str) else len(json.dumps(v)) for k,v in m.items() if k in ('reasoning_content','reasoning','thinking','reasoning_details') and v is not None)
            self.content_chars+=len(m.get('content') or '') if isinstance(m.get('content'),str) else 0
            for i,t in enumerate(m.get('tool_calls') or []):
                slot=state['tools'].setdefault(t.get('index',i),dict(id='',name='',arguments=''))
                if t.get('id'):slot['id']+=t['id']
                f=t.get('function',{})
                if f.get('name'):slot['name']+=f['name']
                if f.get('arguments') is not None:
                    if not isinstance(f['arguments'],str):raise RuntimeError('INVALID_TOOL_ARGUMENT_TYPE')
                    slot['arguments']+=f['arguments'];self.parameter_chars+=len(f['arguments'])
    def completeness(self):
        issues=[];calls=[]
        if not self.choices:issues.append('MISSING_CHOICES')
        for index,c in self.choices.items():
            if not c['finish']:issues.append('MISSING_FINISH_REASON')
            if c['finish'] in ('length','content_filter'):issues.append('INCOMPLETE_RESPONSE')
            for t in c['tools'].values():
                valid=False
                try:valid=isinstance(json.loads(t['arguments']),dict)
                except ValueError:pass
                complete=bool(t['id'] and t['name'] and valid)
                if not complete:issues.append('INCOMPLETE_TOOL_CALL')
                calls.append(dict(name=t['name'],argument_chars=len(t['arguments']),arguments_sha256=digest(t['arguments']),complete=complete))
        return dict(semantic_complete=not issues,issues=sorted(set(issues)),tool_calls=calls,exposed_reasoning_chars=self.reasoning_chars,content_chars=self.content_chars,tool_argument_chars=self.parameter_chars)
    def require_complete(self):
        result=self.completeness()
        if result['issues']:raise RuntimeError(result['issues'][0])
        return result

def reasoning_status(enabled,chars,transport_complete):
    if not enabled:return 'DISABLED_BY_REQUEST'
    if chars:return 'ENABLED_EXPOSED_CONTENT_CAPTURED'
    return 'ENABLED_NOT_EXPOSED' if transport_complete else 'ENABLED_INCOMPLETE_NO_EXPOSED_CONTENT'
