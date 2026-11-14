"""Private scenario-aware loopback transport (v4); upstream Kimi Code owns reasoning and tools.

Relay successful responses unchanged. Bound header, stream-idle and total time
independently; capture provider-visible prompts and exposed thinking in protected controller files.
Never record authentication headers or upstream HTTP error bodies.
Connection failures and retryable HTTP statuses are retried with backoff only
before the first upstream byte, so a partially relayed stream is never replayed.
"""
import hashlib
import json
import queue
import socket
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from tools.model_runner import _log
from tools.kimi_k3v19_capture import Capture
from tools.kimi_k3v19_budget import retry_delay
from tools.kimi_k3v19_retry import RetryLedger, transport_error
from tools.kimi_k3v19_recovery import RecoveryLedger, revision_observation


def native_phase(logs):
    """Use native wire events, not prompt text or absence of tools, for phase."""
    wires=list(logs.glob('kimi-native/sessions/*/*/wire.jsonl'))
    if not wires:return 'agent'
    path=max(wires,key=lambda p:p.stat().st_mtime_ns)
    with path.open('rb') as source:
        source.seek(max(0,source.seek(0,2)-65536))
        lines=source.read().decode('utf-8',errors='replace').splitlines()
    for line in reversed(lines):
        try:kind=json.loads(line).get('message',{}).get('type')
        except ValueError:continue
        if kind=='CompactionBegin':return 'compaction'
        if kind in ('CompactionEnd','StepBegin','TurnBegin'):return 'agent'
    return 'agent'


def request_budget(stage, remaining, *, timeout, compaction_timeout, recovery_reserve):
    """Per-request deadlines recover; only the original task deadline ends work."""
    limit=min(timeout,compaction_timeout) if stage=='compaction' else timeout
    if remaining is not None:limit=min(limit,max(0,remaining))
    mode=stage
    return max(0,limit),mode




RETRYABLE_STATUS=frozenset((408,429,500,502,503,504))


def operation_retry_key(stage,payload,revision,endpoint):
    """Persist the exact request/engineering-version identity across resumes.

    A phase-wide exhausted counter must not reject a different request or a
    fixed tool version before it even reaches the provider. Old ledgers remain.
    The local bridge port is ephemeral across resumes and is deliberately not
    part of operation identity. The requested model is already in the payload.
    No credential or raw request is stored in the key.
    """
    value=[stage,payload,revision]
    return 'provider-operation-v2:'+hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def server(endpoint,key,model,log,*,max_tokens=65536,timeout=600,
           idle_timeout=300,compaction_timeout=600,compaction_max_tokens=8192,
           remaining_seconds=None,phase=None,recovery_reserve=120,compaction_thinking=True, reasoning_effort="xhigh",
           compaction_reasoning_effort="low",capture_root=None,project_revision=None):
    if min(max_tokens,timeout,idle_timeout,compaction_timeout,compaction_max_tokens)<=0 or recovery_reserve<0:
        raise ValueError('Positive provider budgets required')
    if (reasoning_effort not in ('low','medium','xhigh') or compaction_reasoning_effort not in ('low','medium','xhigh')
            or not compaction_thinking or capture_root is None):
        raise ValueError('K3 requires thinking in every phase and a protected capture root')
    ledger=RetryLedger(Path(log).with_name('provider-retries-v5.json'))
    recovery=RecoveryLedger(Path(log).with_name('recovery-operations-v10.json'))
    stopping=threading.Event()
    active=threading.Condition();active_ids=set()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass

        def do_POST(self):
            if self.path!='/v1/chat/completions':self.send_error(404);return
            try:payload=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            except (ValueError,KeyError):self.send_error(400);return
            if payload.get('model')!=model:self.send_error(400,'Model mismatch');return
            stage=phase() if phase else 'agent'
            token_limit=compaction_max_tokens if stage=='compaction' else max_tokens
            remaining=remaining_seconds() if remaining_seconds else None
            total_limit,budget_phase=request_budget(stage,remaining,timeout=timeout,
                compaction_timeout=compaction_timeout,recovery_reserve=recovery_reserve)
            revision=revision_observation(project_revision)
            notice=recovery.notice(revision) if stage=='agent' else None
            from tools.kimi_k3v19_progress import prepare_payload
            thinking=True
            payload,parameter_audit=prepare_payload(payload,model,stage,token_limit,
                compaction_reasoning_effort if stage=='compaction' else reasoning_effort,notice)
            # Count the exact native chat template before reserving generation space.
            # The serving runtime reserves six tokens in addition to the prompt.
            count_payload={k:payload[k] for k in ('model','messages','tools','tool_choice',
                'reasoning_effort','chat_template_kwargs','continue_final_message') if k in payload}
            try:
                count_request=urllib.request.Request(endpoint.removesuffix('/chat/completions')+'/tokenize',
                    data=json.dumps(count_payload).encode(),
                    headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
                with urllib.request.urlopen(count_request,timeout=60) as counted:
                    token_info=json.load(counted)
                prompt_tokens=token_info['count']
                if type(prompt_tokens) is not int or token_info.get('max_model_len')!=262144:
                    raise ValueError('Unexpected tokenizer context binding')
                available=262144-prompt_tokens-6
                if available<1:
                    self.send_error(400,'Input exceeds the model context window');return
                payload['max_tokens']=min(token_limit,available)
                parameter_audit.update(requested_max_tokens=token_limit,max_tokens=payload['max_tokens'],
                    exact_prompt_tokens=prompt_tokens,context_length=262144,server_reserved_tokens=6)
                token_limit=payload['max_tokens']
            except (urllib.error.URLError,ValueError,KeyError,TimeoutError):
                self.send_error(503,'Tokenizer preflight unavailable');return
            raw=json.dumps(payload).encode()
            ident=uuid.uuid4().hex
            capture=Capture(capture_root,ident,payload,stage,model)
            transport_ok=False;capture_error='CONTROLLER_INTERRUPTED'
            retry_key=operation_retry_key(stage,payload,revision,endpoint)
            retry_decision=None
            terminal_counted=False
            started=time.monotonic()
            deadline=started+total_limit
            tool_names=[x.get('function',{}).get('name') for x in payload.get('tools',[])]
            _log(log,'provider_request',request_id=ident,phase=stage,model=model,
                 enable_thinking=thinking,reasoning_effort=payload['reasoning_effort'],max_tokens=token_limit,total_timeout_s=total_limit,
                 idle_timeout_s=idle_timeout,remaining_session_s=remaining,
                 budget_phase=budget_phase,runtime_notice=notice,
                 payload_sha256=hashlib.sha256(raw).hexdigest(),stream=payload.get('stream'),
                 tool_names=tool_names,parameter_audit=parameter_audit,project_revision=revision)
            events=queue.Queue(maxsize=8)
            cancelled=threading.Event()
            holder={}
            status=None;sent=False;received=0;first=None;last=None;upstream_id=None

            def publish(item):
                while not cancelled.is_set():
                    try:events.put(item,timeout=.1);return
                    except queue.Full:pass

            def fetch():
                nonlocal retry_decision,terminal_counted
                attempt=0
                while True:
                    failure=None
                    if not ledger.allowed(retry_key):
                        retry_decision={'retry_allowed':False,'retry_class':'BOUNDED_ERROR','retry_limit':3}
                        publish(('error','RETRY_LIMIT'));return
                    try:
                        request=urllib.request.Request(endpoint,data=raw,headers={
                            'Authorization':'Bearer '+key,'Content-Type':'application/json'})
                        with urllib.request.urlopen(request,timeout=max(.01,min(idle_timeout,deadline-time.monotonic()))) as response:
                            holder['response']=response
                            publish(('headers',(response.status,dict(response.headers))))
                            while not cancelled.is_set():
                                chunk=response.read1(65536)
                                if not chunk:break
                                publish(('data',chunk))
                            publish(('end',None))
                        return
                    except urllib.error.HTTPError as error:
                        error.close()
                        retry_decision=ledger.failed(retry_key,'UPSTREAM_HTTP_ERROR',status=error.code,network=True)
                        if error.code not in RETRYABLE_STATUS or not retry_decision['retry_allowed']:
                            retry_decision=dict(retry_decision,retry_allowed=False)
                            publish(('http_error',error.code));return
                        failure=('http_error',error.code)
                    except Exception as error:
                        kind=transport_error(error)
                        if 'response' in holder:publish(('error',kind));return
                        retry_decision=ledger.failed(retry_key,kind,network=True)
                        if not retry_decision['retry_allowed']:
                            terminal_counted=True
                            publish(('error',kind));return
                        failure=('error',kind)
                    delay=retry_delay(attempt)
                    if cancelled.is_set():return
                    if time.monotonic()>=deadline:
                        publish(('error','REQUEST_DEADLINE'));return
                    _log(log,'provider_retry',request_id=ident,phase=stage,attempt=attempt+1,
                         failure=failure[1] if failure[0]=='error' else 'HTTP_%d'%failure[1],delay_s=delay,
                         payload_sha256=hashlib.sha256(raw).hexdigest(),parameter_audit=parameter_audit,**retry_decision)
                    attempt+=1
                    if cancelled.wait(min(delay,max(0,deadline-time.monotonic()))):return

            def headers(code,content_type):
                self.send_response(code)
                self.send_header('Content-Type',content_type)
                self.send_header('Connection','close')
                self.end_headers()

            worker=threading.Thread(target=fetch,daemon=True)
            content_type='application/json';body=[]
            sse_pending=b'';saw_done=False;content_chars=0;finish_reason=None
            with active:active_ids.add(ident)
            try:
                if total_limit<=0:
                    raise TimeoutError('COMPACTION_BUDGET_EXHAUSTED' if stage=='compaction' and remaining and remaining>0 else 'SESSION_DEADLINE')
                worker.start()
                while True:
                    now=time.monotonic()
                    if stopping.is_set():raise TimeoutError('CONTROLLER_CANCELLED')
                    if now>=deadline:raise TimeoutError('REQUEST_DEADLINE')
                    if now-(last if last is not None else started)>=idle_timeout:
                        raise TimeoutError('STREAM_IDLE_TIMEOUT' if first is not None else 'FIRST_BYTE_TIMEOUT')
                    try:kind,value=events.get(timeout=min(.1,deadline-now))
                    except queue.Empty:continue
                    if kind=='headers':
                        status,upstream_headers=value
                        upstream_headers={k.lower():v for k,v in upstream_headers.items()}
                        content_type=upstream_headers.get('content-type','text/event-stream' if payload.get('stream') else 'application/json')
                        upstream_id=upstream_headers.get('x-request-id') or upstream_headers.get('request-id')
                        _log(log,'provider_headers',request_id=ident,phase=stage,status=status,
                             upstream_request_id=upstream_id,elapsed_s=time.monotonic()-started)
                    elif kind=='data':
                        capture.feed(value)
                        if any(x!=model for x in capture.returned_models):
                            raise RuntimeError('RETURNED_MODEL_MISMATCH')
                        last=time.monotonic();received+=len(value)
                        if first is None:
                            first=last
                            _log(log,'provider_first_byte',request_id=ident,phase=stage,elapsed_s=first-started)
                        if payload.get('stream'):
                            sse_pending+=value
                            while b'\n' in sse_pending:
                                line,sse_pending=sse_pending.split(b'\n',1)
                                if not line.startswith(b'data:'):continue
                                data=line[5:].strip()
                                if data==b'[DONE]':
                                    if stage=='compaction' and (not content_chars or finish_reason=='length'):
                                        raise RuntimeError('COMPACTION_INCOMPLETE')
                                    capture.semantic.require_complete()
                                    saw_done=True;continue
                                try:event=json.loads(data)
                                except ValueError:continue
                                if event.get('error'):raise RuntimeError('UPSTREAM_STREAM_ERROR')
                                for choice in event.get('choices',[]):
                                    content_chars+=len(choice.get('delta',{}).get('content') or '')
                                    finish_reason=choice.get('finish_reason') or finish_reason
                            if not sent:headers(status,content_type);sent=True
                            self.wfile.write(value);self.wfile.flush()
                        else:body.append(value)
                    elif kind=='end':
                        if payload.get('stream') and not saw_done:raise RuntimeError('INCOMPLETE_STREAM')
                        if not payload.get('stream'):
                            capture.parse_nonstream()
                            capture.semantic.require_complete()
                        if any(x!=model for x in capture.returned_models):raise RuntimeError('RETURNED_MODEL_MISMATCH')
                        if not capture.returned_models:raise RuntimeError('RETURNED_MODEL_UNAVAILABLE')
                        if not sent:headers(status or 200,content_type);sent=True
                        if body:self.wfile.write(b''.join(body));self.wfile.flush()
                        break
                    elif kind=='http_error':
                        status=value;raise RuntimeError('UPSTREAM_HTTP_ERROR')
                    else:raise RuntimeError(value)
                capture.semantic.require_complete()
                transport_ok=True;capture_error=None
                ledger.succeeded(retry_key)
                completed_recovery=recovery.completed(capture.semantic,revision)
                _log(log,'provider_response',request_id=ident,phase=stage,status=status,
                     elapsed_s=time.monotonic()-started,bytes_received=received,
                     finish_reason=finish_reason,content_chars=content_chars,
                     first_byte_s=first-started if first is not None else None,
                     last_byte_s=last-started if last is not None else None,
                     upstream_request_id=upstream_id,semantic_progress=capture.semantic.completeness(),parameter_audit=parameter_audit,operation_recovery=completed_recovery)
            except (OSError,TimeoutError,RuntimeError) as error:
                failure=str(error) if isinstance(error,(TimeoutError,RuntimeError)) else transport_error(error)
                # HTTP errors were counted by fetch; stream errors are counted here.
                if not terminal_counted and failure not in ('UPSTREAM_HTTP_ERROR','RETRY_LIMIT'):
                    retry_decision=ledger.failed(retry_key,failure,status=status,network=True)
                if failure in ('SESSION_DEADLINE','CONTROLLER_CANCELLED','RETURNED_MODEL_MISMATCH','RETURNED_MODEL_UNAVAILABLE'):
                    retry_decision=dict(retry_decision or {},retry_allowed=False)
                failed_recovery=recovery.failed(ident,failure,capture.semantic,revision,revision_observation(project_revision))
                _log(log,'provider_error',request_id=ident,phase=stage,status=status,error=failure,
                     elapsed_s=time.monotonic()-started,bytes_received=received,
                     finish_reason=finish_reason,content_chars=content_chars,
                     first_byte_s=first-started if first is not None else None,
                     last_byte_s=last-started if last is not None else None,
                     upstream_request_id=upstream_id,semantic_progress=capture.semantic.completeness(),parameter_audit=parameter_audit,operation_recovery=failed_recovery,**(retry_decision or {'retry_allowed':False}))
                capture_error=failure
                message=json.dumps({'error':{'type':'provider_transport_error','message':failure,'request_id':ident}}).encode()
                try:
                    if not sent:headers(status if status and status>=400 else 504,'application/json');self.wfile.write(message)
                    elif payload.get('stream'):
                        # SSE errors raise in the client, preventing successful
                        # adoption of a truncated call or compaction summary.
                        self.wfile.write(b'\n\ndata: '+message+b'\n\n')
                    self.wfile.flush()
                except OSError:pass
            finally:
                try:
                    capture.finish(transport_ok=transport_ok,error=capture_error)
                finally:
                    cancelled.set()
                    response=holder.get('response')
                    if response is not None:
                        try:response.fp.raw._sock.shutdown(socket.SHUT_RDWR)
                        except (AttributeError,OSError):pass
                    self.close_connection=True
                    with active:active_ids.discard(ident);active.notify_all()

    proxy=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    def cancel_requests():
        stopping.set()
        with active:active.wait_for(lambda:not active_ids,timeout=2)
    proxy.cancel_requests=cancel_requests
    return proxy
