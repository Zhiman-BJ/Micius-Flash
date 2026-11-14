"""Scenario-aware K3 v4 adapter for pinned Kimi Code CLI 1.50.0, with full exposed-thinking capture."""
import argparse,hashlib,json,os,re,signal,subprocess,threading,time,uuid
from pathlib import Path
from tools.kimi_dsflash_v6_tools import register
register()
from tools import agent_session,workspace as ws,task_contract
from tools import kimi_k3_isolation as session_isolation
from tools import kimi_k3v19_provider as kimi_provider
from tools.kimi_k3v23_budget import TaskBudget, recovery_allowed, retry_delay, terminal_result, steps_used, model_step_limit, FINALIZATION_STEPS
from tools.kimi_k3v19_retry import provider_failure
from tools import kimi_k3v19_capture as kimi_capture
from tools.runtime_limits import client_timeout_ms


ADAPTER_VERSION='micius-native-chat-v34-v1'
HARNESS_RETRY_LIMIT=3
# A single slow compaction or agent request must not end the session: compaction
# gets the agent budget with high thinking effort, and transient provider failures resume.
TRANSPORT=dict(max_tokens=131072,timeout=3600,idle_timeout=1800,compaction_timeout=3600,compaction_max_tokens=65536,
               compaction_thinking=True,compaction_reasoning_effort='xhigh',recovery_reserve=600)
DELIVERY_NOTE=('Deliver under output/ in the working directory, using exactly the CAD types and supporting files the public task requests. '
    'Schematic-only tasks require no PCB; preserve all local libraries referenced by their tables. '
    'When the public task requests manufacturing files, export_project(destination="output/release") writes the checked CAD and manufacturing files. '
    'The controller adopts the declared original project into the native board.* working copy, preserving UUIDs, circuit, copper and dependencies; .pcb/native-input-adoption.json records the mapping. Original filenames remain as input provenance, not a second editable candidate. On continuation the existing working CAD is retained. '
    'The project/ prefix in the public task names the input package. Inspect existing candidate files before editing.')


EFFICIENCY_NOTE=('Prefer inspect_board_batch for parts/pads/copper/drawings on one revision; retain snapshot_id and use inspect_board_delta after changes. '
    'Use native_project_status to distinguish native repair from intent design. Native copper edits use edit_board_copper without spec.json/ECO; set_board_outline_shapes preserves unselected and footprint edges. '
    'Use check_track_path before committing a proposed path; inspect_schematic_batch shares a hierarchy revision, and export_interface_table reads actual XML pin/net membership. '
    'Batch read-only queries; follow next_query when has_more, and use exact SHA. Cached full reports are not proof the model observed omitted data. '
    'Use set_schematic_wires, set_board_track_endpoints and place_board_items for small UUID edits instead of regenerating CAD or generating long file patches. '
    'Fetch only facts absent from provided local evidence. Pinned artifacts and unchanged qualified component packages are reused with hash checks. '
    'Use component_package_choices and select_component_package for ambiguous installed MPN versions; the selected package must match the existing CAD identities. '
    'Before finalizing, use mode_power_preflight_v2 on every declared source/battery/mode/load case; use declared_interface_preflight_v2 to compare stated connector pin roles with actual XML membership. These are diagnostics, not a functional PASS. '
    'Manufacturer source queries are available only after a native self-check of the exact current CAD; self-check during design is allowed. Use component_document_options_v1 and read_component_document_v2 for the exact selected part/ref/page; retain their source handles and SHA evidence. Use finalize_delivery once the candidate is ready; it makes a new revision-bound CAD/raw report/requested CAM bundle and preserves earlier outputs. '
    'Read compact summaries first and full receipts only as needed. Keep functional evidence and native structure/file delivery separate. '
    'Every request retains thinking; absent provider-exposed reasoning is not a disabled-thinking verdict. After partial streams inspect actual tool receipts before edits.')


def system_prompt(skill):
    from tools.engineering_tools_v1 import GUIDANCE
    skill += "\n\n" + GUIDANCE
    """Stable initial guidance including the actual available local-edit tools."""
    return ('You are designing a PCB using the public atomic tools. The immutable task contract is available through project_status. Candidate checks are diagnostic for uncompiled briefs. Give concise engineering reasons for decisions and use local repairs. Report design delivery, required simulation and measured hardware qualification separately. Never claim tests that have not executed. Tool names carry an MCP prefix.\n\n'
        +skill+'\n\nUse list_project_files/read_project_file/write_project_file for candidate files when atomic tools do not cover an edit. Preserve the frozen modification scope. Never replace an existing project with a fresh design to avoid an input defect. A no-input task starts with no CAD: create files as needed. Treat runtime ending as submission, not a completion verdict.\n\n'
        +DELIVERY_NOTE+'\n\n'+EFFICIENCY_NOTE+'\n\nUse preview_project_patch before text fallback when a unique exact match is uncertain; no fuzzy edits are applied. A NOT_RUN stage observation supplies actual names, not a failed circuit. Device pin functions, package view and mode/support connections need retained external facts before wiring. Evaluate actual public schematic scope separately from PCB/fit/full protocol; do not substitute packaging or ERC for function.')


def finalize_if_budget(args, logs):
    """One ordinary charged operation; no time/step allowance is extended."""
    if steps_used(logs)>=args.max_steps:
        return None
    from tools import kimi_k3v23_mcp
    try:return kimi_k3v23_mcp.execute('finalize_delivery',{},actor='finalizer')
    except Exception as error:
        return dict(status='INCOMPLETE',error_type=type(error).__name__,engineering_status='UNASSESSED')


def exit_reason(returncode,stream,max_steps):
    """Classify the pinned CLI's terminal record, never model-authored JSON.

    Upstream 1.50.0 prints MaxStepsReached as a plain final line even in
    stream-json mode. A nonzero exit remains unsuccessful, with a precise cause.
    """
    if returncode==0:return 'HARNESS_FINISHED'
    if returncode<0:return 'HARNESS_INTERRUPTED'
    with Path(stream).open('rb') as source:
        source.seek(max(0,source.seek(0,2)-4096))
        lines=source.read().decode('utf-8',errors='replace').splitlines()
    terminal=next((line.strip() for line in reversed(lines) if line.strip()),'')
    matched=re.fullmatch(r'Max number of steps reached: ([0-9]+)',terminal)
    # Only the CLI's terminal error line may name a transport failure. Never
    # derive termination from a model-authored JSON message containing a label.
    failure=re.fullmatch(r'Error: (REQUEST_DEADLINE|SESSION_DEADLINE|COMPACTION_BUDGET_EXHAUSTED|COMPACTION_INCOMPLETE|STREAM_IDLE_TIMEOUT|FIRST_BYTE_TIMEOUT|CONTROLLER_CANCELLED)',terminal)
    if failure:return failure[1]
    return 'STEP_BUDGET' if matched and int(matched[1])==max_steps else 'HARNESS_ERROR'


def thinking_only_harness_error(returncode, stream):
    """Recognize the observed CLI 75 no-action response, not arbitrary failures.

    The pinned CLI emits this terminal diagnostic after a complete provider
    response containing only thinking and no text or tool call. A transport,
    authentication, configuration, or tool failure must not match it.
    """
    if returncode != 75:
        return None
    with Path(stream).open('rb') as source:
        source.seek(max(0, source.seek(0, 2)-4096))
        tail = source.read().decode('utf-8', errors='replace')
    if not re.search(r'The API returned a response containing only thinking content without any text or\s*tool calls\.', tail):
        return None
    return {'class': 'THINKING_ONLY_NO_ACTION', 'cli_returncode': 75,
            'terminal_tail_sha256': hashlib.sha256(tail.encode()).hexdigest()}


def harness_retry_allowed(result, history, remaining_steps):
    """At most three durable same-session resumes for this exact CLI failure."""
    if (remaining_steps <= 0 or result.get('reason') != 'HARNESS_ERROR'
            or result.get('returncode') != 75
            or result.get('recoverable_provider_error') is not None
            or not result.get('recoverable_harness_error')):
        return False
    used = sum(row.get('retry_class') == 'HARNESS_THINKING_ONLY'
               and row.get('session_id') == result.get('session_id') for row in history)
    return used < HARNESS_RETRY_LIMIT


def run(args):
    if not args.model or args.reasoning_effort not in ('low','medium','xhigh'):
        raise ValueError('Select a model and a supported thinking effort')
    version=subprocess.check_output(['/opt/kimi/bin/kimi','--version'],text=True,timeout=30).strip()
    if not re.search(r'(?<![0-9])1\.50\.0(?![0-9])',version):
        raise ValueError('This adapter requires the pinned Kimi Code CLI 1.50.0')
    root=Path(args.workspace).resolve();os.chdir(root)
    if args.resume:
        from tools.continuation_upgrade import preflight
        preflight(root)
    if not args.native_input_layout:
        from tools.native_project_input import normalize_layout
        for name in ('input-layout.json','runtime-input-layout.json','input-contract.json'):
            candidate=Path(args.task).resolve().parent/name
            if candidate.is_file() and normalize_layout(json.loads(candidate.read_bytes())).get('initial_primary'):
                args.native_input_layout=str(candidate);break
    if args.native_input_layout:
        from tools.native_project_input import prepare
        prepare(root,args.native_input_layout,resume=args.resume)
    os.environ['PCB_TASK_PROMPT_PATH']=str(Path(args.task).resolve())
    os.environ['PCB_DELIVERY_REQUIREMENTS']=str(Path(args.delivery_requirements).resolve())
    os.environ['PCB_DELIVERY_REQUIREMENTS_SHA256']=args.delivery_requirements_sha256
    if not args.resume:(root/'session').mkdir(mode=0o755)
    from tools.kimi_k3v23_delivery_context import prepare as prepare_delivery_context
    args.task_phase=prepare_delivery_context(root,args.task_phase,args.resume)
    from tools.kimi_k3v23_delivery import contract
    delivery_contract=contract()
    if delivery_contract['prompt_sha256']!=hashlib.sha256(Path(args.task).read_bytes()).hexdigest():raise ValueError('Actual task and delivery contract differ')
    public_document=task_contract.load()
    if public_document is not None and (not task_contract.is_brief(public_document) or public_document['brief']['text']!=Path(args.task).read_text()):
        raise ValueError('Public engineering requirements must equal the actual prompt; stale contracts are rejected')
    uid,gid=map(int,args.tool_user.split(':'))
    from tools.session_recovery import lease,durable_write
    from tools import harness_recovery,model_runner
    with lease(root,'harness') as harness_fd:
        with lease(root):
            (root/'.pcb').mkdir(exist_ok=True)
            session_isolation.prepare(root,dict(uid=uid,gid=gid),resume=True)
        budget=TaskBudget(root,seconds=args.max_seconds,steps=args.max_steps,task=args.task,resume=args.resume)
        args._task_budget=budget
        policy_path=root/'session/automatic-resumes.json'
        history=json.loads(policy_path.read_text()) if policy_path.exists() else []
        previous=None
        while True:
            used=steps_used(root/'session')
            if used>=args.max_steps:
                result=terminal_result(budget,used,previous)
                durable_write(root/'session/result.json',result)
                return result
            result=_run(args,root,harness_fd)
            used=steps_used(root/'session')
            previous=result
            if not result.get('harness_finished') and used>=args.max_steps:
                result=terminal_result(budget,used,result)
                durable_write(root/'session/result.json',result)
                return result
            provider_retry = recovery_allowed(result,budget.remaining(),args.max_steps-used)
            harness_retry = harness_retry_allowed(result,history,args.max_steps-used)
            if not (provider_retry or harness_retry):
                result=dict(result,elapsed_s=budget.elapsed(),retry_count_limit=3,
                            harness_retry_count=sum(row.get('retry_class')=='HARNESS_THINKING_ONLY' and row.get('session_id')==result.get('session_id') for row in history),
                            harness_retry_limit=HARNESS_RETRY_LIMIT,network_timeout_retry_count_limit=None)
                durable_write(root/'session/result.json',result)
                return result
            event=dict(after_round=result['round_id'],provider_error=result['recoverable_provider_error'],
                       harness_error=result.get('recoverable_harness_error') if harness_retry else None,
                       retry_class='HARNESS_THINKING_ONLY' if harness_retry else 'PROVIDER',
                       remaining_s=budget.remaining(),session_id=result['session_id'],
                       delay_s=retry_delay(len(history)),retry_count_limit=3,network_timeout_retry_count_limit=None)
            history.append(event);durable_write(policy_path,history)
            model_runner._log(root/'session/model-trajectory.jsonl','automatic_resume',**event)
            budget.wait(event['delay_s'])
            # Reconcile pending operations and preserve the SAME session/CAD;
            # neither a failed request nor controller downtime resets budgets.
            args=argparse.Namespace(**dict(vars(args),resume=True))


def _run(args,root,harness_fd):
    started=time.monotonic()
    logs=root/'session';logs.mkdir(exist_ok=True);identity=logs/'kimi-identity.json'
    os.environ['PCB_TOOL_IDENTITY']=args.tool_user
    from tools.kimi_k3v19_runtime import capabilities
    caps=capabilities()
    capfile=logs/'runtime-capabilities-v13.json'
    if not capfile.exists():ws._write(capfile,caps)
    else:
        prior=json.loads(capfile.read_bytes())
        if {k:prior[k].get('status') for k in ('router','run_python')}!={k:caps[k].get('status') for k in ('router','run_python')}:raise ValueError('Runtime capabilities changed across resume')
    agent_session.TOOLS=agent_session.build_tools(agent_session.configured_groups())
    if caps['run_python']['status']!='AVAILABLE':agent_session.TOOLS=[x for x in agent_session.TOOLS if x['function']['name']!='run_python']
    agent_session.TOOLSET_SHA256=hashlib.sha256(json.dumps(agent_session.TOOLS,sort_keys=True).encode()).hexdigest()
    public=task_contract.binding()
    # Design decisions retain thinking. Mechanical context summarization has
    # its own bounded request limit; no model-visible time notices are injected.
    transport=dict(TRANSPORT)
    binding=dict(delivery_context_sha256=os.environ['PCB_DELIVERY_CONTEXT_SHA256'],task_phase=args.task_phase,model=args.model,task_sha256=hashlib.sha256(Path(args.task).read_bytes()).hexdigest(),
        toolset_sha256=agent_session.TOOLSET_SHA256,requirements=public,
        harness_version='1.50.0',request_options={'enable_thinking':True,'reasoning_effort':args.reasoning_effort},transport=transport,
        context_size=args.context_size,adapter_version=ADAPTER_VERSION,cli_sha256=hashlib.sha256(Path('/opt/kimi/bin/kimi').read_bytes()).hexdigest(),
        max_steps=args.max_steps,max_seconds=args.max_seconds,retry_policy='SCENARIO_RETRIES_V1',delivery_contract_sha256=args.delivery_requirements_sha256,
        time_notice_policy='NONE',verification_axes=['answer','structure','completion'],
        reserved_finalization_steps=FINALIZATION_STEPS)
    if args.resume:
        old=json.loads(identity.read_text())
        if old['binding']!=binding:
            from tools.continuation_upgrade import binding_allowed
            if not binding_allowed(root,old['binding'],binding):raise ValueError('Resume requires original task, tools, model and budgets')
        session_id=old['session_id']
    else:
        if identity.exists():raise FileExistsError('Existing Kimi session: use explicit resume')
        session_id=str(uuid.uuid4());ws._write(identity,dict(binding=binding,session_id=session_id))
    skill=(Path(os.environ.get('PCB_SKILLS','/skills'))/'pcb-design-e2e/SKILL.md').read_text()
    skill+='\n\n'+(Path(os.environ.get('PCB_SKILLS','/skills'))/'pcb-component-verification-v1/SKILL.md').read_text()
    skill+='\n\n'+(Path(os.environ.get('PCB_SKILLS','/skills'))/'pcb-mode-interface-audit-v2/SKILL.md').read_text()
    skill+='\n\n'+(Path(os.environ.get('PCB_SKILLS','/skills'))/'pcb-evidence-corners-v1/SKILL.md').read_text()
    system=system_prompt(skill)+'\n\nRuntime capabilities: '+json.dumps({k:{a:b for a,b in caps[k].items() if a in ('status','reason','actual_java_major','required_java_major','unrestricted_fallback')} for k in ('router','run_python')},sort_keys=True)
    from tools import runtime_identity
    runtime_identity.record(logs,system)
    os.environ['PCB_TOOL_IDENTITY']=args.tool_user
    recovery=None
    if args.resume:
        from tools import session_recovery
        recovery=session_recovery.reconcile(root)
    from tools import harness_recovery
    remaining_steps=model_step_limit(args.max_steps,steps_used(logs))
    if remaining_steps<=0:
        result=dict(harness='Kimi Code CLI',harness_version='1.50.0',model=args.model,
            session_id=session_id,round_id=str(time.time_ns()),reason='STEP_BUDGET',returncode=1,
            recoverable_provider_error=None,harness_finished=False,
            assisted_delivery=finalize_if_budget(args,logs),elapsed_s=time.monotonic()-started,
            usage=[],note='Reserved delivery only; no further model request and no engineering verdict')
        result['native_steps_used']=steps_used(logs)
        ws._write(logs/'result.json',result)
        return result
    (logs/'kimi-system.md').write_text(system)
    (logs/'kimi-agent.yaml').write_text('version: 1\nagent:\n  name: pcb\n  system_prompt_path: ./kimi-system.md\n  tools: []\n')
    os.environ['PCB_TOOL_IDENTITY']=args.tool_user
    round_id=str(time.time_ns());os.environ['PCB_TRACE_ROUND_ID']=round_id
    mcp_env={k:v for k,v in os.environ.items() if k.startswith(('PCB_','XDG_')) or k in ('PYTHONPATH','FREEROUTING_JAR','JAVA','OPENEMS_INSTALL_PATH','CSXCAD_INSTALL_PATH')}
    ws._write(logs/'kimi-mcp.json',{'mcpServers':{'pcb':{'command':'/usr/bin/python3','args':['-m','tools.kimi_dsflash_v6_mcp'],'env':mcp_env}}})
    request_clock={}
    from tools.kimi_k3v19_recovery import candidate_revision
    proxy=kimi_provider.server(args.endpoint,os.environ['MODEL_API_KEY'],args.model,logs/'provider.jsonl',
        **transport,reasoning_effort=args.reasoning_effort,capture_root=logs/'provider-capture',remaining_seconds=args._task_budget.remaining,
        phase=lambda:kimi_provider.native_phase(logs),project_revision=lambda:candidate_revision(root))
    threading.Thread(target=proxy.serve_forever,daemon=True).start()
    # The actual API key is held by the private proxy; it never enters CLI config.
    private=Path('/tmp/pcb-kimi-private');private.mkdir(mode=0o700,exist_ok=True)
    config={'default_model':'pcb','default_thinking':True,
        'providers':{'pcb':{'type':'openai_legacy','base_url':f'http://127.0.0.1:{proxy.server_port}/v1','api_key':'local-adapter','reasoning_key':'reasoning_content'}},
        'models':{'pcb':{'provider':'pcb','model':args.model,'max_context_size':args.context_size,'capabilities':['thinking']}},
        # One CLI attempt is a handoff boundary, never a task retry quota.
        # The proxy and outer controller retry against the one task budget.
        'loop_control':{'max_steps_per_turn':remaining_steps,'max_retries_per_step':1,'reserved_context_size':32768},
        'mcp':{'client':{'tool_call_timeout_ms':client_timeout_ms()}}}
    ws._write(private/'config.json',config)
    os.environ['KIMI_SHARE_DIR']=str(logs/'kimi-native')
    env={k:v for k,v in os.environ.items() if not any(x in k.upper() for x in ('API_KEY','TOKEN','SECRET','PASSWORD'))}
    prompt='Continue the same task. Inspect project_status and existing evidence before a local repair; preserve completed work.' if args.resume else Path(args.task).read_text()
    if not args.resume and delivery_contract_phase(args)==2:
        prompt+='\n当前为第二阶段：待修PCB已提供，第一阶段提交原理图已封存并只读挂载；保留其确切版本，完成PCB修复与联合交付。\n'
    if recovery:prompt+='\nController recovery report (not design acceptance):\n'+json.dumps(recovery,ensure_ascii=False)
    command=['/opt/kimi/bin/kimi','--config-file',str(private/'config.json'),'--agent-file',str(logs/'kimi-agent.yaml'),
             '--mcp-config-file',str(logs/'kimi-mcp.json'),'--work-dir',str(root),'--session',session_id,
             '--thinking','--print','--output-format','stream-json','--max-steps-per-turn',str(remaining_steps)]
    from tools.session_recovery import durable_write
    # A killed controller used time even if it never wrote its terminal result.
    # Charge the unrecorded wall interval conservatively; never reset its budget.
    for marker in logs.glob('kimi-start-*.json'):
        terminal=logs/marker.name.replace('kimi-start-','kimi-round-')
        if not terminal.exists():
            start=json.loads(marker.read_text())
            elapsed=max(0,time.time()-start['started_at'])
            durable_write(terminal,dict(reason='CONTROLLER_INTERRUPTED',elapsed_s=elapsed,
                scope='Conservative elapsed wall time; provider outcome is unknown'))
    elapsed_before=sum(json.loads(p.read_text()).get('elapsed_s',0) for p in logs.glob('kimi-round-*.json'))
    remaining=args._task_budget.remaining()
    assert remaining is None, 'V23 cumulative deadline must remain disabled'
    provider_offset=(logs/'provider.jsonl').stat().st_size if (logs/'provider.jsonl').exists() else 0
    durable_write(logs/f'kimi-start-{round_id}.json',dict(started_at=time.time(),remaining_s=remaining))
    with (logs/f'kimi-stream-{round_id}.jsonl').open('w') as out,(logs/f'kimi-stderr-{round_id}.log').open('w') as err:
        process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=out,stderr=err,env=env,text=True,start_new_session=True,pass_fds=(harness_fd,))
        wait_for_cli(process,prompt)
        reason=exit_reason(process.returncode,logs/f'kimi-stream-{round_id}.jsonl',remaining_steps)
    proxy.cancel_requests()
    proxy.shutdown()
    proxy.server_close()
    receipts=[json.loads(p.read_text()) for p in logs.glob('receipt-*.json')]
    assisted_delivery=None
    if reason in ('HARNESS_FINISHED','STEP_BUDGET'):
        assisted_delivery=finalize_if_budget(args,logs)
    result=dict(harness='Kimi Code CLI' ,harness_version='1.50.0',model=args.model,session_id=session_id,
        round_id=round_id,reason=reason,returncode=process.returncode,
        recoverable_provider_error=provider_failure(logs/'provider.jsonl',provider_offset),
        recoverable_harness_error=thinking_only_harness_error(process.returncode,logs/f'kimi-stream-{round_id}.jsonl') if reason=='HARNESS_ERROR' else None,
        native_steps_used=steps_used(logs),harness_finished=reason=='HARNESS_FINISHED',
        elapsed_s=time.monotonic()-started,request_options={'enable_thinking':True,'reasoning_effort':args.reasoning_effort},
        actions=sum(r.get('actor','model')=='model' for r in receipts),
        controller_inspections=sum(r.get('actor')=='controller' for r in receipts),
        assisted_delivery=assisted_delivery,usage=[],note='Harness exit is not independent design acceptance')
    ws._write(logs/f'kimi-round-{round_id}.json',result)
    from tools import trajectory_index
    result['trajectory_index']=trajectory_index.build(logs)
    result['provider_capture']=kimi_capture.export(logs/'provider-capture',logs/f'capture-export-{round_id}')
    from tools.kimi_k3v19_trace import export_session
    # The session and phase context were created once in run(). Reuse that
    # pinned context for first-run and resumed terminal export; never mkdir
    # or reinitialize a session that already contains the completed actions.
    from tools.kimi_k3v23_delivery import contract
    try:
        result['unified_trace']=export_session(logs,logs/f'unified-trace-{round_id}',contract()['question'])
    except Exception as error:
        # Derivative export must not hide an actual generation exit or prevent
        # the existing bounded transport recovery. Raw captures remain primary.
        result['unified_trace']={'status':'DERIVED_EXPORT_FAILED','error_type':type(error).__name__,
                                 'raw_session_retained':True,'engineering_verdict_unchanged':True}
        ws._write(logs/f'unified-trace-error-{round_id}.json',result['unified_trace'])
    result['usage']=result['provider_capture']['usage']
    ws._write(logs/'result.json',dict(result,elapsed_s=elapsed_before+result['elapsed_s']))
    print(json.dumps(result),flush=True)
    return result


def delivery_contract_phase(args):
    return args.task_phase

def wait_for_cli(process,prompt):
    return process.communicate(prompt,timeout=None)


def unlimited_seconds(value):
    if value.lower() not in ('none','off','unlimited'):
        raise argparse.ArgumentTypeError('V23 disables cumulative timeout; use none')
    return None


def step_limit(value):
    steps=int(value)
    if not 1<=steps<=500:raise argparse.ArgumentTypeError('max-steps must be 1..500')
    return steps


def parse_args(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--workspace',default='/workspace');p.add_argument('--task',required=True)
    p.add_argument('--model',required=True);p.add_argument('--endpoint',required=True);p.add_argument('--tool-user',default='1000:1000')
    p.add_argument('--task-phase',type=int,choices=(1,2),default=None);p.add_argument('--max-steps',type=step_limit,default=500);p.add_argument('--max-seconds',type=unlimited_seconds,default=None);p.add_argument('--resume',action='store_true')
    p.add_argument('--reasoning-effort',choices=('low','medium','xhigh'),default='xhigh')
    p.add_argument('--delivery-requirements',required=True);p.add_argument('--delivery-requirements-sha256',required=True)
    p.add_argument('--context-size',type=int,choices=(262144,),default=262144)
    p.add_argument('--native-input-layout',default=None,help='Controller-pinned original native project layout; adopt before model actions')
    return p.parse_args(argv)


def main():
    args=parse_args()
    try:
        result=run(args)
    except BaseException as error:
        logs=Path(args.workspace)/'session'
        if logs.is_dir() and not logs.is_symlink():
            kimi_capture.write_new(logs/f'harness-fatal-{time.time_ns()}.json',dict(
                error_type=type(error).__name__,scope='Interrupted or failed harness; no completion verdict'))
        raise
    raise SystemExit(0 if result['harness_finished'] else 1)
if __name__=='__main__':main()
