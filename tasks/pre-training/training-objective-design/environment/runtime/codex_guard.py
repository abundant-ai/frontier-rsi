#!/usr/bin/env python3
"""Bound Codex provider recovery without resetting the research budget.

Installed as /usr/local/bin/codex; the official, pinned executable stays under
/opt/rsi-codex. It never reads or writes API keys; endpoint routing comes from the existing environment.
"""
import json,os,selectors,signal,subprocess,sys,time
from pathlib import Path

REAL='/opt/rsi-codex/node_modules/.bin/codex'
TEMPFAIL=75
NETWORK=('connection reset','econnreset','waiting for network','connection failed','stream disconnected','error sending request','timed out','service unavailable','at capacity','rate limit','too many requests','http 429','http 503')


def event_kind(line):
 try:event=json.loads(line)
 except (ValueError,TypeError):return 'other'
 if not isinstance(event,dict):return 'other'
 if event.get('type')=='turn.completed':return 'progress'
 if event.get('type') in ('item.started','item.completed'):
  item=event.get('item') or {}
  if item.get('type') in ('agent_message','reasoning','agent_reasoning','command_execution','mcp_tool_call','file_change'):return 'progress'
 # Only native top-level CLI errors count; a tool's quoted output is not parsed.
 if event.get('type') in ('error','turn.failed'):
  text=json.dumps(event).lower()
  if any(marker in text for marker in NETWORK):return 'provider_error'
 return 'other'


def resume_args(args):
 if args[:1]!=['exec']:raise ValueError('guard requires exec')
 prefix=args[:args.index('--')] if '--' in args else list(args)
 if prefix[1:2]!=['resume']:prefix[1:1]=['resume','--last']
 return prefix+['--','Continue the existing task after the provider interruption. Preserve all original task constraints and continue from the current files.']


def stop(p):
 try:os.killpg(p.pid,signal.SIGTERM)
 except ProcessLookupError:return
 try:p.wait(timeout=2)
 except subprocess.TimeoutExpired:
  try:os.killpg(p.pid,signal.SIGKILL)
  except ProcessLookupError:pass
  p.wait()


def run(args,real=REAL,startup_seconds=180,stall_seconds=120,max_attempts=3,budget_seconds=14400,backoffs=(5,15),emit=None):
 if emit is None:
  def emit(data):sys.stdout.buffer.write(data);sys.stdout.buffer.flush()
 began=time.monotonic();ever_progress=False;current=list(args)
 for attempt in range(max_attempts):
  if time.monotonic()-began>=budget_seconds:break
  # Built-in provider IDs are reserved by the pinned CLI. A separate provider
  # preserves the existing endpoint and credential environment while selecting HTTP.
  endpoint=os.environ.get('OPENAI_BASE_URL','https://api.openai.com/v1')
  config={'model_provider':'rsi_http','model_providers.rsi_http.name':'RSI HTTP',
          'model_providers.rsi_http.base_url':endpoint,
          'model_providers.rsi_http.env_key':'OPENAI_API_KEY',
          'model_providers.rsi_http.wire_api':'responses',
          'model_providers.rsi_http.supports_websockets':False}
  flags=[v for key,value in config.items() for v in ('-c',key+'='+json.dumps(value))]
  command=[real,*current[:1],*flags,*current[1:]]
  p=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True,bufsize=0)
  sel=selectors.DefaultSelector();sel.register(p.stdout,selectors.EVENT_READ)
  started=time.monotonic();last_progress=started;progress=False;provider_error=False;pending=b'';aborted=False
  try:
   while True:
    now=time.monotonic()
    if now-began>=budget_seconds or (not progress and now-started>=startup_seconds) or (provider_error and now-last_progress>=stall_seconds):
     stop(p);aborted=True;break
    ready=sel.select(timeout=min(.1,max(.001,budget_seconds-(now-began))))
    if ready:
     chunk=os.read(p.stdout.fileno(),65536)
     if not chunk:break
     emit(chunk);pending+=chunk
     while b'\n' in pending:
      line,pending=pending.split(b'\n',1);kind=event_kind(line)
      if kind=='progress':progress=ever_progress=True;last_progress=time.monotonic();provider_error=False
      elif kind=='provider_error':provider_error=True
    elif p.poll() is not None:break
   code=p.wait()
  finally:
   if p.poll() is None:stop(p)
   sel.close();p.stdout.close()
  if not aborted and code==0 and not provider_error:return 0
  if not aborted and not provider_error:return code
  if attempt+1>=max_attempts:break
  remaining=budget_seconds-(time.monotonic()-began)
  if remaining<=0:break
  delay=min(backoffs[min(attempt,len(backoffs)-1)],remaining) if backoffs else 0
  emit((f'codex: retrying provider connection; attempt={attempt+1} resume={int(ever_progress)}\n').encode())
  time.sleep(delay)
  current=resume_args(args) if ever_progress else list(args)
 emit(b'codex: provider connection unavailable after retries.\n')
 return TEMPFAIL


def main():
 args=sys.argv[1:]
 if not args or args[0]!='exec':os.execv(REAL,[REAL,*args])
 def emit(data):sys.stdout.buffer.write(data);sys.stdout.buffer.flush()
 return run(args,emit=emit)


if __name__=='__main__':raise SystemExit(main())
