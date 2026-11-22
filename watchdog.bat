@echo off
setlocal DisableDelayedExpansion
set "PY=python"
set "PROJ=<PROJECT_ROOT>"
set "LOGDIR=<USER_HOME>\AppData\Local\Temp\opencode"
REM ponytail: the four ensures used to run ONCE and exit, so a service that
REM died two minutes later stayed dead until someone re-ran this by hand
REM (53 "services down" rows in issues.json). Now a hidden --loop keeps
REM cycling. Cadence: set WD_INTERVAL=60 in the SAME shell before calling.
if not defined WD_INTERVAL set "WD_INTERVAL=60"
REM A non-numeric interval makes Start-Sleep throw and the loop spins as fast
REM as PowerShell starts, hammering all four services. Pin it to 60 instead.
for /f "delims=0123456789" %%a in ("%WD_INTERVAL%") do set "WD_INTERVAL=60"
if %WD_INTERVAL% LSS 5 set "WD_INTERVAL=5"
REM How old the lock heartbeat may get before the lock counts as abandoned.
REM Must exceed one full cycle by a wide margin: :ensure_llm alone can block
REM for 2 minutes waiting for health, and a lock that expired mid-cycle would
REM let a second loop start and fight over the :8080 relaunch.
if not defined WD_STALE_S set /a WD_STALE_S=%WD_INTERVAL% * 10 + 300

set "WD_LOCK=%PROJ%\watchdog.lock"
REM ponytail: a resident loop needs a singleton. Before this, N bare runs
REM meant N permanent loops racing on :ensure_llm, which taskkills every
REM llama-server.exe holding :8080 and relaunches — a kill/relaunch war on
REM exactly the service you reached the watchdog for. One loop or none.
REM
REM The claim is `mkdir`, which is atomic and fails if the dir exists.
REM Two earlier attempts are documented here because both looked correct and
REM did nothing:
REM  - Get-CimInstance COMMANDLINE match — a .bat has no %PID%, and the parent
REM    cmd's command line does not carry --loop, so it matched nothing.
REM  - reading tasklist /V into a variable — a `set` inside a parenthesised
REM    block is expanded when the BLOCK is parsed under DisableDelayedExpansion,
REM    so the `if defined` check never saw it.
REM A lock directory has neither problem: no variable crosses a block, and the
REM OS arbitrates the race.
REM
REM The CHILD claims the lock, not the parent: the parent must return
REM immediately so the dashboard can be opened straight after, and the child
REM The PARENT claims the lock, before it spawns anything. The child must
REM NOT claim: `start` gives the child the "anime-watchdog" title, so any
REM liveness test the child runs can match ITSELF and decline — that is why
REM 4 bare invocations produced 4 competing loops. The parent has no such
REM title, so it can judge the lock unambiguously. The child only beats.
if /i not "%~1"=="--loop" (
  if not exist "%LOGDIR%" mkdir "%LOGDIR%" >nul 2>&1
  call :wd_claim
  if errorlevel 1 (
    echo watchdog already running - not starting another
    echo stop it with:  taskkill /FI "WINDOWTITLE eq anime-watchdog" /F
    exit /b 0
  )
  start "anime-watchdog" /min "%~f0" --loop
  exit /b 0
)

:watch
  call :ensure 5003 "%PROJ%\cosy-server" server.py "%LOGDIR%\cosy_watchdog.log" cosy-tts /health SKIP_RENDER

call :ensure 8000 "%PROJ%\hermes-skill\scripts" pipeline_server.py "%LOGDIR%\dash_watchdog.log" dash /api/chapters
call :ensure_llm
call :ensure_comfy
REM Refresh the lock's heartbeat so a second invocation can tell a live loop
REM from a dead one by age. Beat BEFORE the sleep as well as after: :ensure_llm
REM alone can block for minutes on a broken machine, and the old comment here
REM claimed the beat was "skipped when a service is down" so a struggling cycle
REM could not look abandoned — but :wd_beat is unconditional and ran only AFTER
REM the sleep, so a slow cycle really could outrun WD_STALE_S.
call :wd_beat
powershell -NoProfile -NonInteractive -Command "Start-Sleep -Seconds %WD_INTERVAL%"
goto watch

REM Claims the lock dir. Returns errorlevel 1 when a fresh lock already
REM exists (a live loop), 0 when claimed or reclaimed after going stale.
:wd_claim
REM Liveness = a HEARTBEAT the loop refreshes every cycle, judged by file age.
REM
REM Three earlier designs each looked right and silently failed, all found by
REM running the real script rather than reading it:
REM  1. Get-CimInstance COMMANDLINE match - a .bat has no %PID% and the parent
REM     cmd's command line does not carry --loop, so it matched nothing.
REM  2. tasklist /V WINDOWTITLE - `start` gives the CHILD that title, so the
REM     child matched itself and declined every time; and a /min window
REM     spawned from a detached parent is not reported by tasklist /V at all,
REM     so the loop's own pid could never be recorded. A lock whose liveness
REM     depends on identifying the holder cannot work when the holder cannot
REM     be identified.
REM  3. reading tasklist into a variable - under DisableDelayedExpansion a
REM     `set` inside a parenthesised block is expanded when the BLOCK is
REM     parsed, so the `if defined` check never saw it.
REM
REM A heartbeat has none of those dependencies: the loop touches a file it
REM already owns, and staleness is arithmetic on a timestamp. The claim itself
REM is New-Item -Force, which is atomic, so two racing starts cannot both win.
if not exist "%LOGDIR%" mkdir "%LOGDIR%" >nul 2>&1
REM ponytail: the claim must be atomic, and the previous version was not.
REM It ran  Test-Path -> age -> Remove-Item -> New-Item -Force, which has a
REM TOCTOU window: two starters can both observe "stale", both delete, both
REM create, and both exit 0. (And -Force is what made it possible — New-Item
REM WITHOUT -Force fails on an existing directory, which is the mutual
REM exclusion we actually want. -Force is only needed on the reclaim path,
REM which is why the reclaim below uses an atomic RENAME instead.)
REM
REM The reclaim is a compare-and-swap on a directory rename: NTFS makes
REM Rename-Item atomic, so of two racers exactly one succeeds and the loser
REM gets ItemNotFound and exits 1. Fresh-lock path and no-lock path are both
REM a single failing New-Item. No sleep, no polling, no window.
REM
REM The reclaimed directory is renamed NEXT TO the lock (leaf name + suffix),
REM never into the cwd: a bare -NewName is relative to the CURRENT directory,
REM so 'reclaim.<guid>' landed in the project root instead and the cleanup glob
REM below never matched. Derive the name from the leaf so both agree.
powershell -NoProfile -NonInteractive -Command ^
  "$d='%WD_LOCK%'; $stale=[double]'%WD_STALE_S%'; if(Test-Path -LiteralPath $d){ $age=((Get-Date)-(Get-Item -LiteralPath $d).LastWriteTime).TotalSeconds; if($age -lt $stale){ exit 1 }; try{ Rename-Item -LiteralPath $d -NewName ((Split-Path -Leaf $d)+'.reclaim.'+[guid]::NewGuid().ToString('N')) -ErrorAction Stop }catch{ exit 1 } }; try{ New-Item -ItemType Directory -Path $d -ErrorAction Stop | Out-Null }catch{ exit 1 }; (Get-Item -LiteralPath $d).LastWriteTime=Get-Date; exit 0"
REM Tidy first, THEN decide. Previously this was after `if errorlevel 1`, so
REM the CAS LOSER - the one that renamed the stale lock and then failed
REM New-Item - exited before the cleanup and leaked its directory.
REM
REM But `rd` CLOBBERS the verdict: a successful rd leaves ERRORLEVEL alone,
REM while an rd on an already-deleted directory sets it to 2. A winner whose
REM reclaim dir was deleted by another racer between the glob and the rd would
REM therefore read 2 as "claim lost", print "watchdog already running", and exit
REM WITHOUT spawning while holding a fresh lock - dead until WD_STALE_S (900s
REM by default). So save the verdict first and decide on the saved value.
if errorlevel 1 (set "WD_CLAIM_FAILED=1") else (set "WD_CLAIM_FAILED=")
for /d %%R in ("%WD_LOCK%.reclaim.*") do rd /s /q "%%~R" 2>nul
if defined WD_CLAIM_FAILED exit /b 1
exit /b 0

:wd_beat
powershell -NoProfile -NonInteractive -Command "try{ (Get-Item -LiteralPath '%WD_LOCK%').LastWriteTime=Get-Date }catch{}" >nul 2>&1
exit /b 0

:ensure_comfy
set "LOGF=%LOGDIR%\comfy_watchdog.log"
set "TAG=comfy"
REM ponytail: a cold ComfyUI takes 2-4 minutes to scan 9k model paths and bind
REM :8188; this loop probed every 60s and re-launched every cycle, so two loops
REM (or one patient loop) multiplied servers that starved each other of RAM and
REM died - :8188 never came back. Stand down while a ComfyUI python exists or a
REM launch was requested in the last 300s; relaunch only when nothing is up.
REM The marker is LastWriteTime-stamped; age beyond 300s ignores it.
powershell -NoProfile -NonInteractive -Command "try { $m = Get-Item -LiteralPath '%LOGDIR%\comfy_launch_pending' -ErrorAction Stop; if (((Get-Date) - $m.LastWriteTime).TotalSeconds -lt 300) { exit 0 } } catch {}; $p = Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -like '*ComfyUI*main.py*' }; if ($p) { exit 0 }; exit 1" >nul 2>&1
if not errorlevel 1 (
  echo [%date% %time%] %TAG%: ComfyUI booting or launch pending, skipping relaunch >> "%LOGF%"
  exit /b 0
)
set "ATTEMPT=0"
:comfy_probe
set /a ATTEMPT+=1 >nul
curl -sf --max-time 5 http://127.0.0.1:8188/system_stats >nul 2>&1
if not errorlevel 1 (
  echo [%date% %time%] %TAG%: healthy >> "%LOGF%"
  del "%LOGDIR%\comfy_launch_pending" 2>nul
  exit /b 0
)
echo [%date% %time%] %TAG%: health probe %ATTEMPT% failed >> "%LOGF%"
if %ATTEMPT% GEQ 3 goto comfy_recovery
powershell -NoProfile -NonInteractive -Command "Start-Sleep -Seconds 3"
goto comfy_probe
:comfy_recovery
REM ponytail: the old guard here skipped recovery whenever a render was live,
REM because "a live render owns the VRAM" and booting Comfy Desktop underneath
REM one risks OOM. That deadlocked ch007 twice - :5003 stayed dead for ~40
REM minutes while the render sat on scene 22 and this loop logged "render live,
REM skipping recovery" every 60s, because the render was BLOCKED waiting on the
REM very service being skipped. :render_live now allows recovery when the port
REM has no listener at all: there is no process to disturb, and standing down is
REM always wrong. Booting ComfyUI under a live render is therefore now an
REM accepted trade - ComfyUI frees VRAM for the GPU phases, and the alternative
REM is a job that never finishes. If OOM ever shows up here, revisit.
call :render_live 8188
if errorlevel 1 exit /b 0
REM ponytail: `start "" Comfy Desktop.exe` launched the GUI app, which does not
REM bring the headless server up in a non-interactive session — so :8188 was
REM never restored by this loop and every render cycle re-discovered it dead
REM (the pipeline records the same lesson in _recall_comfy_launch()). Use the
REM recorded launch recipe, which is what the producer itself replays.
REM ponytail: Get-Content on a missing .comfy_launch.json is a NON-terminating
REM error, so $c is null, nothing launches, and the old code still echoed
REM "launch requested" and returned 0 - a false success. That mattered less
REM while a live render suppressed recovery outright; now that :render_live
REM deliberately recovers a dead port mid-render, a silent no-op would leave
REM exactly the deadlock this function was changed to prevent. Fail loudly.
REM ponytail: the launcher moved to comfy_launch.ps1 because this line used to
REM pass -WorkingDirectory $c.cwd straight through, and a cache entry with
REM cwd=null made Start-Process throw a parameter-binding error - so the
REM watchdog logged "no usable .comfy_launch.json, NOT launching" and :8188
REM stayed dead. The .ps1 recovers the cwd from the relative main.py argument
REM exactly like the producer's _recall_comfy_launch() does.
powershell -NoProfile -NonInteractive -ExecutionPolicy Bypass -File "%PROJ%\comfy_launch.ps1" >> "%LOGF%" 2>&1
if errorlevel 1 (
  echo [%date% %time%] %TAG%: no usable .comfy_launch.json, NOT launching >> "%LOGF%"
  exit /b 1
)
echo [%date% %time%] %TAG%: launch requested >> "%LOGF%"
REM pending marker: a cold ComfyUI needs 2-4 minutes; the guard in :ensure_comfy
REM stands down for 300s after a launch instead of re-launching every minute.
type nul > "%LOGDIR%\comfy_launch_pending"
exit /b 0

:ensure
set "PORT=%~1"
set "WORKDIR=%~2"
set "SCRIPT=%~3"
set "LOGF=%~4"
set "TAG=%~5"
set "HPATH=%~6"
REM ponytail: the 7th argument is a FLAG, not a value: pass the word
REM SKIP_RENDER to leave the service alone while a render is live (the
REM render starts and stops it itself, and resurrecting it steals VRAM
REM from the video model).
set "SKIP_RENDER=%~7"
set "EXPECTED=%WORKDIR%\%SCRIPT%"
set "LISTENER_PID="
set "ATTEMPT=0"
:probe
set /a ATTEMPT+=1 >nul
curl -sf --max-time 5 http://localhost:%PORT%%HPATH% >nul 2>&1
if not errorlevel 1 (
  echo [%date% %time%] %TAG%: healthy >> "%LOGF%"
  exit /b 0
)
echo [%date% %time%] %TAG%: health probe %ATTEMPT% failed >> "%LOGF%"
if %ATTEMPT% GEQ 3 goto recovery
powershell -NoProfile -NonInteractive -Command "Start-Sleep -Seconds 3"
goto probe
:recovery
  REM ponytail: a service the RENDER needs is the render's business. The
  REM producer starts the voice server itself (start_tts_server) before the
  REM voice phase, and it stops it again before the GPU phases to give the
  REM board back. The watchdog bringing it straight back stole ~3GB of VRAM
  REM from the video model, and ch018 lost ComfyUI mid-clip because of it.
  REM The LLM has the same rule (see :llm_recovery).
  set "SKIPPED="
  if defined SKIP_RENDER call :skip_during_render
  if errorlevel 1 set "SKIPPED=1"
  if defined SKIP_RENDER if defined SKIPPED exit /b 0
  call :render_live %PORT%
  if errorlevel 1 exit /b 0

set "BLOCKED="
for /f %%p in ('powershell -NoProfile -NonInteractive -Command "Get-NetTCPConnection -State Listen -ErrorAction Stop | Where-Object { $_.LocalPort -eq [int]$env:PORT } | ForEach-Object { $_.OwningProcess } | Sort-Object -Unique"') do call :recover_pid %%p
if defined BLOCKED exit /b 0
powershell -NoProfile -NonInteractive -Command "try { $listeners = @(Get-NetTCPConnection -State Listen -ErrorAction Stop); if ($listeners.LocalPort -contains [int]$env:PORT) { exit 1 }; exit 0 } catch { exit 2 }" >nul 2>&1
if errorlevel 1 (
  echo [%date% %time%] %TAG%: listener present or state unknown, skipping launch >> "%LOGF%"
  exit /b 0
)
call :render_live %PORT%
  if errorlevel 1 exit /b 0
  REM ponytail: DO NOT redirect cscript's own output into %LOGF%. run_hidden.vbs
REM opens that same path with CreateTextFile(args(2), True) to write its
REM diagnostic, and cmd's `>>` handle conflicts with the VBS's write handle —
REM "Permission denied" on every launch, so the dashboard could never be started
REM by the watchdog and always reported "launch requested" while dying. The VBS
REM already logs to %LOGF% itself, so cscript's extra output is discarded.
cscript //nologo "%PROJ%\run_hidden.vbs" "%PY%" "%WORKDIR%" "%LOGF%" "%EXPECTED%" >nul 2>&1
if errorlevel 1 (
  echo [%date% %time%] %TAG%: launch failed >> "%LOGF%"
  exit /b 1
)
echo [%date% %time%] %TAG%: launch requested >> "%LOGF%"
exit /b 0

:recover_pid
if defined BLOCKED exit /b 0
set "LISTENER_PID=%~1"
curl -sf --max-time 5 http://localhost:%PORT%%HPATH% >nul 2>&1
if not errorlevel 1 (
  echo [%date% %time%] %TAG%: healthy, skipping recovery >> "%LOGF%"
  set "BLOCKED=1"
  exit /b 0
)
call :render_live %PORT%
  if errorlevel 1 (
    set "BLOCKED=1"
    exit /b 0
  )
powershell -NoProfile -NonInteractive -Command "try { $p = Get-CimInstance Win32_Process -Filter ('ProcessId=' + [int]$env:LISTENER_PID) -ErrorAction Stop; $pattern = '(?i)(?:^|\s)' + [char]34 + [regex]::Escape($env:EXPECTED) + [char]34 + '(?=\s|$)'; if (!$p -or $p.Name -notin @('python.exe','pythonw.exe','python','wscript.exe') -or $p.CommandLine -notmatch $pattern) { exit 1 }; $listeners = @(Get-NetTCPConnection -State Listen -ErrorAction Stop); if (!($listeners | Where-Object { $_.LocalPort -eq [int]$env:PORT -and $_.OwningProcess -eq $p.ProcessId })) { exit 1 }; & taskkill.exe /PID $p.ProcessId /F; exit $LASTEXITCODE } catch { exit 2 }" >> "%LOGF%" 2>&1
if errorlevel 1 (
  echo [%date% %time%] %TAG%: PID %LISTENER_PID% foreign, unverified or recovery failed; skipping >> "%LOGF%"
  set "BLOCKED=1"
  exit /b 0
)
echo [%date% %time%] %TAG%: stopped verified unhealthy PID %LISTENER_PID% >> "%LOGF%"
exit /b 0

:ensure_llm
set "LOGF=%LOGDIR%\llm_watchdog.log"
set "TAG=llm"
set "ATTEMPT=0"
:llm_probe
set /a ATTEMPT+=1 >nul
curl -sf --max-time 5 http://127.0.0.1:8080/v1/models >nul 2>&1
if not errorlevel 1 (
  echo [%date% %time%] llm: healthy >> "%LOGF%"
  exit /b 0
)
echo [%date% %time%] llm: health probe %ATTEMPT% failed >> "%LOGF%"
if %ATTEMPT% GEQ 3 goto llm_recovery
powershell -NoProfile -NonInteractive -Command "Start-Sleep -Seconds 3"
goto llm_probe
:llm_recovery
  REM ponytail: do NOT bring the text model back while a render is live. The
  REM render stops it on purpose before the GPU phases, and the watchdog
  REM restarting it took the memory the video model needed: ch018 lost
  REM ComfyUI mid-clip (scene 1 degraded to a still-pan, scene 3 hung on a
  REM job whose ComfyUI had died) because the 13GB model was back on the
  REM board. The render starts the model itself when it needs it
  REM (ensure_llm), so leaving :8080 alone here costs nothing.
  call :render_active
  if errorlevel 2 (
    echo [%date% %time%] llm: render state unknown, leaving :8080 alone >> "%LOGF%"
    exit /b 0
  )
  if errorlevel 1 (
    echo [%date% %time%] llm: render live, leaving :8080 alone >> "%LOGF%"
    exit /b 0
  )
  call :render_live 8080
  if errorlevel 1 exit /b 0

REM ponytail: kill EVERY llama-server on :8080, not only the one holding the
REM listener. A relaunch that loses the bind race (the port was still held
REM when the new server tried to bind) keeps the model in RAM and NEVER
REM SERVES - so /v1/models on the port stays green and the watchdog sees
REM "healthy" forever while a zombie holds ~11 GB of system RAM. That is how
REM ComfyUI died for 90 minutes on 2026-10-20: RAM exhausted, the server
REM never came back, and no probe ever reported it.
powershell -NoProfile -NonInteractive -Command "try { $p = Get-CimInstance Win32_Process -Filter \"Name='llama-server.exe'\" -ErrorAction Stop; foreach ($proc in $p) { $cl = [string]$proc.CommandLine; $onPort = $cl -match '--port\s+8080'; $listeners = @(Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue); $isListener = @($listeners | Where-Object { $_.LocalPort -eq 8080 -and $_.OwningProcess -eq $proc.ProcessId }).Count -gt 0; if ($onPort -or $isListener) { & taskkill.exe /PID $proc.ProcessId /F } }; exit 0 } catch { exit 2 }" >> "%LOGF%" 2>&1
powershell -NoProfile -NonInteractive -Command "for ($i=0; $i -lt 30 -and @(Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue | Where-Object { $_.LocalPort -eq 8080 }).Count -gt 0; $i++) { Start-Sleep -Seconds 1 }" >> "%LOGF%" 2>&1
powershell -NoProfile -NonInteractive -Command "& '%PY%' '%PROJ%\hermes-skill\scripts\launch_llm.py'" >> "%LOGF%" 2>&1
set "LATTEMPT=0"
:llm_verify
set /a LATTEMPT+=1 >nul
curl -sf --max-time 5 http://127.0.0.1:8080/v1/models >nul 2>&1
if not errorlevel 1 (
  echo [%date% %time%] %TAG%: relaunched and healthy >> "%LOGF%"
  exit /b 0
)
if %LATTEMPT% GEQ 24 (
  echo [%date% %time%] %TAG%: relaunch unhealthy after 2min >> "%LOGF%"
  exit /b 1
)
powershell -NoProfile -NonInteractive -Command "Start-Sleep -Seconds 5"
goto llm_verify
:skip_during_render
REM 1 = a render is live (skip the recovery), 0 = none, 2 = state unknown.
call :render_active
if errorlevel 2 exit /b 1
if errorlevel 1 exit /b 1
exit /b 0

:render_active
REM 1 = a render is live, 0 = no render, 2 = state unknown.
REM ponytail: :render_live answers a DIFFERENT question ("is the port
REM dead"), so it cannot be reused here - the LLM recovery needs to know
REM whether a render owns the machine, not whether the port has a listener.
powershell -NoProfile -NonInteractive -Command "try { foreach ($name in @('render.pid','render.lock')) { $file = Join-Path $env:PROJ $name; if (Test-Path -LiteralPath $file) { $text = [IO.File]::ReadAllText($file).Trim(); if ($text) { $renderId = 0; if (![int]::TryParse($text, [ref]$renderId) -or $renderId -le 0) { exit 2 }; if (Get-Process -Id $renderId -ErrorAction SilentlyContinue) { exit 1 } } } }; exit 0 } catch { exit 2 }" >nul 2>&1
exit /b %errorlevel%

:render_live

REM ponytail: a DEAD service must be recovered EVEN MID-RENDER. The render is
REM blocked waiting on exactly this service, so skipping recovery deadlocks the
REM job until a human notices: ch007 sat on scene 22 of 40 for ~40 minutes with
REM :5003 down and the watchdog logging "render live, skipping recovery" every
REM 60s, and earlier the same thing happened to :8080. The guard exists to avoid
REM KILLING a service a live render is using; when the port has no listener there
REM is no process to disturb, and standing down is always the wrong call.
REM Pass the port; it defaults to the %PORT% :ensure already set.
set "_RP=%~1"
if not defined _RP set "_RP=%PORT%"
REM SilentlyContinue, NOT Stop: with a -LocalPort filter and nothing listening,
REM Get-NetTCPConnection raises a NON-terminating "No matching
REM MSFT_NetTCPConnection objects" error, and -ErrorAction Stop promotes it to
REM terminating -> catch -> exit 2. So the dead-port case - the one recovery
REM exists for - read as "cannot inspect" and stood down, and the watchdog could
REM not cold-start anything (every service down after a reboot: 3 failed probes
REM then "cannot inspect port X, skipping recovery", forever). A genuinely
REM broken CIM stack still throws and still fails CLOSED via the catch below;
REM SilentlyContinue does not mask it, because the exception is independent of
REM -ErrorAction.
powershell -NoProfile -NonInteractive -Command "try { exit [int](@(Get-NetTCPConnection -LocalPort %_RP% -State Listen -ErrorAction SilentlyContinue).Count -gt 0) } catch { exit 2 }" >nul 2>&1
if errorlevel 2 goto RL_UNKNOWN
if errorlevel 1 goto RL_RENDER_CHECK
echo [%date% %time%] %TAG%: port %_RP% has no listener, recovering despite live render >> "%LOGF%"
exit /b 0
:RL_UNKNOWN
REM Reached only when the CIM stack itself is broken (the cmdlet threw), not when
REM the port is merely dead. A dead port now falls through to the checks above.
echo [%date% %time%] %TAG%: cannot inspect port %_RP%, skipping recovery >> "%LOGF%"
exit /b 1
:RL_RENDER_CHECK
powershell -NoProfile -NonInteractive -Command "try { foreach ($name in @('render.pid','render.lock')) { $file = Join-Path $env:PROJ $name; if (Test-Path -LiteralPath $file) { $text = [IO.File]::ReadAllText($file).Trim(); if ($text) { $renderId = 0; if (![int]::TryParse($text, [ref]$renderId) -or $renderId -le 0) { exit 2 }; if (Get-Process -Id $renderId -ErrorAction SilentlyContinue) { exit 1 } } }; }; exit 0 } catch { exit 2 }" >nul 2>&1
if errorlevel 2 (
  echo [%date% %time%] %TAG%: render state unknown, skipping recovery >> "%LOGF%"
  exit /b 1
)
if errorlevel 1 (
  echo [%date% %time%] %TAG%: render live, skipping recovery >> "%LOGF%"
  exit /b 1
)
exit /b 0
