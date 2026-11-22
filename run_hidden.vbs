' run_hidden.vbs — launch a python script with no visible window.
'
' Contract (see watchdog.bat :ensure, the only caller):
'   args(0) = interpreter path
'   args(1) = working directory
'   args(2) = log file
'   args(3) = script to run
'
' ponytail: all four are required. A 3-arg call used to run bare `python`
' in a hidden window: an interactive REPL nobody can see or close, holding
' a GPU box. WScript.Quit is checked here so the caller sees a real error
' instead of an async sh.Run that always reports success.
Dim args, cmd, sh, logfile, script, q, fso, f
Set args = WScript.Arguments
If args.Count < 4 Then WScript.Quit 1
Set sh = CreateObject("Wscript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
q = Chr(34)
sh.CurrentDirectory = args(1)
logfile = args(2)
script = args(3)

' Append, never truncate: this file holds the watchdog's diagnostics.
Dim lf
Set lf = fso.CreateTextFile(logfile, True)
lf.WriteLine "run_hidden: launching " & script
lf.Close

If Not fso.FileExists(script) Then
  Set lf = fso.CreateTextFile(logfile, True)
  lf.WriteLine "run_hidden: script not found: " & script
  lf.Close
  WScript.Quit 1
End If

cmd = q & args(0) & q & " -u " & q & script & q
cmd = q & sh.ExpandEnvironmentStrings("%ComSpec%") & q & " /d /s /c " & q & cmd & " >> " & q & logfile & q & " 2>&1" & q
sh.Run cmd, 0, False
