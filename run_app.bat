@echo off
setlocal
cd /d "%~dp0"
title Yagli Boya Atolyesi 8.0
"C:\Users\MARK_A\AppData\Local\Python\bin\python.exe" "%~dp0main.py"
if errorlevel 1 pause
