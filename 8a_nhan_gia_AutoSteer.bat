@echo off
REM Buoc 1 cua tinh chinh: AutoSteer 2.0 gan nhan gia cho ~4.000 khung TP.HCM (bo qua +-8s quanh khung test va doan demo)
cd /d "%~dp0"
set PY=%USERPROFILE%\Anaconda3\envs\torch_env\python.exe
if not exist "%PY%" set PY=python
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
if not exist logs mkdir logs
echo Dang gan nhan gia (khoang 15-25 phut)...
"%PY%" tools\make_pseudo_labels.py --videos hcmc\raw\vid01.mp4 hcmc\raw\vid03.mp4 hcmc\raw\vid04.mp4 --out hcmc\pseudo --autosteer-repo auto_steer --autosteer-weights weights\autosteer.pth > logs\pseudo.txt 2>&1
"%PY%" tools\preview_pseudo.py --pseudo hcmc\pseudo --out hcmc\pseudo\preview >> logs\pseudo.txt 2>&1
echo XONG > logs\done_pseudo.txt
echo XONG. Xem hcmc\pseudo\preview
