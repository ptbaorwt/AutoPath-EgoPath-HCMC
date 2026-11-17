@echo off
REM Buoc 2: tinh chinh AutoPath-ResNet18 tren 3.830 khung TP.HCM co nhan gia (AutoSteer lam thay), danh gia tren 187 khung test, dung video demo
cd /d "%~dp0"
set PY=%USERPROFILE%\Anaconda3\envs\torch_env\python.exe
if not exist "%PY%" set PY=python
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
echo [1/3] Tinh chinh (khoang 20-40 phut)...
"%PY%" tools\finetune_hcmc.py --init runs\res18_aug\best.pt --pseudo hcmc\pseudo --name res18_ft_pseudo --epochs 15 --batch 32 > logs\ft_pseudo.txt 2>&1
echo [2/3] Danh gia tren 187 khung test TP.HCM...
"%PY%" evaluate.py --custom-json hcmc\labels.json --custom-root hcmc --method autopath --ckpt runs\res18_ft_pseudo\best.pt --out results\res18_ft_pseudo_hcmc > logs\eval_ft_pseudo.txt 2>&1
echo [3/3] Video demo (90 giay moi video)...
"%PY%" demo_video.py --video hcmc\raw\vid01.mp4 --ckpt runs\res18_ft_pseudo\best.pt --start 1:00 --max-frames 2700 --warn --out results\demo_ft\vid01_ft.mp4 --stats-json results\demo_ft\vid01.json > logs\demo_ft.txt 2>&1
"%PY%" demo_video.py --video hcmc\raw\vid03.mp4 --ckpt runs\res18_ft_pseudo\best.pt --start 1:00 --max-frames 2700 --warn --out results\demo_ft\vid03_ft.mp4 --stats-json results\demo_ft\vid03.json >> logs\demo_ft.txt 2>&1
"%PY%" demo_video.py --video hcmc\raw\vid04.mp4 --ckpt runs\res18_ft_pseudo\best.pt --start 1:00 --max-frames 2700 --warn --out results\demo_ft\vid04_ft.mp4 --stats-json results\demo_ft\vid04.json >> logs\demo_ft.txt 2>&1
echo XONG > logs\done_ft.txt
echo XONG.
