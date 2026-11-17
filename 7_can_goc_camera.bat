@echo off
REM Can goc camera (test-time alignment): danh gia lai 3 mo hinh AutoPath tren TP.HCM va dung video demo moi
cd /d "%~dp0"
set PY=%USERPROFILE%\Anaconda3\envs\torch_env\python.exe
if not exist "%PY%" set PY=python
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set J=--custom-json hcmc\labels.json --custom-root hcmc --align-vp-json hcmc\vp.json
echo [1/2] Danh gia co can goc camera...
"%PY%" evaluate.py %J% --method autopath --ckpt runs\res18_aug\best.pt --out results\res18_aug_hcmc_align > logs\align_res18.txt 2>&1
"%PY%" evaluate.py %J% --method autopath --ckpt runs\res18_noaug\best.pt --out results\res18_noaug_hcmc_align > logs\align_res18_noaug.txt 2>&1
"%PY%" evaluate.py %J% --method autopath --ckpt runs\mbv3s_aug\best.pt --out results\mbv3s_aug_hcmc_align > logs\align_mbv3s.txt 2>&1
echo [2/2] Video demo co can goc camera (90 giay moi video)...
"%PY%" demo_video.py --video hcmc\raw\vid01.mp4 --ckpt runs\res18_aug\best.pt --align-vp 653 422 --start 1:00 --max-frames 2700 --warn --out results\demo_align\vid01_align.mp4 --stats-json results\demo_align\vid01.json > logs\align_demo.txt 2>&1
"%PY%" demo_video.py --video hcmc\raw\vid03.mp4 --ckpt runs\res18_aug\best.pt --align-vp 648 411 --start 1:00 --max-frames 2700 --warn --out results\demo_align\vid03_align.mp4 --stats-json results\demo_align\vid03.json >> logs\align_demo.txt 2>&1
"%PY%" demo_video.py --video hcmc\raw\vid04.mp4 --ckpt runs\res18_aug\best.pt --align-vp 633 471 --start 1:00 --max-frames 2700 --warn --out results\demo_align\vid04_align.mp4 --stats-json results\demo_align\vid04.json >> logs\align_demo.txt 2>&1
echo XONG > logs\done_align.txt
echo XONG. Xem results\demo_align
