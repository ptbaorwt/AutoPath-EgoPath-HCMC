@echo off
REM Do toc do va kiem thu tren laptop RTX 3050. Ket qua: thu muc results\ va logs\
cd /d "%~dp0"
set PY=%USERPROFILE%\Anaconda3\envs\torch_env\python.exe
if not exist "%PY%" set PY=python
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
if not exist logs mkdir logs
set CLIP=hcmc\clips_small\vid01_clip.mp4
if not exist %CLIP% set CLIP=_test\fake\fake_drive.mp4
"%PY%" -c "import torch;print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'KHONG CO CUDA')" > logs\gpu.txt 2>&1
type logs\gpu.txt
"%PY%" -c "import ultralytics" 2>nul || "%PY%" -m pip install -q ultralytics
echo [1/4] Do toc do mo hinh (khoang 2-3 phut)...
"%PY%" benchmark_fps.py --ckpts runs\res18_aug\best.pt runs\mbv3s_aug\best.pt --autosteer-repo auto_steer --classical --out results\benchmark_rtx3050.json > logs\benchmark_rtx3050.txt 2>&1
echo [2/4] Do FPS toan he thong tren video TP.HCM (khoang 5 phut)...
"%PY%" demo_video.py --video %CLIP% --ckpt runs\res18_aug\best.pt --max-frames 1500 --out results\sys_res18.mp4 --stats-json results\sys_res18.json > logs\sysfps_res18.txt 2>&1
"%PY%" demo_video.py --video %CLIP% --ckpt runs\res18_aug\best.pt --max-frames 1500 --warn --out results\sys_res18_warn.mp4 --stats-json results\sys_res18_warn.json > logs\sysfps_res18_warn.txt 2>&1
"%PY%" demo_video.py --video %CLIP% --ckpt runs\mbv3s_aug\best.pt --max-frames 1500 --out results\sys_mbv3s.mp4 --stats-json results\sys_mbv3s.json > logs\sysfps_mbv3s.txt 2>&1
"%PY%" demo_video.py --video %CLIP% --method autosteer --autosteer-repo auto_steer --max-frames 1500 --out results\sys_autosteer.mp4 --stats-json results\sys_autosteer.json > logs\sysfps_autosteer.txt 2>&1
echo [3/4] Kiem thu chuc nang voi mo hinh that...
"%PY%" tools\run_functional_tests.py --ckpt runs\res18_aug\best.pt --autosteer-repo auto_steer --out results\functest > logs\functest.txt 2>&1
echo [4/4] Ve hinh...
"%PY%" tools\make_figures.py --runs "runs/*" --results "results/*/results.json" --benchmark results\benchmark_rtx3050.json --out figures_rtx3050 > logs\figures.txt 2>&1
echo XONG > logs\done_rtx3050.txt
echo Da xong. Xem thu muc logs va results.
pause
