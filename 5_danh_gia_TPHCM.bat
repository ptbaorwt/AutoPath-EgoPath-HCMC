@echo off
REM Danh gia 5 phuong phap tren tap TP.HCM, quet alpha, dung video demo. Ket qua: results\*_hcmc, figures_hcmc
cd /d "%~dp0"
set PY=%USERPROFILE%\Anaconda3\envs\torch_env\python.exe
if not exist "%PY%" set PY=python
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
if not exist logs mkdir logs
REM Gop nhan cua 4 thanh vien (hcmc\labels_*.json) thanh hcmc\labels.json
"%PY%" tools\merge_labels.py --hcmc hcmc
if not exist hcmc\labels.json (echo Chua co nhan - hay gan nhan truoc & pause & exit /b)
"%PY%" -m pip install -q gdown ultralytics
if not exist weights\autosteer.pth (
  if not exist weights mkdir weights
  "%PY%" -m gdown -q https://drive.google.com/uc?id=1-wfPyu7HId7YDSh_T_DkB3Ma1zPc7Fu_ -O weights\autosteer.pth
)
REM Doan video dung cho chon alpha va demo: bat dau tu phut 1, dai 3 phut (5400 khung o 30 FPS)
set CLIP_START=1:00
set CLIP_FRAMES=5400
set J=--custom-json hcmc\labels.json --custom-root hcmc
echo [1/4] Danh gia tren tap TP.HCM...
"%PY%" evaluate.py %J% --method autopath --ckpt runs\res18_aug\best.pt --out results\res18_aug_hcmc > logs\hcmc_res18_aug.txt 2>&1
"%PY%" evaluate.py %J% --method autopath --ckpt runs\res18_noaug\best.pt --out results\res18_noaug_hcmc > logs\hcmc_res18_noaug.txt 2>&1
"%PY%" evaluate.py %J% --method autopath --ckpt runs\mbv3s_aug\best.pt --out results\mbv3s_aug_hcmc > logs\hcmc_mbv3s.txt 2>&1
"%PY%" evaluate.py %J% --method autosteer --autosteer-repo auto_steer --autosteer-weights weights\autosteer.pth --out results\autosteer_hcmc > logs\hcmc_autosteer.txt 2>&1
"%PY%" evaluate.py %J% --method classical --out results\classical_hcmc > logs\hcmc_classical.txt 2>&1
echo [2/4] Quet alpha tren tung video (Thi nghiem 4)...
for %%v in (hcmc\raw\*.mp4) do "%PY%" demo_video.py --video "%%v" --ckpt runs\res18_aug\best.pt --start %CLIP_START% --max-frames %CLIP_FRAMES% --alphas 0.1 0.2 0.35 0.5 0.7 1.0 --no-write --out results\alpha\%%~nv.mp4 >> logs\alpha.txt 2>&1
"%PY%" tools\mean_alpha_csv.py results\alpha_hcmc_mean.csv "results/alpha/*_alpha_sweep.csv" >> logs\alpha.txt 2>&1
echo [3/4] Dung video demo (co canh bao vat can)...
for %%v in (hcmc\raw\*.mp4) do "%PY%" demo_video.py --video "%%v" --ckpt runs\res18_aug\best.pt --start %CLIP_START% --max-frames %CLIP_FRAMES% --alpha 0.35 --warn --out results\demo\%%~nv_demo.mp4 --stats-json results\demo\%%~nv_stats.json >> logs\demo.txt 2>&1
echo [4/4] Ve hinh va bang...
"%PY%" tools\make_figures.py --results "results/*_hcmc/results.json" --alpha-csv results\alpha_hcmc_mean.csv --out figures_hcmc > logs\figures_hcmc.txt 2>&1
echo XONG > logs\done_hcmc.txt
echo XONG. Xem figures_hcmc\tables.md va thu muc results.
pause
