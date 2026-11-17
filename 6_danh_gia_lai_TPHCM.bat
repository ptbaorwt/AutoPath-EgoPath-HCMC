@echo off
cd /d "%~dp0"
set PY=%USERPROFILE%\Anaconda3\envs\torch_env\python.exe
if not exist "%PY%" set PY=python
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
if not exist logs mkdir logs
REM Gop nhan cua 4 thanh vien (hcmc\labels_*.json) thanh hcmc\labels.json
"%PY%" tools\merge_labels.py --hcmc hcmc
if not exist hcmc\labels.json (echo Chua co nhan - hay gan nhan truoc & pause & exit /b)
if not exist weights\autosteer.pth (
  if not exist weights mkdir weights
  "%PY%" -m gdown -q https://drive.google.com/uc?id=1-wfPyu7HId7YDSh_T_DkB3Ma1zPc7Fu_ -O weights\autosteer.pth
)
set J=--custom-json hcmc\labels.json --custom-root hcmc
echo [1/2] Danh gia tren tap TP.HCM...
"%PY%" evaluate.py %J% --method autopath --ckpt runs\res18_aug\best.pt --out results\res18_aug_hcmc > logs\hcmc_res18_aug.txt 2>&1
"%PY%" evaluate.py %J% --method autopath --ckpt runs\res18_noaug\best.pt --out results\res18_noaug_hcmc > logs\hcmc_res18_noaug.txt 2>&1
"%PY%" evaluate.py %J% --method autopath --ckpt runs\mbv3s_aug\best.pt --out results\mbv3s_aug_hcmc > logs\hcmc_mbv3s.txt 2>&1
"%PY%" evaluate.py %J% --method autosteer --autosteer-repo auto_steer --autosteer-weights weights\autosteer.pth --out results\autosteer_hcmc > logs\hcmc_autosteer.txt 2>&1
"%PY%" evaluate.py %J% --method classical --out results\classical_hcmc > logs\hcmc_classical.txt 2>&1
echo [2/2] Ve hinh va bang...
"%PY%" tools\make_figures.py --results "results/*_hcmc/results.json" --alpha-csv results\alpha_hcmc_mean.csv --out figures_hcmc > logs\figures_hcmc.txt 2>&1
echo XONG > logs\done_hcmc2.txt
echo XONG. Xem figures_hcmc\tables.md va thu muc results.
pause
