# 调用Python脚本并传递参数
conda run -n fan_main python.exe F:\Study_Non-stationary_py\FAN-main\FAN-main\torch_timeseries\experiments\experiment.py "FEDformer" "FAN" "Electricity Traffic" "96 168 336 720" "cuda:0" 96 "{freq_topk:4}"
Write-Host "All operations completed."
