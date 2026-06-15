# Сборка диплома в PDF без PyCharm.
# Запуск:  в PowerShell из этой папки:  ./build.ps1
#          либо правый клик по файлу -> "Run with PowerShell".
$ErrorActionPreference = 'Stop'
# Предпочитаем рабочий portable-MiKTeX (у старого сломан менеджер пакетов); запасной — старый.
$portable = "C:\Users\user\AppData\Local\Programs\MiKTeX-portable\texmfs\install\miktex\bin\x64"
$main     = "C:\Users\user\AppData\Local\Programs\MiKTeX\miktex\bin\x64"
$miktex   = if (Test-Path $portable) { $portable } else { $main }
$env:PATH = "$miktex;$env:PATH"
Set-Location $PSScriptRoot

Write-Host "Сборка thesis.pdf (2 прохода)..." -ForegroundColor Cyan
pdflatex -interaction=nonstopmode -output-directory=out thesis.tex | Out-Null
pdflatex -interaction=nonstopmode -output-directory=out thesis.tex | Out-Null

if (Test-Path out\thesis.pdf) {
    Copy-Item out\thesis.pdf thesis.pdf -Force
    Write-Host "Готово: $(Resolve-Path thesis.pdf)" -ForegroundColor Green
} else {
    Write-Host "Ошибка сборки — смотрите out\thesis.log" -ForegroundColor Red
    Select-String -Path out\thesis.log -Pattern '^!' | Select-Object -First 5
}
