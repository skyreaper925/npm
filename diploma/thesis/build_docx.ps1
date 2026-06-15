# Пересборка Word-версии (thesis.docx) после правок .tex.
# Запуск:  в PowerShell из этой папки:  ./build_docx.ps1
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$pandoc = Resolve-Path "..\tools\pandoc.exe"

python ..\tools\preprocess_cites.py          # развернуть \input + заменить \cite на [N]
& $pandoc thesis_docx.tex -o thesis.docx --reference-doc=reference.docx --resource-path=images --number-sections
python ..\tools\fix_docx.py thesis.docx      # заголовок «Список литературы» + нумерация источников
Write-Host "Готово: $(Resolve-Path thesis.docx)" -ForegroundColor Green
