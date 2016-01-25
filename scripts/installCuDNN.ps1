$ZipSourcePath = $args[0]
if (-not $ZipSourcePath)
{
    Write-Host ('usage: installCuDNN.ps1 $downloadedZipFilePath') 
    EXIT
} elseif (-not (Test-Path $ZipSourcePath))
{
    Write-Host ('Please double check the CuDNN zip file path') 
    EXIT
}  

$CurrentDir = $(get-location).Path
Add-Type -A System.IO.Compression.FileSystem
$ZipTargetDir = $CurrentDir + '\cudnn'
[System.IO.Compression.ZipFile]::ExtractToDirectory($ZipSourcePath,$ZipTargetDir)

$CopySourceDir = $ZipTargetDir+'\cuda\*'
$CopyDestDir =  $env:CUDA_PATH
$ScriptBlockString = "Copy-Item -Path '$CopySourceDir' -Destination '$CopyDestDir' -Recurse -Force ; rm -r $ZipTargetDir"
$scriptblock = [scriptblock]::Create($ScriptBlockString)
$sh = new-object -com 'Shell.Application'
$sh.ShellExecute('powershell', "-Command $scriptblock", '', 'runas')







