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





