<#
 .SYNOPSIS
    Makes API documentation for current version of Infer.NET.
 .DESCRIPTION
    Builds PrepareSource.csproj, creates API documentation using docfx for Infer2 to docs/apiguide/ folder.
#>

# Licensed to the .NET Foundation under one or more agreements.
# The .NET Foundation licenses this file to you under the MIT license.
# See the LICENSE file in the project root for more information.

$scriptDir = Split-Path -Path $MyInvocation.MyCommand.Definition -Parent

$sourceDirectory = [IO.Path]::GetFullPath((join-path $scriptDir '../../'))
$destinationDirectory = [IO.Path]::GetFullPath((join-path $scriptDir '../../InferNet_Copy_Temp/'))
$excludes = @("InferNet_Copy_Temp", "docs", "packages", "_site", "build", "apiguide-tmp", ".git")

Write-Host "Copy subfolders to InferNet_Copy_Temp directory"
Get-ChildItem $sourceDirectory -Directory | 
    Where-Object{$_.Name -notin $excludes} | 
    Copy-Item -Destination $destinationDirectory -Recurse -Force

Write-Host "Copy root files to InferNet_Copy_Temp directory"
Get-ChildItem -Path $sourceDirectory -Include "*.*" | Copy-Item -Destination $destinationDirectory -Force 


Write-Host "Build PrepareSource project"
if ([Environment]::Is64BitOperatingSystem) {
    $pfiles = ${env:PROGRAMFILES(X86)}
} else {
    $pfiles = $env:PROGRAMFILES
}
$msBuildExe = Resolve-Path -Path "${pfiles}\Microsoft Visual Studio\*\*\MSBuild\15.0\bin\msbuild.exe" -ErrorAction SilentlyContinue
if (!($msBuildExe)) {
    $msBuildExe = Resolve-Path -Path "~/../../usr/bin/msbuild" -ErrorAction SilentlyContinue
    $useMono = "mono "
    if (!($msBuildExe)) {
        Write-Error -Message ('ERROR: Falied to locate MSBuild at' + $msBuildExe)
        exit 1
    }
}
if ($msbuildExe.GetType() -Eq [object[]]) {
  $msbuildExe = $msbuildExe | Select -index 0
}

$projPath = [IO.Path]::GetFullPath((join-path $scriptDir '../PrepareSource/PrepareSource.csproj'))
if (!(Test-Path $projPath)) {
    Write-Error -Message ('ERROR: Failed to locate PrepareSource project file at ' + $projPath)
    exit 1
}
$BuildArgs = @{
  FilePath = $msBuildExe
  ArgumentList = $projPath, "/t:rebuild", "/p:Configuration=Release", "/v:minimal" 
}
Start-Process @BuildArgs -NoNewWindow -Wait

Write-Host "Run PrepareSource for InferNet_Copy_Temp folder"
$prepareSourcePath = [IO.Path]::GetFullPath((join-path $scriptDir '../PrepareSource/bin/Release/PrepareSource.exe'))
$prepareSourceCmd = "& $useMono ""$prepareSourcePath"" ""$destinationDirectory"""
Invoke-Expression $prepareSourceCmd

Write-Host "Install nuget package docfx.console"
Install-Package -Name docfx.console -provider Nuget -Source https://nuget.org/api/v2 -RequiredVersion 2.38.0 -Destination $scriptDir\..\..\packages -Force
Write-Host "Run docfx"
$docFXPath = [IO.Path]::GetFullPath((join-path $scriptDir '../../packages/docfx.console.2.38.0/tools/docfx.exe'))
$docFxJsonPath = "$scriptDir/../docfx.json"
$docFxCmd = "& $useMono ""$docFXPath"" ""$docFxJsonPath"""
Invoke-Expression $docFxCmd

if ((Test-Path $destinationDirectory)) {
    Write-Host "Remove temp repository"
    Remove-Item -Path $destinationDirectory -Recurse -Force
}
$apiguideTmp = "./apiguide-tmp"
if (!(Test-Path $apiguideTmp)) {
    Write-Host "Couldn't find the folder \apiguide-tmp."
    exit 1
} else {
    Write-Host "Switch to gh-pages. All uncommited changes will be stashed."
    Try {
        git stash
        git checkout gh-pages

        $apiguidePath = "./apiguide"
        git pull origin gh-pages
        if ((Test-Path $apiguidePath)) {
            Remove-Item $apiguidePath -Force -Recurse
        } else {
            Write-Host "apiguide folder is not found."
        }
        Rename-Item -path ./apiguide-tmp -newName $apiguidePath 
        git add --all
        git commit -m "Update API Documentation"
        # git push origin gh-pages
    }
    Catch {
        Write-Host $Error
    }
}