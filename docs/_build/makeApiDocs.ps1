<#
 .SYNOPSIS
    Makes API documentation for current version of Infer.NET.
 .DESCRIPTION
    Builds PrepareSource.csproj, creates API documentation using docfx for Infer to docs/apiguide/ folder.
#>

# Licensed to the .NET Foundation under one or more agreements.
# The .NET Foundation licenses this file to you under the MIT license.
# See the LICENSE file in the project root for more information.

$scriptDir = Split-Path -Path $MyInvocation.MyCommand.Definition -Parent

$sourceDirectory = [IO.Path]::GetFullPath((join-path $scriptDir '../../'))
$destinationDirectory = [IO.Path]::GetFullPath((join-path $scriptDir '../../InferNet_Copy_Temp/'))

$dotnetExe = 'dotnet'

Write-Host $sourceDirectory
Write-Host "Copy src to InferNet_Copy_Temp directory"
Copy-Item -Path "$sourceDirectory/src/" -Destination "$destinationDirectory/src/" -Recurse -Force

Write-Host "Copy root files to InferNet_Copy_Temp directory"
Get-ChildItem -Path $sourceDirectory -Filter "*.*" | Copy-Item -Destination $destinationDirectory -Force 


Write-Host "Build PrepareSource project"
if (!($dotnetExe))
{
    Write-Error -Message ("ERROR: Failed to use 'dotnet'")
    exit 1
}

$projPath = [IO.Path]::GetFullPath((join-path $sourceDirectory 'src/FactorDoc/PrepareSource/FactorDoc.PrepareSource.csproj'))
if (!(Test-Path $projPath)) {
    Write-Error -Message ('ERROR: Failed to locate PrepareSource project file at ' + $projPath)
    exit 1
}
$BuildArgs = @{
  FilePath = $dotnetExe
  ArgumentList = "build", $projPath, "/p:Configuration=Release"
}
Start-Process @BuildArgs -NoNewWindow -Wait

Write-Host "Run PrepareSource for InferNet_Copy_Temp folder"
$prepareSourcePath = [IO.Path]::GetFullPath((join-path $sourceDirectory 'src/FactorDoc/PrepareSource/bin/Release/netcoreapp2.1/Microsoft.ML.Probabilistic.FactorDoc.PrepareSource.dll'))
$prepareSourceCmd = "& $dotnetExe ""$prepareSourcePath"" ""$destinationDirectory"""
Invoke-Expression $prepareSourceCmd

Write-Host "Install nuget package docfx.console"
Install-Package -Name docfx.console -provider Nuget -Source https://nuget.org/api/v2 -RequiredVersion 2.38.0 -Destination $scriptDir\..\..\packages -Force
Write-Host "Run docfx"
$docFXPath = [IO.Path]::GetFullPath((join-path $scriptDir '../../packages/docfx.console.2.38.0/tools/docfx.exe'))
$docFxJsonPath = "$scriptDir/../docfx.json"
$docFxCmd = "& ""$docFXPath"" ""$docFxJsonPath"""
if(!(Invoke-Expression $docFxCmd))
{
    if(!(Invoke-Expression "& mono ""$docFXPath"" ""$docFxJsonPath"""))
    {
        Write-Error -Message ("ERROR: Unable to evaluate """ + $docFxCmd + """. Maybe Mono hasn't been installed")
    }
}

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