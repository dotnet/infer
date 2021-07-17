<#
 .SYNOPSIS
    Makes API documentation for current version of Infer.NET.
 .DESCRIPTION
    Builds PrepareSource.csproj, creates API documentation using docfx for Infer.NET to the gh-pages branch.
#>

# Licensed to the .NET Foundation under one or more agreements.
# The .NET Foundation licenses this file to you under the MIT license.
# See the LICENSE file in the project root for more information.

$scriptDir = Split-Path -Path $MyInvocation.MyCommand.Definition -Parent

$sourceDirectory = [IO.Path]::GetFullPath((join-path $scriptDir '../../'))
$destinationDirectory = [IO.Path]::GetFullPath((join-path $scriptDir '../../InferNet_Copy_Temp/'))

$dotnetExe = 'dotnet'

Write-Host $sourceDirectory
Write-Host "Copy src and build to InferNet_Copy_Temp directory"
Copy-Item -Path "$sourceDirectory/src/" -Destination "$destinationDirectory/src/" -Recurse -Force
Copy-Item -Path "$sourceDirectory/build/" -Destination "$destinationDirectory/build/" -Recurse -Force

Write-Host "Copy root files to InferNet_Copy_Temp directory"
Get-ChildItem -Path $sourceDirectory -Filter "*.*" | Copy-Item -Destination $destinationDirectory -Force 


Write-Host "Build PrepareSource project"
if (!($dotnetExe))
{
    Write-Error -Message ("ERROR: Failed to use 'dotnet'")
    exit 1
}

$projPath = [IO.Path]::GetFullPath((join-path $sourceDirectory 'src/Tools/PrepareSource/Tools.PrepareSource.csproj'))
if (!(Test-Path $projPath)) {
    Write-Error -Message ('ERROR: Failed to locate PrepareSource project file at ' + $projPath)
    exit 1
}

& "$dotnetExe" build "$projPath" /p:Configuration=Release

Write-Host "Run PrepareSource for InferNet_Copy_Temp folder"
$prepareSourcePath = [IO.Path]::GetFullPath((join-path $sourceDirectory 'src/Tools/PrepareSource/bin/Release/net5.0/Microsoft.ML.Probabilistic.Tools.PrepareSource.dll'))
& "$dotnetExe" "$prepareSourcePath" "$destinationDirectory"

Write-Host "Install nuget package docfx.console"
Install-Package -Name docfx.console -provider Nuget -Source https://nuget.org/api/v2 -RequiredVersion 2.48.1 -Destination $scriptDir\..\..\packages -Force
Write-Host "Run docfx"
$docFXPath = [IO.Path]::GetFullPath((join-path $scriptDir '../../packages/docfx.console.2.48.1/tools/docfx.exe'))
$docFxJsonPath = "$scriptDir/../docfx.json"
& "$docFXPath" "$docFxJsonPath"
if($LASTEXITCODE)
{
    if(!(Invoke-Expression "& mono ""$docFXPath"" ""$docFxJsonPath"""))
    {
        Write-Error -Message ("ERROR: Unable to evaluate """ + $docFxCmd + """. Maybe Mono hasn't been installed")
    }
}

Write-Warning "Three warnings about invalid file links in toc.yml are expected and benign, because those files don't exist yet. However, the links are still set up correctly."

if ((Test-Path $destinationDirectory)) {
    Write-Host "Remove temp repository"
    Remove-Item -Path $destinationDirectory -Recurse -Force
}
$apiguideTmp = [IO.Path]::GetFullPath((join-path $sourceDirectory 'apiguide-tmp'))
if (!(Test-Path $apiguideTmp)) {
    Write-Host ("Couldn't find the folder """ + $apiguideTmp + """.")
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