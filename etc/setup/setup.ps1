# Setting up environment with OS type
$OSTYPE = [System.Environment]::OSVersion.Platform
Write-Output "Setting up environment with OS type: $OSTYPE"

$WORKINGDIR = Split-Path -Path $PSScriptRoot -Parent
$CONFIG = Join-Path $WORKINGDIR "etc"
$FRONTEND = Join-Path $WORKINGDIR "frontend"
$MODEL = Join-Path $WORKINGDIR "models"
$TESTS = Join-Path $WORKINGDIR "tests"
$EXAMPLES = Join-Path $WORKINGDIR "examples"


# Creating the conda environment
$envFile = Join-Path -Path $WORKINGDIR -ChildPath ".devcontainer/environment.yml"

$envs = & conda env list
if (-Not($envs -match "mimic3")) {
    Write-Output "Installing Libmamba solver"
    conda update -n base conda  # 24.3.0 at time of creation
    conda install -n base conda-libmamba-solver
    conda config --set solver libmamba
    Write-Output "Creating (mimic3) environment from $envFile"
    conda env create -f $envFile
} else {
    Write-Output "(mimic3) conda env already exists. Delete if you want to recreate."
}
conda activate mimic3

# Creating environment variables
$env:CONFIG = $CONFIG
$env:WORKINGDIR = $WORKINGDIR
$env:FRONTEND = $FRONTEND
$env:MODEL = $MODEL
$env:TESTS = $TESTS

$currentPaths = $env:PYTHONPATH -split ";"
$sourcePath = "$WORKINGDIR\src"

if ($sourcePath -notin $currentPaths) {
    $env:PYTHONPATH = "${env:PYTHONPATH};$sourcePath"
}

# Dotenv file helps navigate the directory 
function Set-DotEnvVariable {
    param (
        [string]$filePath,
        [string]$key,
        [string]$value
    )
    # Check if the file exists. If not, create the file.
    if (-not (Test-Path $filePath)) {
        New-Item -Path $filePath -ItemType File | Out-Null
    }

    $content = Get-Content $filePath -Raw

    $newLine = "${key}=${value}"
    if ($content -match "$key=") {
        $content = $content -replace "$key=.*", $newLine
    } else {
        if ($content -ne "") {
            $content += "`n$newLine"
        } else {
            $content = $newLine
        }
    }

    $content | Set-Content $filePath
}

if (-Not (Test-Path $WORKINGDIR\.env)) {
    Set-DotEnvVariable -filePath "$WORKINGDIR\.env" -key "WORKINGDIR" -value "$WORKINGDIR"
    Set-DotEnvVariable -filePath "$WORKINGDIR\.env" -key "CONFIG" -value "$CONFIG"
    Set-DotEnvVariable -filePath "$WORKINGDIR\.env" -key "MODEL" -value "$MODEL"
    Set-DotEnvVariable -filePath "$WORKINGDIR\.env" -key "TESTS" -value "$TESTS"
    Set-DotEnvVariable -filePath "$WORKINGDIR\.env" -key "EXAMPLES" -value "$EXAMPLES"
    Set-DotEnvVariable -filePath "$WORKINGDIR\.env" -key "PYTHONPATH" -value "$env:PYTHONPATH"
}

# Copy descriptor files to the resource dir
$destinationDir = Join-Path -Path $CONFIG -ChildPath "mimic3benchmark"
$outputVariableMap = Join-Path -Path $destinationDir -ChildPath "itemid_to_variable_map.csv"
$outputDefinitions = Join-Path -Path $destinationDir -ChildPath "hcup_ccs_2015_definitions.yaml"

if (-Not (Test-Path $outputVariableMap)) {
    Invoke-WebRequest "https://raw.githubusercontent.com/YerevaNN/mimic3-benchmarks/master/mimic3benchmark/resources/itemid_to_variable_map.csv" -o $outputVariableMap
}
if (-Not (Test-Path $outputDefinitions)) {
    Invoke-WebRequest "https://raw.githubusercontent.com/YerevaNN/mimic3-benchmarks/master/mimic3benchmark/resources/hcup_ccs_2015_definitions.yaml" -o $outputDefinitions
}