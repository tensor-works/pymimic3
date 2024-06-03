function Download-FilesFromWebPage([string]$sourceUrl, [string]$destinationDir) {
    # Ensure WebClient is available for downloading
    $webClient = New-Object System.Net.WebClient

    # Ensure the destination directory exists
    if (!(Test-Path -Path $destinationDir)) {
        New-Item -Path $destinationDir -ItemType Directory -Force | Out-Null
    }

    # Download the web page content
    $webPageContent = $webClient.DownloadString($sourceUrl)

    # Match all 'a' elements and extract the URLs
    $matches = [Regex]::Matches($webPageContent, '<a href="([^"]+)">')
    foreach ($match in $matches) {
        $relativeUrl = $match.Groups[1].Value

        # Skip the parent directory link
        if ($relativeUrl -eq "../") {
            continue
        }

        $fileName = [System.IO.Path]::GetFileName($relativeUrl)
        $fileUrl = $sourceUrl + $relativeUrl
        $destinationPath = Join-Path -Path $destinationDir -ChildPath $fileName

        try {
            $webClient.DownloadFile($fileUrl, $destinationPath)
            Write-Host "Downloaded '$fileName' to '$destinationPath'"
        } catch {
            Write-Warning "Failed to download '$fileUrl': $_"
        }
    }
}

# Create the existing folder index
$sourceUrl = "https://physionet.org/files/mimiciii-demo/1.4/"
$testFolder = Split-Path -Path $PSScriptRoot -Parent
$etcDir = Join-Path -Path $testFolder -ChildPath "etc"

# Download the MIMIC-III demo dataset from the web
$destinationDir = Join-Path -Path $testFolder -ChildPath "/data/physionet.org/files/mimiciii-demo/1.4/"
Write-Output "Renaming erroneous demo dataset columns and dtypes "

if (-Not (Test-Path $destinationDir)) {
    Download-FilesFromWebPage -sourceUrl $sourceUrl -destinationDir $destinationDir
    # Correcting defaults of the demo dataset
    Write-Output "Renaming erroneous demo dataset columns and dtypes "
    $convertScript = Join-Path -Path $destinationDir -ChildPath "convert_columns.py"
    Copy-Item (Join-Path $etcDir -ChildPath "convert_columns.py") -Destination $convertScript
    python $convertScript
}

# Copy descriptor files to the resource dir
$resourceDir = Join-Path -Path $destinationDir -ChildPath "resources"
$outputVariableMap = Join-Path -Path $resourceDir -ChildPath "itemid_to_variable_map.csv"
Write-Output "Downloading item IDs to: $outputVariableMap"
$outputDefinitions = Join-Path -Path $resourceDir -ChildPath "hcup_ccs_2015_definitions.yaml"
Write-Output "Downloading definitions to: $outputDefinitions"

if (-Not (Test-Path $resourceDir)) {
    New-Item -Path $resourceDir -ItemType Directory
}
if (-Not (Test-Path $outputVariableMap)) {
    Invoke-WebRequest "https://raw.githubusercontent.com/YerevaNN/mimic3-benchmarks/master/mimic3benchmark/resources/itemid_to_variable_map.csv" -o $outputVariableMap
}
if (-Not (Test-Path $outputDefinitions)) {
    Invoke-WebRequest "https://raw.githubusercontent.com/YerevaNN/mimic3-benchmarks/master/mimic3benchmark/resources/hcup_ccs_2015_definitions.yaml" -o $outputDefinitions
}

# Download the MIMIC-III benchmarks dataset from github if necessary
$benchmarkDir = Join-Path -Path $testFolder -ChildPath "data/mimic3benchmarks"
$generatedDir = Join-Path -Path $testFolder -ChildPath "data/generated-benchmark"
Write-Output "Benchmark dir $benchmarkDir"
if (-Not (Test-Path $generatedDir)) {
    Write-Output "Downloading MIMIC-III benchmarks dataset from github"
    git clone "https://github.com/YerevaNN/mimic3-benchmarks.git" $benchmarkDir
}

# Define the source and destination directory paths
$benchmarkDir = Join-Path -Path $testFolder -ChildPath "data/mimic3benchmarks"

# Define a dictionary of file names and iterate over them to set paths and copy items
$fileNames = @("convert_columns.py", "rename_files.py", "revert_split.py", "engineer_data.py", "discretize_data.py")

foreach ($fileName in $fileNames) {
    # Construct source and destination paths
    $sourcePath = Join-Path -Path $etcDir -ChildPath $fileName
    $destinationPath = Join-Path -Path $benchmarkDir -ChildPath $fileName

    # Copy file from source to destination
    Copy-Item -Path $sourcePath -Destination $destinationPath
}

# Define the paths to the scripts for the MIMIC-III benchmarks dataset processing
$renameScript = Join-Path -Path $benchmarkDir -ChildPath "rename_files.py"
$revertSplitScript = Join-Path -Path $benchmarkDir -ChildPath "revert_split.py"
$engineScript = Join-Path -Path $benchmarkDir -ChildPath "engineer_data.py"
$discretizerScript = Join-Path -Path $benchmarkDir -ChildPath "discretize_data.py"




# Change into the MIMIC-III benchmarks directory
$currentDirectory = Get-Location
Set-Location -Path $benchmarkDir

$extractedDir = Join-Path -Path $generatedDir -ChildPath "extracted"    

# Run the MIMIC-III benchmarks dataset processing
if (-Not (Test-Path $extractedDir)) {
    New-item $extractedDir -ItemType Directory
    Write-Output "Extracting subject informations and timeseries data using original MIMIC-III github"
    python -m mimic3benchmark.scripts.extract_subjects $destinationDir $extractedDir
    python -m mimic3benchmark.scripts.validate_events $extractedDir
    python -m mimic3benchmark.scripts.extract_episodes_from_subjects $extractedDir
    Write-Output "Renaming files to include icustay_id in the filename"
    python $renameScript
    python -m mimic3benchmark.scripts.split_train_and_test $extractedDir
}

$processedDir = Join-Path -Path $generatedDir -ChildPath "processed"
$ihmProcessed = Join-Path -Path $processedDir -ChildPath "in-hospital-mortality"
$decompProcessed = Join-Path -Path $processedDir -ChildPath "decompensation"
$losProcessed = Join-Path -Path $processedDir -ChildPath "length-of-stay"
$phenotypingProcessed = Join-Path -Path $processedDir -ChildPath "phenotyping"

$processedTaskDirs = @( $ihmProcessed, $decompProcessed, $losProcessed, $phenotypingProcessed )

if (-Not (Test-Path $ihmProcessed) -Or -Not (Test-Path $decompProcessed) -Or -Not (Test-Path $losProcessed) -Or -Not (Test-Path $phenotypingProcessed)) {
    
    Write-Output "Processing task data using the original MIMIC-III github"
    foreach ($taskDir in $processedTaskDirs) {
        if (-Not (Test-Path $taskDir)) {
            New-item $taskDir -ItemType Directory
        }
    }
    python -m mimic3benchmark.scripts.create_in_hospital_mortality $extractedDir $ihmProcessed
    python -m mimic3benchmark.scripts.create_decompensation $extractedDir $decompProcessed
    python -m mimic3benchmark.scripts.create_length_of_stay $extractedDir $losProcessed
    python -m mimic3benchmark.scripts.create_phenotyping $extractedDir $phenotypingProcessed
    python -m mimic3benchmark.scripts.create_multitask $extractedDir "../generated-benchmark/processed/multitask/"
    Write-Output "Reverting the dataset split from original MIMIC-III github"
    python $revertSplitScript
}

$engineeredDir = Join-Path -Path $generatedDir -ChildPath "engineered"
$ihmEngineered = Join-Path -Path $engineeredDir -ChildPath "in-hospital-mortality"
$decompEngineered = Join-Path -Path $engineeredDir -ChildPath "decompensation"
$losEngineered = Join-Path -Path $engineeredDir -ChildPath "length-of-stay"
$phenoEngineered = Join-Path -Path $engineeredDir -ChildPath "phenotyping"

$engineeredTaskDirs = @( $ihmEngineered, $decompEngineered, $losEngineered, $phenoEngineered )

if (-Not (Test-Path $ihmEngineered) -Or -Not (Test-Path $decompEngineered) -Or -Not (Test-Path $losEngineered) -Or -Not (Test-Path $phenoEngineered)) {
    Write-Output "Engineering task data using the original MIMIC-III github"
    if (-Not (Test-Path $engineeredDir)) {
        New-item $engineeredDir -ItemType Directory
    }
    Write-Output "$engineScript"
    python $engineScript
}

$discretizedDir = Join-Path -Path $generatedDir -ChildPath "discretized"
$ihmDiscretized = Join-Path -Path $discretizedDir -ChildPath "in-hospital-mortality"
$decompDiscretized = Join-Path -Path $discretizedDir -ChildPath "decompensation"
$losDiscretized = Join-Path -Path $discretizedDir -ChildPath "length-of-stay"
$phenoDiscretized = Join-Path -Path $discretizedDir -ChildPath "phenotyping"

if (-Not (Test-Path $ihmDiscretized) -Or -Not (Test-Path $decompDiscretized) -Or -Not (Test-Path $losDiscretized) -Or -Not (Test-Path $phenoDiscretized)) {
    Write-Output "Discretizing task data using the original MIMIC-III github"
    if (-Not (Test-Path $discretizedDir)) {
        New-item $discretizedDir -ItemType Directory
    }
    Write-Output "$discretizerScript"
    python $discretizerScript
}


Set-Location -Path $currentDirectory

Write-Output "Removing MIMIC-III benchmarks dataset from github"
# Remove-Item -Path $generatedDir -Force -Recurse
