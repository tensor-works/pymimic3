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

# Download the MIMIC-III demo dataset from the web
$destinationDir = Join-Path -Path $testFolder -ChildPath "/data/physionet.org/files/mimiciii-demo/1.4/"

if (-Not (Test-Path $destinationDir)) {
    Download-FilesFromWebPage -sourceUrl $sourceUrl -destinationDir $destinationDir
    # Correcting defaults of the demo dataset
    Write-Output "Renaming erroneous demo dataset columns and dtypes "
    $convertScript = Join-Path -Path $benchmarkDir -ChildPath "convert_columns.py"
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
