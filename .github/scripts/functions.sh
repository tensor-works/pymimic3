# MongoDB database naming and access functions

function generate_db_name() {
  local branch_name="$1"
  local sha="$2"
  
  # Sanitize branch name and create unique database name
  local sanitized_branch=$(echo "$branch_name" | sed 's/[^a-zA-Z0-9]/_/g')
  echo "db_${sanitized_branch}_${sha}"
}

function setup_mongodb_connection() {
  local db_name="$1"
  
  # Set MongoDB connection details
  export MONGODB_HOST="${MONGODB_REGISTRY_HOST:-localhost}:27017"
  export MONGODB_DATABASE="$db_name"
  
  # Wait for MongoDB to be ready
  for i in {1..30}; do
    if mongosh --host "$MONGODB_HOST" --eval "db.adminCommand('ping')" >/dev/null 2>&1; then
      echo "MongoDB registry is available"
      return 0
    fi
    echo "Waiting for MongoDB registry... ($i/30)"
    sleep 1
  done
  
  echo "Error: MongoDB registry is not available"
  return 1
}

function cleanup_old_databases() {
    local keep_days=7
    local current_time=$(date +%s)
    local cutoff_time=$((current_time - (keep_days * 24 * 60 * 60)))

    # Get list of databases and remove old ones
    local dbs=$(mongosh --host "$MONGODB_HOST" --quiet --eval "db.adminCommand('listDatabases').databases.forEach(db => print(db.name))")

    for db in $dbs; do
        # Skip databases containing 'workspace'
        if [[ "$db" == *"workspace"* ]]; then
            continue
        fi

        if [[ "$db" =~ ^db_.*_[0-9a-f]{40}_.*$ ]]; then
            local db_time=$(echo "$db" | cut -d'_' -f3)
            if [[ $db_time -lt $cutoff_time ]]; then
                echo "Dropping old database: $db"
                mongosh --host "$MONGODB_HOST" --eval "db.getSiblingDB('$db').dropDatabase()"
            fi
        fi
    done
}

function get_sha() {
    basename "$(dirname "$1")"
}
