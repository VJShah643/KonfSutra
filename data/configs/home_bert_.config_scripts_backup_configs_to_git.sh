#!/bin/bash

# This script backs up all the configuration files to a git repository
#
# Author: Sourav Sharan

# Exit on any error
set -e

# Check if CONFIG_BACKUP_REPO is set
if [[ -z "$CONFIG_BACKUP_REPO" ]]; then
    echo "Error: CONFIG_BACKUP_REPO environment variable not set!"
    exit 1
fi

# Define machine-specific directory name
MACHINE_NAME=$(hostname)
MACHINE_DIR="$CONFIG_BACKUP_REPO/$MACHINE_NAME"

# Git repository URL
GIT_REPO_URL="git@github.com:SouravSharan/linux_configs.git"

# Clone or update the repo
if [[ ! -d "$CONFIG_BACKUP_REPO/.git" ]]; then
    echo "Cloning backup repository..."
    git clone "$GIT_REPO_URL" "$CONFIG_BACKUP_REPO"
    cd "$CONFIG_BACKUP_REPO"
else
    echo "Updating repository..."
    cd "$CONFIG_BACKUP_REPO" || { echo "Failed to enter $CONFIG_BACKUP_REPO"; exit 1; }
    git pull origin main || { echo "Failed to pull from origin"; exit 1; }
fi

# Ensure machine-specific directory exists
mkdir -p "$MACHINE_DIR"

# Use the config file from the environment variable or default location
BACKUP_CONFIG_PATHS="${BACKUP_CONFIG_PATHS:-$MACHINE_DIR/.config/scripts/backup_config_paths}"

# Check if config file exists
if [[ ! -f "$BACKUP_CONFIG_PATHS" ]]; then
    echo "Error: Backup configuration file $BACKUP_CONFIG_PATHS not found!"
    exit 1
fi

# Read each line from the config file
while IFS= read -r line || [[ -n "$line" ]]; do
    # Skip empty lines and comments
    [[ -z "$line" || "$line" =~ ^# ]] && continue

    # Expand $USER in paths if USER is set, otherwise use current user
    USER=${USER:-$(whoami)}
    expanded_line=$(echo "$line" | sed "s|\$USER|$USER|g")

    # Extract source and optional destination
    source=$(echo "$expanded_line" | awk '{print $1}')
    custom_dest=$(echo "$expanded_line" | awk '{print $2}')

    # Determine destination
    if [[ -n "$custom_dest" ]]; then
        destination="$MACHINE_DIR/$custom_dest"
    else
        if [[ "$source" == "/home/$USER"* ]]; then
            rel_path="${source#/home/$USER/}"
            destination="$MACHINE_DIR/.config/$rel_path"
        else
            rel_path="${source#/}"
            destination="$MACHINE_DIR/$rel_path"
        fi
    fi

    # Copy files if source exists
    if [[ -e "$source" ]]; then
        mkdir -p "$(dirname "$destination")"
        rsync -a --exclude='.git' "$source" "$destination"
        echo "Copied $source → $destination (excluding .git)"
    else
        echo "Warning: Source $source not found!"
    fi
done < "$BACKUP_CONFIG_PATHS"

# Commit and push changes if any
if [[ -n $(git status --porcelain) ]]; then
    git add .
    git commit -m "Backup for $MACHINE_NAME: $(date '+%Y-%m-%d %H:%M:%S')"
    git push origin main || { echo "Failed to push to origin"; exit 1; }
    echo "Backup successfully committed and pushed to GitHub!"
else
    echo "No changes to commit. Backup is up to date."
fi