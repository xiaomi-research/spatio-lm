#!/bin/bash -e

# Check if required environment variables are set
if [ -z "$GIT_USER" ] || [ -z "$GIT_TOKEN" ] || [ -z "$GIT_REPO" ] || [ -z "$COMMIT_ID" ]; then
  echo "Error: Required environment variables (GIT_USER, GIT_TOKEN, GIT_REPO, COMMIT_ID) are not set."
  exit 1
fi

TARGET_DIR="${TARGET_DIR:-/root/code/}"

# Clone the repository into the target directory
git config --global credential.helper store
echo "Cloning repository into $TARGET_DIR..."
git clone https://"$GIT_USER":"$GIT_TOKEN"@"$GIT_REPO" "$TARGET_DIR"

# Check if the clone was successful
if [ $? -ne 0 ]; then
  echo "Error: Failed to clone the repository."
  exit 1
fi

# Change to the repository directory
cd "$TARGET_DIR"

# Checkout the specific commit
echo "Checking out commit $COMMIT_ID..."
git fetch origin "$COMMIT_ID"
git switch -c "${COMMIT_ID:0:8}" "$COMMIT_ID"

# Check if the checkout was successful
if [ $? -ne 0 ]; then
  echo "Error: Failed to checkout the commit $COMMIT_ID."
  exit 1
fi

echo "Repository successfully cloned, checked out to commit $COMMIT_ID, and submodules updated in folder $TARGET_DIR."
