#!/bin/bash

# Install async database drivers and other dependencies
echo "Installing async database drivers and dependencies..."
pip install -r requirements-async.txt

# Check if installation was successful
if [ $? -eq 0 ]; then
    echo "Installation successful!"
    echo "You can now run the application with: python run.py"
else
    echo "Installation failed. Please check the error messages above."
fi 