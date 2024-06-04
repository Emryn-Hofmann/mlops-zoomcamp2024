#!/bin/bash

# Function to print the current time
print_time() {
    echo "Current time: $(date)"
}

# Infinite loop to print time every 30 minutes
while true; do
    print_time
    sleep 1800  # Sleep for 1800 seconds (30 minutes)
done
