#!/bin/bash
# Script to visualize the last two experiments

cd "$(dirname "$0")"

# Check if Docker is available
if command -v docker &> /dev/null; then
    echo "Running visualization in Docker..."
    docker run --rm \
        -v "$(pwd)/experiments:/app/experiments" \
        -v "$(pwd)/utils:/app/utils" \
        federated-test \
        python utils/visualize_experiments.py experiments
    
    if [ -f "experiments/comparison_last_two.png" ]; then
        echo ""
        echo "✓ Visualization created: experiments/comparison_last_two.png"
        echo ""
        echo "To view the image:"
        if [[ "$OSTYPE" == "darwin"* ]]; then
            echo "  open experiments/comparison_last_two.png"
        elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
            echo "  xdg-open experiments/comparison_last_two.png"
        fi
    fi
else
    echo "Docker not found. Trying to run with local Python..."
    python3 utils/visualize_experiments.py experiments
fi

