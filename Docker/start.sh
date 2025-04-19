#!/bin/bash

# Ensure ngrok config directory is created
mkdir -p /tmp/.ngrok2

# Save the authtoken to a custom config file
ngrok authtoken "2vxmnTTVzgK0BUZs9G4tkSRqcVw_7F8UY8Di42cAkDttigyxt" --config /tmp/.ngrok2/ngrok.yml

# Start ngrok in the background
ngrok http 8000 --config /tmp/.ngrok2/ngrok.yml &

# Wait for ngrok to establish a tunnel
sleep 5 # Wait for ngrok to initialize

# Check if ngrok tunnel is up by calling the ngrok API
while true; do
  TUNNEL_URL=$(curl -s http://127.0.0.1:4040/api/tunnels | jq -r '.tunnels[0].public_url')
  if [ "$TUNNEL_URL" != "null" ]; then
    echo "Ngrok tunnel established at $TUNNEL_URL"
    break
  fi
  echo "Waiting for ngrok tunnel..."
  sleep 1
done

# Start FastAPI app
echo "Starting FastAPI app..."
uvicorn Main:app --host 0.0.0.0 --port 8000
