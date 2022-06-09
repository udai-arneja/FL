#!/bin/sh

osascript -e 'tell app "Terminal"
    do script "FYPC && python3 server.py"
    delay 8
    do script "FYPC && python3 client.py"
    do script "FYPC && python3 client.py"
    do script "FYPC && python3 client.py"
    do script "FYPC && python3 client.py"
    do script "FYPC && python3 client.py"
end tell'
