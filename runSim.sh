#!/bin/sh

osascript -e 'tell app "Terminal"
    do script "FYPC2 && python3 server.py"
    delay 10
    do script "FYPC2 && python3 client.py"
    do script "FYPC2 && python3 client.py"
    do script "FYPC2 && python3 client.py"
    do script "FYPC2 && python3 client.py"
    do script "FYPC2 && python3 client.py"
end tell'

'tell application "Terminal" to if it is running then quit'