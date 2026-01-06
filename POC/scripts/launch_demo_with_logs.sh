#!/bin/bash
xterm -geometry 62x53+1300 -fa 'Monospace' -fs 12 -T "0117-NUT logs" -e "bash -c 'python scan_and_dump.py; exec bash'"
#lxterminal doesn't support x+y offset on RPI5, use xterm instead
#lxterminal --title="0117-NUT logs" --working-directory=/home/bluegriot/Desktop/0117-NUT/0117-NUT-Firmware/firmware/POC/scripts --geometry=82x60+1250+35 -e "bash -c 'python scan_and_dump.py; exec bash'"
