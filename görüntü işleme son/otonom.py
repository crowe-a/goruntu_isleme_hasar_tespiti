from dronekit import Command, connect, VehicleMode, LocationGlobalRelative

from pymavlink import mavutil

connection_string="127.0.0.1:14550"
# connection_string="tcp:127.0.0.1:5763"

# iha = connect(connectinString, wait_ready=True)

#iha = connect('/dev/ttyAMA0', wait_ready=True, baud=57600)

#iha = connect(connection_string, baud=115200, wait_ready=True, timeout=60)

iha = connect(connection_string, wait_ready=False);iha.wait_ready(True,timeout=300)

