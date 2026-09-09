"""Task-local SSH multiplexing, avoiding repeated agent authentication for polling."""
from store import ROOT


def command():
    return ['ssh', '-S', str(ROOT/'ha-status.sock'),
            '-o', 'ControlMaster=auto', '-o', 'ControlPersist=8h',
            '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=3',
            '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', 'root@192.168.1.202']
