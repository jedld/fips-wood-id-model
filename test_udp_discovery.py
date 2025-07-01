#!/usr/bin/env python3
"""
Test script to verify UDP broadcasting from the Wood ID Upload Server.
This script listens for UDP broadcasts and displays server information.
"""

import socket
import json
import time
import threading

def listen_for_broadcasts():
    """Listen for UDP broadcasts from the Wood ID server"""
    
    # Create UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    
    # Bind to the broadcast port
    broadcast_port = 5001
    sock.bind(('', broadcast_port))
    
    print(f"🔍 Listening for UDP broadcasts on port {broadcast_port}")
    print("Make sure the Flask server is running with UDP broadcasting enabled")
    print("Press Ctrl+C to stop listening\n")
    
    discovered_servers = set()
    
    try:
        while True:
            try:
                # Receive broadcast message
                data, addr = sock.recvfrom(1024)
                message = data.decode('utf-8')
                
                # Parse JSON message
                try:
                    server_info = json.loads(message)
                    
                    if server_info.get('type') == 'wood_id_server':
                        server_key = f"{server_info['ip']}:{server_info['port']}"
                        
                        if server_key not in discovered_servers:
                            discovered_servers.add(server_key)
                            print("✅ New server discovered!")
                            print(f"   Name: {server_info['name']}")
                            print(f"   IP: {server_info['ip']}")
                            print(f"   Port: {server_info['port']}")
                            print(f"   Version: {server_info['version']}")
                            print(f"   From: {addr[0]}:{addr[1]}")
                            print(f"   Time: {time.strftime('%H:%M:%S')}")
                            print()
                        else:
                            # Server already discovered, just show heartbeat
                            print(f"💓 Heartbeat from {server_info['name']} ({server_info['ip']}) - {time.strftime('%H:%M:%S')}")
                    
                except json.JSONDecodeError as e:
                    print(f"❌ Invalid JSON received: {e}")
                    print(f"   Raw message: {message}")
                    print()
                    
            except socket.timeout:
                # Timeout is expected, continue listening
                continue
            except Exception as e:
                print(f"❌ Error receiving broadcast: {e}")
                
    except KeyboardInterrupt:
        print("\n🛑 Stopping UDP listener...")
    finally:
        sock.close()
        print("✅ UDP listener stopped")

def test_manual_broadcast():
    """Test function to manually send a broadcast message (for testing)"""
    
    # Create UDP socket for sending
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    
    # Test server info
    test_server_info = {
        "type": "wood_id_server",
        "ip": "192.168.1.100",
        "port": 5000,
        "name": "Test Wood ID Server",
        "version": "1.0"
    }
    
    message = json.dumps(test_server_info).encode('utf-8')
    
    try:
        sock.sendto(message, ('<broadcast>', 5001))
        print("📡 Sent test broadcast message")
    except Exception as e:
        print(f"❌ Error sending test broadcast: {e}")
    finally:
        sock.close()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        print("🧪 Testing UDP broadcast...")
        test_manual_broadcast()
    else:
        print("🌳 Wood ID Server UDP Discovery Test")
        print("=" * 40)
        listen_for_broadcasts() 