"""
Industrial Sensor Telemetry MQTT Test Publisher (Phase 1).
Simulates real factory sensor traffic and publishes to MQTT broker.
Can be executed locally, on a Raspberry Pi, or pointing to a public/private broker.
"""
import json
import random
import time
import argparse
import paho.mqtt.client as mqtt


def run_publisher(broker: str = "127.0.0.1", port: int = 1883, topic: str = "factory/node_alpha/telemetry", interval: float = 1.0):
    client = mqtt.Client()
    try:
        client.connect(broker, port, 60)
        print(f"[MQTT Publisher] Connected to {broker}:{port}, publishing to topic: '{topic}'...")
    except Exception as e:
        print(f"[MQTT Publisher] Failed to connect to {broker}:{port}: {e}")
        return

    sample_count = 0
    while True:
        sample_count += 1
        # Realistic industrial telemetry distribution
        is_stress = (sample_count % 30) > 20
        base_throughput = 30.0 if is_stress else 55.0
        
        payload = {
            "device_id": "PLC_NODE_ALPHA",
            "throughput_mbps": max(5.0, round(base_throughput + random.gauss(0, 6), 2)),
            "packet_drop_percentage": max(0.01, round(random.gauss(1.8 if is_stress else 0.25, 0.15), 3)),
            "buffer_utilization_percentage": round(min(98.0, random.uniform(65.0 if is_stress else 25.0, 90.0 if is_stress else 50.0)), 2),
            "node_temperature_celsius": round(45.0 + (15.0 if is_stress else 0.0) + random.gauss(0, 2.0), 2),
            "timestamp": time.time()
        }
        
        msg_str = json.dumps(payload)
        client.publish(topic, msg_str)
        print(f"[MQTT] TX #{sample_count} -> {topic}: {msg_str}")
        time.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HELM-IIoT MQTT Telemetry Publisher")
    parser.add_argument("--broker", default="127.0.0.1", help="MQTT Broker Host (e.g., 127.0.0.1 or test.mosquitto.org)")
    parser.add_argument("--port", type=int, default=1883, help="MQTT Broker Port")
    parser.add_argument("--topic", default="factory/node_alpha/telemetry", help="MQTT Topic")
    parser.add_argument("--interval", type=float, default=1.0, help="Publish interval in seconds")
    args = parser.parse_args()
    
    run_publisher(broker=args.broker, port=args.port, topic=args.topic, interval=args.interval)
