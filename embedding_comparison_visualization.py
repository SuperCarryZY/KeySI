# udp_probe_client.py
import socket, time, statistics

SERVER_IP   = "100.102.167.107"  # ← 改成 Windows 的 Tailscale IP 或公网 IP
SERVER_PORT = 7070               # ← 与服务器一致
INTERVAL    = 1.0                # 发送间隔秒
WARM_IDLE   = 20                 # 首轮发送 N 个包后，空闲 N 秒，测试NAT老化
RUN_SECONDS = 120                # 总时长（不含 idle），可加大

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.settimeout(2.0)

seq, sent, recv = 0, 0, 0
rtts = []
start = time.time()
phase = "warmup"

def send_one(seq):
    global sent
    ts = time.time()
    payload = f"PING seq={seq} cli_ts={ts}".encode()
    sock.sendto(payload, (SERVER_IP, SERVER_PORT))
    sent += 1
    t0 = time.time()
    try:
        data, _ = sock.recvfrom(4096)
        t1 = time.time()
        rtt = (t1 - ts) * 1000.0
        return True, rtt, data
    except socket.timeout:
        return False, None, None

print(f"[CLIENT] probing {SERVER_IP}:{SERVER_PORT} ...")
print(f"[CLIENT] phase1: {int(RUN_SECONDS/2)}s warmup -> idle {WARM_IDLE}s -> phase2: {int(RUN_SECONDS/2)}s")
half = int(RUN_SECONDS/2)

# phase1
for _ in range(half):
    ok, rtt, data = send_one(seq); seq += 1
    if ok:
        recv += 1
        rtts.append(rtt)
        if len(rtts) % 5 == 0:
            print(f"[CLIENT] RTT last={rtt:.1f} ms, recv={recv}/{sent}")
    else:
        print("[CLIENT] timeout")
    time.sleep(INTERVAL)

# idle：模拟 AnyDesk 会话空闲，看 NAT 是否失效
print(f"[CLIENT] idle for {WARM_IDLE}s to test NAT binding...")
time.sleep(WARM_IDLE)

# phase2
print("[CLIENT] resume sending...")
for _ in range(half):
    ok, rtt, data = send_one(seq); seq += 1
    if ok:
        recv += 1
        rtts.append(rtt)
        if len(rtts) % 5 == 0:
            print(f"[CLIENT] RTT last={rtt:.1f} ms, recv={recv}/{sent}")
    else:
        print("[CLIENT] timeout")
    time.sleep(INTERVAL)

# 汇总
loss = 0.0 if sent == 0 else (1 - recv / sent) * 100
jitter = statistics.pstdev(rtts) if rtts else 0.0
if rtts:
    print(f"\n[RESULT] sent={sent}, recv={recv}, loss={loss:.1f}%")
    print(f"[RESULT] RTT avg={statistics.mean(rtts):.1f} ms, min={min(rtts):.1f}, max={max(rtts):.1f}, jitter={jitter:.1f} ms")
else:
    print("\n[RESULT] no replies at all (UDP blocked or server not echoing)")