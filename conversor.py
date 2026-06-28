import sys
import csv
import math
import argparse
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np

# scapy silencioso 
import logging
logging.getLogger("scapy.runtime").setLevel(logging.ERROR)
from scapy.all import rdpcap, IP, IPv6, TCP, UDP, ICMP

# Cabeçalho exato do CICFlowMeter v3 / CICIDS-2017

CICFLOW_HEADER = [
    "Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port", "Protocol",
    "Timestamp", "Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts",
    "TotLen Fwd Pkts", "TotLen Bwd Pkts",
    "Fwd Pkt Len Max", "Fwd Pkt Len Min", "Fwd Pkt Len Mean", "Fwd Pkt Len Std",
    "Bwd Pkt Len Max", "Bwd Pkt Len Min", "Bwd Pkt Len Mean", "Bwd Pkt Len Std",
    "Flow Byts/s", "Flow Pkts/s",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Tot", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Tot", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "Fwd Header Len", "Bwd Header Len",
    "Fwd Pkts/s", "Bwd Pkts/s",
    "Pkt Len Min", "Pkt Len Max", "Pkt Len Mean", "Pkt Len Std", "Pkt Len Var",
    "FIN Flag Cnt", "SYN Flag Cnt", "RST Flag Cnt", "PSH Flag Cnt",
    "ACK Flag Cnt", "URG Flag Cnt", "CWE Flag Count", "ECE Flag Cnt",
    "Down/Up Ratio", "Pkt Size Avg",
    "Fwd Seg Size Avg", "Bwd Seg Size Avg",
    "Fwd Byts/b Avg", "Fwd Pkts/b Avg", "Fwd Blk Rate Avg",
    "Bwd Byts/b Avg", "Bwd Pkts/b Avg", "Bwd Blk Rate Avg",
    "Subflow Fwd Pkts", "Subflow Fwd Byts",
    "Subflow Bwd Pkts", "Subflow Bwd Byts",
    "Init Fwd Win Byts", "Init Bwd Win Byts",
    "Fwd Act Data Pkts", "Fwd Seg Size Min",
    "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean",  "Idle Std",  "Idle Max",  "Idle Min",
    "Label",
]

# Timeout de fluxo: 120 s (igual ao CICFlowMeter)
FLOW_TIMEOUT = 120.0

# Limiar de "ativo/ocioso" em microssegundos (1 s) — igual ao CICFlowMeter
ACTIVITY_TIMEOUT_US = 1_000_000.0   # 1 s em µs


# Helpers estatísticos

def _safe(arr, func, fallback=0.0):
    """Aplica func sobre arr; retorna fallback se arr estiver vazio."""
    if not arr:
        return fallback
    return func(arr)


def _iat(ts_list):
    """Devolve lista de inter-arrival times (µs) a partir de timestamps (s)."""
    if len(ts_list) < 2:
        return []
    return [(ts_list[i] - ts_list[i - 1]) * 1e6
            for i in range(1, len(ts_list))]


def _active_idle(ts_list, activity_timeout_us=ACTIVITY_TIMEOUT_US):
    """
    Replica o cálculo Active/Idle do CICFlowMeter.
    Retorna (active_list_µs, idle_list_µs).
    """
    if len(ts_list) < 2:
        return [], []
    active, idle = [], []
    start = ts_list[0]
    last  = ts_list[0]
    for t in ts_list[1:]:
        gap = (t - last) * 1e6        # µs
        if gap > activity_timeout_us:
            # período activo encerra; período idle começa
            active.append((last - start) * 1e6)
            idle.append(gap)
            start = t
        last = t
    # último segmento activo
    active.append((last - start) * 1e6)
    return active, idle


def _fmt(v):
    """Formata número para CSV sem notação científica e sem zeros inúteis."""
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return "0"
        # 6 casas decimais, sem zeros à direita
        return f"{v:.6f}".rstrip("0").rstrip(".")
    return str(v)


# Estrutura de fluxo 

class Flow:
    """Armazena todos os dados de um fluxo bidirecional."""

    __slots__ = (
        "src_ip", "dst_ip", "src_port", "dst_port", "protocol",
        "fwd_ts", "bwd_ts",          # timestamps por direção (s float)
        "fwd_len", "bwd_len",         # tamanho payload por pacote
        "fwd_hdr", "bwd_hdr",         # tamanho header por pacote
        "fwd_flags", "bwd_flags",     # OR de flags TCP por pacote
        "first_ts",                   # timestamp absoluto do 1º pacote
        "init_fwd_win", "init_bwd_win",
        "fwd_act_data",               # pkts fwd com payload > 0
    )

    def __init__(self, src_ip, dst_ip, src_port, dst_port, protocol, ts):
        self.src_ip   = src_ip
        self.dst_ip   = dst_ip
        self.src_port = src_port
        self.dst_port = dst_port
        self.protocol = protocol
        self.first_ts = ts

        self.fwd_ts   = [ts]
        self.bwd_ts   = []
        self.fwd_len  = []
        self.bwd_len  = []
        self.fwd_hdr  = []
        self.bwd_hdr  = []
        self.fwd_flags = []
        self.bwd_flags = []
        self.init_fwd_win = -1
        self.init_bwd_win = -1
        self.fwd_act_data = 0

    @property
    def last_ts(self):
        all_ts = self.fwd_ts + self.bwd_ts
        return max(all_ts) if all_ts else self.first_ts

    def add_packet(self, is_fwd: bool, ts: float,
                   payload_len: int, hdr_len: int,
                   tcp_flags: int = 0, win: int = 0):
        if is_fwd:
            self.fwd_ts.append(ts)
            self.fwd_len.append(payload_len)
            self.fwd_hdr.append(hdr_len)
            self.fwd_flags.append(tcp_flags)
            if payload_len > 0:
                self.fwd_act_data += 1
            if self.init_fwd_win == -1:
                self.init_fwd_win = win
        else:
            self.bwd_ts.append(ts)
            self.bwd_len.append(payload_len)
            self.bwd_hdr.append(hdr_len)
            self.bwd_flags.append(tcp_flags)
            if self.init_bwd_win == -1:
                self.init_bwd_win = win

    # ─────────── serialização para linha CSV ────────────────────────────────
    def to_row(self, label: str) -> list:
        fwd_ts = sorted(self.fwd_ts)
        bwd_ts = sorted(self.bwd_ts)
        all_ts = sorted(fwd_ts + bwd_ts)

        # Duração em µs (CICFlowMeter usa µs internamente, exporta µs)
        duration_us = (all_ts[-1] - all_ts[0]) * 1e6 if len(all_ts) > 1 else 0.0

        # Tamanhos de payload
        fwd_lens = self.fwd_len
        bwd_lens = self.bwd_len
        all_lens = fwd_lens + bwd_lens

        tot_fwd_bytes = sum(fwd_lens)
        tot_bwd_bytes = sum(bwd_lens)
        tot_fwd_pkts  = len(fwd_ts)
        tot_bwd_pkts  = len(bwd_ts)

        # Flow rate
        dur_s = duration_us / 1e6
        flow_bytes_s = (tot_fwd_bytes + tot_bwd_bytes) / dur_s if dur_s > 0 else 0.0
        flow_pkts_s  = (tot_fwd_pkts  + tot_bwd_pkts ) / dur_s if dur_s > 0 else 0.0
        fwd_pkts_s   = tot_fwd_pkts  / dur_s if dur_s > 0 else 0.0
        bwd_pkts_s   = tot_bwd_pkts  / dur_s if dur_s > 0 else 0.0

        # IATs
        all_iat = _iat(all_ts)
        fwd_iat = _iat(fwd_ts)
        bwd_iat = _iat(bwd_ts)

        def stats(arr):
            if not arr:
                return 0.0, 0.0, 0.0, 0.0
            a = np.array(arr, dtype=float)
            return float(a.mean()), float(a.std(ddof=0)), float(a.max()), float(a.min())

        ai_mean, ai_std, ai_max, ai_min = stats(all_iat)
        fi_mean, fi_std, fi_max, fi_min = stats(fwd_iat)
        bi_mean, bi_std, bi_max, bi_min = stats(bwd_iat)

        # Pkt length stats (payload)
        al_mean, al_std, al_max, al_min = stats(all_lens)
        fl_mean, fl_std, fl_max, fl_min = stats(fwd_lens)
        bl_mean, bl_std, bl_max, bl_min = stats(bwd_lens)
        al_var = float(np.array(all_lens, float).var(ddof=0)) if all_lens else 0.0

        # Packet size avg (CIC usa total bytes / total pkts)
        tot_pkts  = tot_fwd_pkts + tot_bwd_pkts
        pkt_size_avg = (tot_fwd_bytes + tot_bwd_bytes) / tot_pkts if tot_pkts else 0.0

        # TCP flags
        def flag_sum(flags_list, bit):
            return sum(1 for f in flags_list if f & bit)

        fwd_flags = self.fwd_flags
        bwd_flags = self.bwd_flags
        all_flags = fwd_flags + bwd_flags

        fin = flag_sum(all_flags, 0x01)
        syn = flag_sum(all_flags, 0x02)
        rst = flag_sum(all_flags, 0x04)
        psh = flag_sum(all_flags, 0x08)
        ack = flag_sum(all_flags, 0x10)
        urg = flag_sum(all_flags, 0x20)
        cwe = flag_sum(all_flags, 0x80)   # CWR
        ece = flag_sum(all_flags, 0x40)

        fwd_psh = flag_sum(fwd_flags, 0x08)
        bwd_psh = flag_sum(bwd_flags, 0x08)
        fwd_urg = flag_sum(fwd_flags, 0x20)
        bwd_urg = flag_sum(bwd_flags, 0x20)

        # Headers
        fwd_hdr_tot = sum(self.fwd_hdr)
        bwd_hdr_tot = sum(self.bwd_hdr)

        # Down/Up ratio
        down_up = tot_bwd_pkts / tot_fwd_pkts if tot_fwd_pkts else 0.0

        # Seg size avg (payload avg)
        fwd_seg_avg = float(np.mean(fwd_lens)) if fwd_lens else 0.0
        bwd_seg_avg = float(np.mean(bwd_lens)) if bwd_lens else 0.0

        # Subflow (CIC usa 1 subflow por fluxo nesta implementação)
        sf_fwd_pkts  = tot_fwd_pkts
        sf_fwd_bytes = tot_fwd_bytes
        sf_bwd_pkts  = tot_bwd_pkts
        sf_bwd_bytes = tot_bwd_bytes

        # Active / Idle
        active_list, idle_list = _active_idle(all_ts)
        ac_mean, ac_std, ac_max, ac_min = stats(active_list)
        id_mean, id_std, id_max, id_min = stats(idle_list)

        # Fwd seg size min (menor payload fwd não nulo, ou 0)
        fwd_seg_min = min((l for l in fwd_lens if l > 0), default=0)

        # Timestamp em formato "dd/MM/YYYY HH:MM:SS" (igual ao CIC)
        dt = datetime.fromtimestamp(self.first_ts, tz=timezone.utc)
        ts_str = dt.strftime("%d/%m/%Y %H:%M:%S")

        # Flow ID
        flow_id = (f"{self.src_ip}-{self.dst_ip}-"
                   f"{self.src_port}-{self.dst_port}-{self.protocol}")

        row = [
            flow_id,
            self.src_ip, self.src_port,
            self.dst_ip, self.dst_port,
            self.protocol,
            ts_str,
            _fmt(duration_us),
            tot_fwd_pkts, tot_bwd_pkts,
            tot_fwd_bytes, tot_bwd_bytes,
            _fmt(fl_max), _fmt(fl_min), _fmt(fl_mean), _fmt(fl_std),
            _fmt(bl_max), _fmt(bl_min), _fmt(bl_mean), _fmt(bl_std),
            _fmt(flow_bytes_s), _fmt(flow_pkts_s),
            _fmt(ai_mean), _fmt(ai_std), _fmt(ai_max), _fmt(ai_min),
            _fmt(sum(fwd_iat)), _fmt(fi_mean), _fmt(fi_std), _fmt(fi_max), _fmt(fi_min),
            _fmt(sum(bwd_iat)), _fmt(bi_mean), _fmt(bi_std), _fmt(bi_max), _fmt(bi_min),
            fwd_psh, bwd_psh, fwd_urg, bwd_urg,
            fwd_hdr_tot, bwd_hdr_tot,
            _fmt(fwd_pkts_s), _fmt(bwd_pkts_s),
            _fmt(al_min), _fmt(al_max), _fmt(al_mean), _fmt(al_std), _fmt(al_var),
            fin, syn, rst, psh, ack, urg, cwe, ece,
            _fmt(down_up),
            _fmt(pkt_size_avg),
            _fmt(fwd_seg_avg), _fmt(bwd_seg_avg),
            0, 0, 0,   # Fwd Byts/b Avg, Fwd Pkts/b Avg, Fwd Blk Rate Avg
            0, 0, 0,   # Bwd Byts/b Avg, Bwd Pkts/b Avg, Bwd Blk Rate Avg
            sf_fwd_pkts, sf_fwd_bytes, sf_bwd_pkts, sf_bwd_bytes,
            self.init_fwd_win, self.init_bwd_win,
            self.fwd_act_data,
            fwd_seg_min,
            _fmt(ac_mean), _fmt(ac_std), _fmt(ac_max), _fmt(ac_min),
            _fmt(id_mean), _fmt(id_std), _fmt(id_max), _fmt(id_min),
            label,
        ]
        return row


# Engine de fluxos 

class FlowTable:
    """Gerencia fluxos ativos e expirados."""

    def __init__(self, timeout=FLOW_TIMEOUT):
        self.flows: dict[tuple, Flow] = {}
        self.finished: list[Flow] = []
        self.timeout = timeout

    # chave bidirecional canônica
    @staticmethod
    def _key(src_ip, dst_ip, src_port, dst_port, proto):
        if (src_ip, src_port) < (dst_ip, dst_port):
            return (src_ip, dst_ip, src_port, dst_port, proto)
        return (dst_ip, src_ip, dst_port, src_port, proto)

    def add_packet(self, ts, src_ip, dst_ip, src_port, dst_port,
                   proto, payload_len, hdr_len, tcp_flags=0, win=0):
        key = self._key(src_ip, dst_ip, src_port, dst_port, proto)
        flow = self.flows.get(key)

        # Expirar fluxo antigo se timeout
        if flow and (ts - flow.last_ts) > self.timeout:
            self.finished.append(flow)
            del self.flows[key]
            flow = None

        if flow is None:
            flow = Flow(src_ip, dst_ip, src_port, dst_port, proto, ts)
            self.flows[key] = flow
            # 1º pacote já foi adicionado pelo __init__; precisamos registrar
            # comprimento e header
            flow.fwd_len.append(payload_len)
            flow.fwd_hdr.append(hdr_len)
            flow.fwd_flags.append(tcp_flags)
            if payload_len > 0:
                flow.fwd_act_data += 1
            flow.init_fwd_win = win
            return

        # Determinar direção
        is_fwd = (src_ip == flow.src_ip and
                  src_port == flow.src_port)
        flow.add_packet(is_fwd, ts, payload_len, hdr_len, tcp_flags, win)

        # FIN ou RST: encerrar fluxo TCP
        if tcp_flags & 0x01 or tcp_flags & 0x04:
            self.finished.append(flow)
            del self.flows[key]

    def flush_all(self):
        self.finished.extend(self.flows.values())
        self.flows.clear()


# Processamento do pcap 

def _proto_num(pkt):
    if TCP in pkt:  return 6
    if UDP in pkt:  return 17
    if ICMP in pkt: return 1
    if IP in pkt:   return pkt[IP].proto
    return 0


def _ports(pkt):
    if TCP in pkt:  return pkt[TCP].sport, pkt[TCP].dport
    if UDP in pkt:  return pkt[UDP].sport, pkt[UDP].dport
    return 0, 0


def _payload_len(pkt):
    """Tamanho do payload de transporte (sem cabeçalho IP nem L2)."""
    if TCP in pkt:
        return max(0, len(pkt[TCP].payload))
    if UDP in pkt:
        return max(0, len(pkt[UDP].payload))
    return 0


def _header_len(pkt):
    """Tamanho combinado dos cabeçalhos IP + transporte."""
    ip_hdr = pkt[IP].ihl * 4 if IP in pkt else 0
    if TCP in pkt:
        return ip_hdr + pkt[TCP].dataofs * 4
    if UDP in pkt:
        return ip_hdr + 8
    return ip_hdr


def pcap_to_cicflow(pcap_path: str, csv_path: str, label: str = "BENIGN"):
    print(f"[*] Lendo {pcap_path} …")
    packets = rdpcap(pcap_path)
    print(f"[*] {len(packets)} pacotes carregados.")

    table = FlowTable()
    skipped = 0

    for pkt in packets:
        if IP not in pkt:
            skipped += 1
            continue

        ts       = float(pkt.time)
        src_ip   = pkt[IP].src
        dst_ip   = pkt[IP].dst
        proto    = _proto_num(pkt)
        sp, dp   = _ports(pkt)
        pay_len  = _payload_len(pkt)
        hdr_len  = _header_len(pkt)
        flags    = int(pkt[TCP].flags) if TCP in pkt else 0
        win      = pkt[TCP].window    if TCP in pkt else 0

        table.add_packet(ts, src_ip, dst_ip, sp, dp,
                         proto, pay_len, hdr_len, flags, win)

    table.flush_all()
    flows = table.finished
    print(f"[*] {len(flows)} fluxos gerados  |  {skipped} pacotes não-IP ignorados.")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CICFLOW_HEADER)
        for flow in flows:
            w.writerow(flow.to_row(label))

    print(f"[✓] CSV salvo em: {csv_path}")


# Entrada principal 

def main():
    parser = argparse.ArgumentParser(
        description="Converte .pcap para formato CICFlowMeter CSV"
    )
    parser.add_argument("pcap",   help="Arquivo de entrada (.pcap/.pcapng)")
    parser.add_argument("output", nargs="?", help="Arquivo de saída (.csv)")
    parser.add_argument("--label", default="BENIGN",
                        help="Rótulo de classe para a coluna Label (padrão: BENIGN)")
    args = parser.parse_args()

    out = args.output or args.pcap.rsplit(".", 1)[0] + "_cicflow.csv"
    pcap_to_cicflow(args.pcap, out, args.label)


if __name__ == "__main__":
    main()