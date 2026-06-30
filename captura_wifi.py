import os
import sys
import time
import signal
import argparse
import platform
import subprocess
from datetime import datetime
from pathlib import Path

from scapy.all import (
    sniff,
    wrpcap,
    get_if_list,
    get_if_addr,
    conf,
    IP,
    TCP,
    UDP,
    IFACES,
)

# CONFIGURAÇÃO

OUTPUT_DIR = Path("./capturas")
OUTPUT_DIR.mkdir(exist_ok=True)

packet_buffer = []

stats = {
    "total": 0,
    "tcp": 0,
    "udp": 0,
    "outros": 0,
    "bytes_total": 0,
}

IS_WINDOWS = platform.system() == "Windows"


# =============================================================================
# DETECÇÃO DE INTERFACE — WINDOWS
# =============================================================================

def _guid_para_npf(guid: str) -> str:
    """Converte um GUID puro para o nome NPF usado pelo Npcap/WinPcap."""
    guid = guid.strip("{}")
    return f"\\Device\\NPF_{{{guid}}}"


def _mapear_interfaces_windows() -> dict[str, str]:
    """
    Executa Get-NetAdapter via PowerShell e devolve um dicionário
        nome_legível  →  nome_NPF
    ex: {"Wi-Fi": "\\Device\\NPF_{BF1BE681-...}", "Ethernet": "\\Device\\NPF_{...}"}

    Retorna {} se não conseguir executar o PowerShell.
    """
    try:
        resultado = subprocess.run(
            [
                "powershell", "-NoProfile", "-NonInteractive", "-Command",
                "Get-NetAdapter | Select-Object Name,InterfaceGuid,Status "
                "| ConvertTo-Csv -NoTypeInformation",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if resultado.returncode != 0:
            return {}

        mapa = {}
        linhas = resultado.stdout.strip().splitlines()
        if len(linhas) < 2:
            return {}

        # Primeira linha é o cabeçalho CSV
        cabecalho = [c.strip('"') for c in linhas[0].split(",")]
        idx_nome  = cabecalho.index("Name")
        idx_guid  = cabecalho.index("InterfaceGuid")

        for linha in linhas[1:]:
            partes = [c.strip('"') for c in linha.split(",")]
            if len(partes) <= max(idx_nome, idx_guid):
                continue
            nome = partes[idx_nome]
            guid = partes[idx_guid]
            if nome and guid:
                mapa[nome] = _guid_para_npf(guid)

        return mapa

    except Exception:
        return {}


def _detectar_wifi_windows() -> tuple[str | None, str | None]:
    """
    Tenta identificar a interface Wi-Fi no Windows.
    Retorna (nome_legível, nome_NPF) ou (None, None).
    """
    mapa = _mapear_interfaces_windows()
    if not mapa:
        return None, None

    # Palavras-chave comuns em adaptadores Wi-Fi no Windows
    palavras_wifi = ["wi-fi", "wifi", "wireless", "wlan", "802.11", "wi fi"]

    for nome, npf in mapa.items():
        if any(kw in nome.lower() for kw in palavras_wifi):
            return nome, npf

    return None, None


def _listar_interfaces_windows():
    """Exibe tabela legível das interfaces no Windows, cruzando NPF com nomes reais."""
    mapa = _mapear_interfaces_windows()

    print("\n  [Windows — adaptadores via Get-NetAdapter]")

    if mapa:
        print(f"  {'Nome legível':<20} {'Nome para -i (NPF)'}")
        print("  " + "-" * 72)
        for nome, npf in mapa.items():
            is_wifi = any(kw in nome.lower()
                          for kw in ["wi-fi", "wifi", "wireless", "wlan"])
            marcador = " ← Wi-Fi?" if is_wifi else ""
            print(f"  {nome:<20} {npf}{marcador}")
    else:
        print("  Não foi possível consultar Get-NetAdapter.")
        print("  Interfaces brutas do Scapy:")
        for npf in get_if_list():
            print(f"    {npf}")

    print()
    print("  DICA: passe o 'Nome legível' com -i e o script resolve o NPF automaticamente.")
    print("        Exemplo:  python captura_wifi.py -i \"Wi-Fi\" -d 150")


# =============================================================================
# FUNÇÕES AUXILIARES (originais, preservadas)
# =============================================================================

def listar_interfaces() -> list:
    print("\n" + "=" * 70)
    print("INTERFACES DE REDE DISPONÍVEIS")
    print("=" * 70)

    if IS_WINDOWS:
        _listar_interfaces_windows()
        interfaces_scapy = get_if_list()
        print("\n" + "=" * 70)
        return interfaces_scapy

    # ── Linux / macOS (comportamento original) ──────────────────────────────
    interfaces_scapy = []
    try:
        print("\n  [Scapy IFACES — fonte primária]")
        print(f"  {'Índice':<6} {'Nome':<20} {'IP':<18} {'Flags'}")
        print("  " + "-" * 62)
        for nome, iface in IFACES.items():
            try:
                ip = get_if_addr(nome) or "sem IP"
            except Exception:
                ip = "sem IP"
            flags = getattr(iface, "flags", "")
            is_wifi = any(x in nome.lower() for x in ["wl", "wifi", "wlan", "wi-fi"])
            marcador = " ← Wi-Fi?" if is_wifi else ""
            print(f"  [{len(interfaces_scapy):<4}] {nome:<20} {ip:<18} {flags}{marcador}")
            interfaces_scapy.append(nome)
    except Exception as e:
        print(f"  IFACES indisponível: {e}")

    try:
        lista_simples = get_if_list()
        extras = [x for x in lista_simples if x not in interfaces_scapy]
        if extras:
            print(f"\n  [get_if_list() — interfaces adicionais não em IFACES]")
            for nome in extras:
                print(f"    {nome}")
            interfaces_scapy.extend(extras)
    except Exception:
        pass

    try:
        resultado = subprocess.run(
            ["ip", "link", "show"],
            capture_output=True, text=True, timeout=5,
        )
        if resultado.returncode == 0:
            print("\n  [ip link show — visão do sistema operacional]")
            for linha in resultado.stdout.strip().split("\n"):
                if ": " in linha and not linha.startswith(" "):
                    partes = linha.split(": ")
                    if len(partes) >= 2:
                        nome_iface = partes[1].split("@")[0]
                        estado = "UP" if "UP" in linha else "DOWN"
                        is_wifi = any(x in nome_iface.lower()
                                      for x in ["wl", "wifi", "wlan"])
                        marcador = " ← Wi-Fi?" if is_wifi else ""
                        print(f"    {nome_iface:<20} [{estado}]{marcador}")
    except Exception:
        pass

    print("\n" + "=" * 70)
    print("  Use o nome exato da coluna 'Nome' no argumento --interface / -i")
    print("=" * 70)

    return interfaces_scapy


def detectar_interface_wifi() -> str | None:
    """
    Detecta automaticamente a interface Wi-Fi.
    No Windows: cruza Get-NetAdapter com as interfaces do Npcap.
    No Linux/macOS: usa prefixos típicos (wlp, wlan, etc.).
    """
    if IS_WINDOWS:
        _, npf = _detectar_wifi_windows()
        return npf

    # ── Linux / macOS (comportamento original) ──────────────────────────────
    candidatos = []
    prefixos_wifi = ["wlp", "wlan", "wlx", "wl"]

    try:
        for nome in IFACES.keys():
            if any(nome.lower().startswith(p) for p in prefixos_wifi):
                candidatos.append(nome)
    except Exception:
        pass

    try:
        lista = get_if_list()
        for nome in lista:
            if any(nome.lower().startswith(p) for p in prefixos_wifi):
                if nome not in candidatos:
                    candidatos.append(nome)
    except Exception:
        pass

    if not candidatos:
        return None

    for pref in ["wlp", "wlan", "wlx", "wl"]:
        for c in candidatos:
            if c.lower().startswith(pref):
                return c

    return candidatos[0]


def _resolver_interface(interface_arg: str) -> str:
    """
    No Windows: aceita tanto o nome legível ("Wi-Fi") quanto o NPF bruto.
    Se receber um nome legível, converte para o NPF correspondente.
    Em outros sistemas: devolve como está.
    """
    if not IS_WINDOWS:
        return interface_arg

    # Já é um caminho NPF?
    if interface_arg.startswith("\\Device\\") or interface_arg.startswith(r"\Device"):
        return interface_arg

    # Tenta resolver pelo nome legível via Get-NetAdapter
    mapa = _mapear_interfaces_windows()
    if interface_arg in mapa:
        npf = mapa[interface_arg]
        print(f"  Interface '{interface_arg}' resolvida para: {npf}")
        return npf

    # Busca case-insensitive
    for nome, npf in mapa.items():
        if nome.lower() == interface_arg.lower():
            print(f"  Interface '{interface_arg}' resolvida para: {npf}")
            return npf

    # Não encontrou — devolve como veio (vai falhar com mensagem clara)
    return interface_arg


# =============================================================================
# CALLBACK E SALVAMENTO (originais, preservados)
# =============================================================================

def processar_pacote(pkt):
    packet_buffer.append(pkt)
    stats["total"] += 1
    stats["bytes_total"] += len(pkt)

    if pkt.haslayer(TCP):
        stats["tcp"] += 1
    elif pkt.haslayer(UDP):
        stats["udp"] += 1
    else:
        stats["outros"] += 1

    if stats["total"] % 100 == 0:
        print(
            f"\r  Capturados: {stats['total']:>6} pacotes | "
            f"TCP: {stats['tcp']:>5} | UDP: {stats['udp']:>5} | "
            f"{stats['bytes_total'] / 1024:.1f} KB",
            end="",
            flush=True,
        )


def salvar_captura(nome_base: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = OUTPUT_DIR / f"pacotes_capturados.pcap"
    wrpcap(str(filename), packet_buffer)
    return filename


def signal_handler(sig, frame):
    print("\n\n  Captura interrompida pelo usuário (Ctrl+C).")
    if packet_buffer:
        path = salvar_captura("captura_interrompida")
        print(f"  {len(packet_buffer)} pacotes salvos em: {path}")
    sys.exit(0)


# =============================================================================
# FUNÇÃO PRINCIPAL DE CAPTURA
# =============================================================================

def capturar_trafico(
    interface: str,
    duracao_segundos: int = None,
    n_pacotes: int = None,
    nome_base: str = "wifi_capture",
    filtro_bpf: str = None,
):
    # ── Resolução de nome legível → NPF (Windows) ───────────────────────────
    interface = _resolver_interface(interface)

    # ── Validação da interface ───────────────────────────────────────────────
    interfaces_disponiveis = []
    try:
        interfaces_disponiveis = list(IFACES.keys()) + get_if_list()
        interfaces_disponiveis = list(dict.fromkeys(interfaces_disponiveis))
    except Exception:
        pass

    if interface not in interfaces_disponiveis and interfaces_disponiveis:
        print(f"\n  ERRO: interface '{interface}' não encontrada pelo Scapy.")

        if IS_WINDOWS:
            print("\n  No Windows, use --listar-interfaces para ver os nomes disponíveis.")
            print("  Você pode passar o nome legível (ex: -i \"Wi-Fi\") ou o caminho NPF.")
        else:
            print(f"  Interfaces disponíveis: {interfaces_disponiveis}")

        sugestao = detectar_interface_wifi()
        if sugestao:
            nome_exib = sugestao
            if IS_WINDOWS:
                # Mostra o nome legível junto ao NPF, se possível
                mapa = _mapear_interfaces_windows()
                mapa_inv = {v: k for k, v in mapa.items()}
                nome_exib = f"{mapa_inv.get(sugestao, sugestao)}  ({sugestao})"

            print(f"\n  Interface Wi-Fi detectada automaticamente: {nome_exib}")
            resposta = input("  Usar essa interface? [S/n]: ").strip().lower()
            if resposta in ("", "s", "sim", "y", "yes"):
                interface = sugestao
                print(f"  Interface substituída para: {interface}")
            else:
                print("  Abortando.")
                sys.exit(1)
        else:
            print("\n  Não foi possível detectar automaticamente uma interface Wi-Fi.")
            if IS_WINDOWS:
                print("  Execute:  python captura_wifi.py --listar-interfaces")
            else:
                print("  Execute:  ip link show")
            sys.exit(1)

    signal.signal(signal.SIGINT, signal_handler)

    print("=" * 70)
    print("INICIANDO CAPTURA DE TRÁFEGO")
    print("=" * 70)
    print(f"  Interface     : {interface}")
    if duracao_segundos:
        print(f"  Duração       : {duracao_segundos}s")
    else:
        print("  Duração       : indefinida (Ctrl+C para parar)")
    if n_pacotes:
        print(f"  Limite pacotes: {n_pacotes}")
    else:
        print("  Limite pacotes: indefinido")
    print(f"  Filtro BPF    : {filtro_bpf or 'nenhum (todo tráfego IP)'}")
    print(f"  Saída         : {OUTPUT_DIR}/")
    print("=" * 70)
    print("\n  Capturando... (pressione Ctrl+C para interromper)\n")

    t0 = time.time()

    try:
        sniff(
            iface=interface,
            prn=processar_pacote,
            timeout=duracao_segundos,
            count=n_pacotes if n_pacotes else 0,
            filter=filtro_bpf,
            store=False,
        )
    except PermissionError:
        print("\n\n  ERRO DE PERMISSÃO.")
        if IS_WINDOWS:
            print("  Execute o terminal (cmd / PowerShell) como Administrador.")
        else:
            print("  Execute o script com 'sudo'.")
            print("  Exemplo: sudo python captura_wifi.py -i wlp2s0 -d 15")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n  ERRO durante a captura: {e}")
        if packet_buffer:
            print(f"  Salvando os {len(packet_buffer)} pacotes coletados até agora…")
            path = salvar_captura(nome_base + "_erro")
            print(f"  Salvos em: {path}")
        sys.exit(1)

    elapsed = time.time() - t0

    print("\n\n" + "=" * 70)
    print("CAPTURA FINALIZADA")
    print("=" * 70)
    print(f"  Duração real        : {elapsed:.1f}s")
    print(f"  Total de pacotes    : {stats['total']:,}")
    print(f"  Pacotes TCP         : {stats['tcp']:,}")
    print(f"  Pacotes UDP         : {stats['udp']:,}")
    print(f"  Outros protocolos   : {stats['outros']:,}")
    print(f"  Volume total        : {stats['bytes_total'] / 1024:.2f} KB")

    if stats["total"] == 0:
        print("\n  AVISO: nenhum pacote foi capturado. Verifique se:")
        print("    - A interface informada está correta (use --listar-interfaces)")
        if IS_WINDOWS:
            print("    - O Npcap está instalado: https://npcap.com")
            print("    - O terminal está rodando como Administrador")
        else:
            print("    - O script está rodando com privilégios root")
        print("    - Há tráfego real passando pela interface durante a captura")
        return None

    output_path = salvar_captura(nome_base)
    print(f"\n  Arquivo .pcap salvo em: {output_path}")
    print(f"  Tamanho do arquivo    : {output_path.stat().st_size / 1024:.2f} KB")

    return output_path

# CLI

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Captura de tráfego Wi-Fi com Scapy (Windows e Linux).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos — Windows:
  python captura_wifi.py --listar-interfaces
  python captura_wifi.py -i "Wi-Fi" -d 150
  python captura_wifi.py -i "Wi-Fi" -d 0   (indefinido, Ctrl+C para parar)
  python captura_wifi.py -i "Wi-Fi" -d 15 --filtro "tcp port 443"

Exemplos — Linux:
  sudo python captura_wifi.py --listar-interfaces
  sudo python captura_wifi.py -i wlp2s0 -d 120 --nome navegacao_normal
  sudo python captura_wifi.py -i wlp2s0 -d 0
        """,
    )
    parser.add_argument(
        "--interface", "-i", type=str, default=None,
        help=(
            "Nome da interface. No Windows aceita nome legível (ex: \"Wi-Fi\") "
            "ou o caminho NPF bruto. No Linux use wlp2s0, wlan0, etc. "
            "Se omitido, tenta detectar automaticamente."
        ),
    )
    parser.add_argument(
        "--duracao", "-d", type=int, default=15,
        help="Duração em segundos (padrão: 15). Use 0 para indefinido.",
    )
    parser.add_argument(
        "--max-pacotes", "-n", type=int, default=None,
        help="Número máximo de pacotes (opcional).",
    )
    parser.add_argument(
        "--nome", type=str, default="wifi_capture",
        help="Prefixo do arquivo .pcap de saída (padrão: wifi_capture).",
    )
    parser.add_argument(
        "--filtro", type=str, default="ip",
        help="Filtro BPF (padrão: 'ip'). Ex: 'tcp', 'udp', 'tcp port 443'.",
    )
    parser.add_argument(
        "--listar-interfaces", action="store_true",
        help="Lista todas as interfaces disponíveis e encerra.",
    )

    args = parser.parse_args()

    if args.listar_interfaces:
        listar_interfaces()
        sys.exit(0)

    interface_escolhida = args.interface
    if interface_escolhida is None:
        print("  --interface não especificado. Tentando detectar Wi-Fi automaticamente…")
        interface_escolhida = detectar_interface_wifi()
        if interface_escolhida:
            print(f"  Interface Wi-Fi detectada: '{interface_escolhida}'")
        else:
            print("  ERRO: não foi possível detectar automaticamente uma interface Wi-Fi.")
            print("  Use --listar-interfaces para ver as opções e passe -i <nome>.")
            sys.exit(1)

    duracao = None if args.duracao == 0 else args.duracao

    capturar_trafico(
        interface=interface_escolhida,
        duracao_segundos=duracao,
        n_pacotes=args.max_pacotes,
        nome_base=args.nome,
        filtro_bpf=args.filtro,
    )