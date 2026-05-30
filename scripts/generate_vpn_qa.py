"""
Генератор VPN/proxy-датасета для обучения Mi-AI ассистента по сетям.

Запуск: py -3.11 scripts/generate_vpn_qa.py
Выход: data/texts/mi_vpn_training.txt — 4000-6000 Q&A пар

Покрытие:
- VPN: OpenVPN, WireGuard, IKEv2/IPsec, L2TP, PPTP, SSTP
- Прокси: Shadowsocks, SSR, VMess, VLESS, Trojan, Hysteria, TUIC, Naive
- Reality (XTLS-Reality, Vision flow)
- SOCKS4/4a/5, HTTP-proxy
- Tools: xray, v2ray, sing-box, Hysteria, WireGuard, OpenVPN, Hiddify, Marzban, 3X-UI
- Концепции: TLS, SNI, ALPN, ECH, DoH/DoT, DPI, обход блокировок
- Конфиги: JSON для xray/v2ray, WG .conf, .ovpn, ссылки SS/VMess/VLESS/Trojan
- «Я не знаю» паттерны для защиты от галлюцинаций
"""

from __future__ import annotations
from pathlib import Path
import random

random.seed(42)


# =============================================================================
# 1. VPN-ПРОТОКОЛЫ
# =============================================================================

VPN_QA = [
    # OpenVPN
    ("Что такое OpenVPN?",
     "Открытый протокол VPN на основе SSL/TLS. Использует UDP (по умолчанию) или TCP. Очень стабильный, гибкий, кроссплатформенный. Шифрование AES-256."),
    ("Какие порты использует OpenVPN?",
     "По умолчанию UDP 1194. Можно настроить на любой порт, часто маскируют под TCP 443 (HTTPS) для обхода DPI."),
    ("Чем UDP лучше TCP для OpenVPN?",
     "UDP быстрее и не имеет двойной потери пакетов (TCP-in-TCP). TCP стабильнее в сетях с ограничениями, но медленнее."),
    ("Как сгенерировать сертификаты OpenVPN?",
     "Через easy-rsa: init-pki, build-ca, build-server-full, build-client-full. Или одной командой openvpn --genkey secret для статического ключа."),
    ("Что такое .ovpn файл?",
     "Клиентский конфиг OpenVPN. Содержит адрес сервера, порт, протокол, сертификаты, ключи, параметры шифрования. Импортируется в клиент OpenVPN Connect."),

    # WireGuard
    ("Что такое WireGuard?",
     "Современный VPN-протокол. Быстрее OpenVPN, проще в настройке, меньше кода (~4000 строк vs 100K у OpenVPN). Работает в ядре Linux, использует ChaCha20."),
    ("Чем WireGuard лучше OpenVPN?",
     "Скорость в 2-5 раз выше, меньше overhead, проще конфиг (.conf), быстрее handshake (1 RTT), современная криптография (Curve25519, ChaCha20-Poly1305)."),
    ("Какие порты использует WireGuard?",
     "По умолчанию UDP 51820. Можно настроить любой UDP-порт. TCP не поддерживает (через сторонние обёртки только)."),
    ("Что такое WireGuard handshake?",
     "Noise IK handshake — обмен ключами Curve25519. Происходит каждые 2 минуты. Очень быстро (1 RTT)."),
    ("Как настроить WireGuard сервер?",
     "1) Сгенерировать ключи: wg genkey | tee privatekey | wg pubkey > publickey. 2) Создать /etc/wireguard/wg0.conf с [Interface] и [Peer]. 3) wg-quick up wg0."),
    ("Что в WireGuard конфиге?",
     "[Interface]: PrivateKey, Address, ListenPort. [Peer]: PublicKey, AllowedIPs, Endpoint. Минимальный конфиг — 6 строк."),

    # IKEv2/IPsec
    ("Что такое IKEv2?",
     "Internet Key Exchange v2 — протокол создания VPN-туннелей IPsec. Быстрая реконнект при смене сети (актуально для мобильных). Поддерживается нативно в Windows/macOS/iOS."),
    ("Чем IKEv2 отличается от L2TP?",
     "IKEv2 быстрее, безопаснее, лучше переподключается при смене сети. L2TP/IPsec — старая комбинация двух протоколов, медленнее."),
    ("Какие порты использует IKEv2?",
     "UDP 500 (IKE), UDP 4500 (NAT-T для прохождения через NAT)."),

    # PPTP / SSTP
    ("Что такое PPTP?",
     "Point-to-Point Tunneling Protocol. Самый старый VPN-протокол, использует TCP 1723 + GRE. СЛАБОЕ шифрование MS-CHAPv2, ВЗЛОМАН. НЕ используй для серьёзной защиты."),
    ("Что такое SSTP?",
     "Secure Socket Tunneling Protocol от Microsoft. Использует TCP 443 (HTTPS), хорошо проходит через файрволы. Кроссплатформенность плохая (в основном Windows)."),

    # Общее про VPN
    ("Чем VPN отличается от proxy?",
     "VPN шифрует ВЕСЬ трафик системы на уровне сети (L3). Proxy работает на уровне приложения (HTTP/SOCKS) и шифрует только трафик приложений настроенных на proxy."),
    ("VPN скрывает мой IP?",
     "Да. Сайты видят IP VPN-сервера, не твой. Но VPN-провайдер видит твой реальный IP и весь трафик."),
    ("Какой VPN самый быстрый?",
     "WireGuard — лидер по скорости среди VPN-протоколов. Hysteria быстрее для прокси-задач. OpenVPN заметно медленнее."),
    ("Зачем шифровать VPN если уже HTTPS?",
     "HTTPS защищает содержимое отдельных запросов. VPN скрывает САМИ домены (SNI), DNS-запросы, метаданные, факт использования определённых сайтов."),
    ("Что такое kill switch?",
     "Функция VPN-клиента которая блокирует весь интернет если VPN-туннель упал. Предотвращает утечку реального IP при разрыве."),
    ("Что такое DNS leak?",
     "Когда VPN активен, но DNS-запросы идут через провайдера а не через VPN. Провайдер видит какие сайты ты посещаешь несмотря на VPN."),
]


# =============================================================================
# 2. PROXY-ПРОТОКОЛЫ
# =============================================================================

PROXY_QA = [
    # SOCKS
    ("Что такое SOCKS?",
     "Прокси-протокол. SOCKS4 — только TCP. SOCKS4a + DNS-разрешение на сервере. SOCKS5 — TCP + UDP, аутентификация, IPv6. SOCKS5 — современный стандарт."),
    ("Чем SOCKS4 отличается от SOCKS5?",
     "SOCKS5 поддерживает UDP, аутентификацию (логин/пароль), IPv6, и разрешение DNS на сервере. SOCKS4 — только TCP без всего этого."),
    ("Чем SOCKS5 отличается от HTTP-proxy?",
     "SOCKS5 — низкоуровневый, проксирует ЛЮБОЙ TCP/UDP трафик. HTTP-proxy — только HTTP(S). SOCKS гибче, но HTTP может кэшировать и понимает протокол."),
    ("Какой порт у SOCKS5 по умолчанию?",
     "1080. Но можно настроить на любой. Часто используют 7890, 1081 для второго инстанса."),
    ("SOCKS5 шифрует трафик?",
     "Нет. SOCKS5 — это просто проксирование, без шифрования. Если нужно шифрование — оберни в SSH-туннель или используй Shadowsocks/VLESS."),

    # Shadowsocks
    ("Что такое Shadowsocks?",
     "Secure proxy protocol изначально для обхода Великого файрвола Китая. Использует SOCKS5-подобный интерфейс + симметричное шифрование. Простой и быстрый."),
    ("Чем Shadowsocks отличается от VPN?",
     "Shadowsocks — proxy, шифрует только нужный трафик. VPN шифрует всё. Shadowsocks сложнее обнаружить (трафик выглядит как обычный random TCP)."),
    ("Какие алгоритмы шифрования в Shadowsocks?",
     "Современные: chacha20-ietf-poly1305, aes-256-gcm, aes-128-gcm. Старые (НЕ использовать): rc4-md5, salsa20."),
    ("Что такое плагин Shadowsocks?",
     "Дополнительный обёрточный слой: v2ray-plugin (WebSocket+TLS), simple-obfs (HTTP-маскировка), kcptun (KCP). Помогают при глубокой DPI."),
    ("Какая структура Shadowsocks URL?",
     "ss://method:password@server:port/?plugin=...#remark. Пример: ss://chacha20-ietf-poly1305:mypass@example.com:8388#my-server"),
    ("Что такое SSR?",
     "ShadowsocksR — форк Shadowsocks с обфускацией трафика (protocol + obfs параметры). Менее активно развивается, многие переходят на VLESS/Reality."),

    # VMess
    ("Что такое VMess?",
     "Протокол прокси из V2Ray. UUID-аутентификация, AES/ChaCha20 шифрование. Был стандартом до VLESS. Сейчас уязвим для активного DPI."),
    ("Какая структура VMess URL?",
     "vmess://base64({\"v\":\"2\",\"ps\":\"name\",\"add\":\"server\",\"port\":\"443\",\"id\":\"uuid\",\"aid\":\"0\",\"net\":\"ws\",\"type\":\"none\",\"host\":\"\",\"path\":\"/\",\"tls\":\"tls\"})"),
    ("Чем VMess плох?",
     "Активный DPI Китая научился определять VMess по паттернам. AlterID в новых конфигах = 0. Лучше использовать VLESS+Reality для современной анти-цензуры."),

    # VLESS
    ("Что такое VLESS?",
     "Облегчённый протокол V2Ray/Xray. Без шифрования внутри (полагается на внешний TLS/Reality). Быстрее VMess, меньше overhead, лучше для обхода DPI."),
    ("Чем VLESS лучше VMess?",
     "VLESS не шифрует данные сам (полагается на TLS/Reality), что даёт +20% скорости. Лучше работает с XTLS Vision. Меньше fingerprint для DPI."),
    ("Что такое VLESS+Reality?",
     "Комбинация: VLESS как протокол + Reality как TLS-обёртка. Reality маскирует трафик под TLS-handshake с настоящим сайтом (например google.com). Невозможно отличить от обычного HTTPS."),
    ("Какая структура VLESS URL?",
     "vless://uuid@server:port?type=tcp&security=reality&pbk=publickey&fp=chrome&sni=google.com&sid=shortid#name"),
    ("Что такое XTLS-Vision?",
     "Flow для VLESS оптимизирующий передачу TLS-трафика. Обходит проблему 'TLS в TLS' — определяет inner TLS handshake и не шифрует его повторно. Быстрее на 50-100%."),
    ("Что такое Reality protocol?",
     "Технология маскировки в Xray. Сервер ВЫГЛЯДИТ как настоящий сайт (например www.microsoft.com) для всех кроме клиента с правильным public key. Идеально обходит активный DPI."),
    ("Что такое sid в Reality?",
     "Short ID — короткий идентификатор клиента. Сервер хранит список разрешённых sid. Можно создать пустой '' или конкретный hex (8-16 символов)."),
    ("Что такое pbk в Reality?",
     "Public key Reality-сервера. Клиент использует его для аутентификации. Генерируется командой: xray x25519. Сервер хранит соответствующий private key."),
    ("Что такое fp в Reality?",
     "TLS fingerprint — какой браузер имитировать. Варианты: chrome, firefox, safari, edge. Помогает выглядеть как настоящий браузерный трафик."),

    # Trojan
    ("Что такое Trojan?",
     "Прокси-протокол маскирующийся под HTTPS-трафик. Полагается на TLS. Если пароль неверный — сервер ведёт себя как обычный веб-сайт (fallback). Хорош для обхода DPI."),
    ("Чем Trojan отличается от VLESS?",
     "Trojan использует пароль (string), VLESS — UUID. Trojan имеет fallback на обычный HTTPS при неудаче. VLESS+Reality сейчас сложнее обнаружить."),
    ("Какая структура Trojan URL?",
     "trojan://password@server:port?sni=domain.com&type=tcp#name"),

    # Hysteria
    ("Что такое Hysteria?",
     "Прокси на основе QUIC поверх UDP. Очень быстрый, особенно в плохих сетях с потерями. Использует Brutal/BBR для congestion control. Преодолевает throttling."),
    ("Чем Hysteria2 отличается от Hysteria1?",
     "Hysteria2 — переработанная версия. Проще конфиг, лучше совместимость, новый password-based аутентификация. Hysteria1 deprecated."),
    ("Какие порты использует Hysteria?",
     "UDP, любой настраиваемый. Часто 443 (HTTPS), 8443. По умолчанию 36712 в конфигах."),

    # TUIC
    ("Что такое TUIC?",
     "TLS over UDP, QUIC-based прокси. Молодой протокол, быстрее Hysteria в некоторых сценариях. TUIC v5 — актуальная версия."),

    # Naive
    ("Что такое Naive proxy?",
     "Прокси на основе HTTP/2 от Klzgrad. Полностью мимикрирует под Chrome (использует Chromium-стек). Очень сложно обнаружить."),
]


# =============================================================================
# 3. CONFIGS — реальные примеры конфигов (структура)
# =============================================================================

CONFIG_QA = [
    ("Покажи конфиг WireGuard клиента.",
     "[Interface]\nPrivateKey = ваш_приватный_ключ\nAddress = 10.0.0.2/24\nDNS = 1.1.1.1\n\n[Peer]\nPublicKey = публичный_ключ_сервера\nEndpoint = server.com:51820\nAllowedIPs = 0.0.0.0/0\nPersistentKeepalive = 25"),

    ("Покажи конфиг WireGuard сервера.",
     "[Interface]\nPrivateKey = серверный_приватный_ключ\nAddress = 10.0.0.1/24\nListenPort = 51820\nPostUp = iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE\n\n[Peer]\nPublicKey = публичный_ключ_клиента\nAllowedIPs = 10.0.0.2/32"),

    ("Покажи JSON конфиг xray для VLESS+Reality.",
     '{\n  "inbounds": [{\n    "port": 443,\n    "protocol": "vless",\n    "settings": {\n      "clients": [{"id": "uuid-here", "flow": "xtls-rprx-vision"}],\n      "decryption": "none"\n    },\n    "streamSettings": {\n      "network": "tcp",\n      "security": "reality",\n      "realitySettings": {\n        "dest": "www.microsoft.com:443",\n        "serverNames": ["www.microsoft.com"],\n        "privateKey": "private-key",\n        "shortIds": [""]\n      }\n    }\n  }]\n}'),

    ("Покажи JSON клиент xray для VLESS+Reality.",
     '{\n  "outbounds": [{\n    "protocol": "vless",\n    "settings": {\n      "vnext": [{\n        "address": "server.com",\n        "port": 443,\n        "users": [{"id": "uuid", "flow": "xtls-rprx-vision", "encryption": "none"}]\n      }]\n    },\n    "streamSettings": {\n      "network": "tcp",\n      "security": "reality",\n      "realitySettings": {\n        "publicKey": "pub-key",\n        "fingerprint": "chrome",\n        "serverName": "www.microsoft.com",\n        "shortId": ""\n      }\n    }\n  }]\n}'),

    ("Покажи Shadowsocks JSON конфиг.",
     '{\n  "server": "0.0.0.0",\n  "server_port": 8388,\n  "password": "yourpassword",\n  "method": "chacha20-ietf-poly1305",\n  "mode": "tcp_and_udp"\n}'),

    ("Покажи минимальный .ovpn конфиг.",
     "client\ndev tun\nproto udp\nremote server.com 1194\nresolv-retry infinite\nnobind\npersist-key\npersist-tun\nremote-cert-tls server\nauth SHA256\ncipher AES-256-GCM\nverb 3\n<ca>\n...CA сертификат...\n</ca>\n<cert>\n...клиентский сертификат...\n</cert>\n<key>\n...приватный ключ...\n</key>"),

    ("Покажи Trojan клиент конфиг JSON.",
     '{\n  "run_type": "client",\n  "local_addr": "127.0.0.1",\n  "local_port": 1080,\n  "remote_addr": "server.com",\n  "remote_port": 443,\n  "password": ["yourpassword"],\n  "ssl": {"sni": "server.com", "verify": true}\n}'),

    ("Как сгенерировать UUID для VLESS?",
     "В Xray: `xray uuid`. Через Python: `python -c \"import uuid; print(uuid.uuid4())\"`. Через openssl: `openssl rand -hex 16` (нужно отформатировать). Любой UUIDv4 подойдёт."),

    ("Как сгенерировать Reality ключи?",
     "Команда `xray x25519` выведет PrivateKey (для сервера) и Password/PublicKey (для клиента). Сохрани оба."),

    ("Как сгенерировать ключи WireGuard?",
     "Приватный: `wg genkey > privatekey`. Публичный: `cat privatekey | wg pubkey > publickey`. Pre-shared (опц.): `wg genpsk > psk`."),
]


# =============================================================================
# 4. TOOLS И КЛИЕНТЫ
# =============================================================================

TOOLS_QA = [
    # Server-side
    ("Что такое xray-core?",
     "Форк V2Ray-core с XTLS, Reality, дополнительными фичами. Самый популярный server для VLESS+Reality. Активно развивается."),
    ("Что такое v2ray-core?",
     "Оригинальный V2Ray. Поддерживает VMess, VLESS, Trojan, Shadowsocks. Xray — его форк с большим функционалом."),
    ("Что такое sing-box?",
     "Универсальный proxy-роутер. Поддерживает все основные протоколы: VLESS, Trojan, Shadowsocks, Hysteria, TUIC. Альтернатива xray + Clash."),
    ("Что такое 3X-UI?",
     "Веб-панель для управления Xray. Удобный GUI для создания пользователей, генерации ссылок, мониторинга трафика. Форк X-UI с фиксами."),
    ("Что такое Marzban?",
     "Современная веб-панель для Xray на FastAPI + Vue. Multi-user, мониторинг, API. Поддерживает Reality, все современные протоколы."),
    ("Что такое Hiddify?",
     "Open-source панель + клиент-приложение. Простая установка VLESS+Reality сервера одной командой. Один из самых user-friendly вариантов."),
    ("Чем отличается Marzban от 3X-UI?",
     "Marzban — более новый, FastAPI + Vue, REST API. 3X-UI — на Go, проще в установке, чуть менее фичастый. Оба покрывают основные задачи."),

    # Client-side
    ("Какой клиент VPN использовать на iPhone?",
     "Streisand (бесплатный, open-source), Shadowrocket (платный, мощный), FoXray. Все поддерживают VLESS+Reality."),
    ("Какой клиент использовать на Android?",
     "v2rayNG (классика), NekoBox для Android, Hiddify, sing-box. Все open-source, поддерживают современные протоколы."),
    ("Какой клиент использовать на Windows?",
     "v2rayN — самый популярный для Windows. Альтернативы: NekoBox для PC, Hiddify Desktop, Clash для Windows."),
    ("Какой клиент использовать на macOS?",
     "ClashX Pro, V2rayU, Hiddify, FoXray. Многие используют сторонние через xray + GUI обёртку."),
    ("Какой клиент использовать на Linux?",
     "Чистый xray/sing-box через systemd. GUI варианты: Nekoray, Hiddify Desktop, V2RayW."),

    # WireGuard tools
    ("Какой клиент WireGuard?",
     "Официальный WireGuard приложение (доступен на всех ОС). На Android — WireGuard from WireGuard Development Team. На iOS — WireGuard."),
    ("Что такое wg-quick?",
     "Скрипт для быстрого подъёма WireGuard-туннеля по конфигу. wg-quick up wg0 — поднять, wg-quick down wg0 — опустить."),
]


# =============================================================================
# 5. КОНЦЕПЦИИ И ТЕРМИНЫ
# =============================================================================

CONCEPTS_QA = [
    ("Что такое TLS?",
     "Transport Layer Security — криптографический протокол. Шифрует данные между клиентом и сервером. HTTPS = HTTP over TLS. Современная версия — TLS 1.3."),
    ("Что такое SNI?",
     "Server Name Indication — расширение TLS. Клиент в handshake указывает к какому домену подключается. DPI использует SNI для блокировок."),
    ("Что такое ALPN?",
     "Application-Layer Protocol Negotiation — расширение TLS. Договоренность о протоколе приложения (http/1.1, h2, h3). Используется в xray для маскировки."),
    ("Что такое ECH?",
     "Encrypted Client Hello — расширение TLS которое шифрует SNI. Защищает от DPI по SNI. Поддерживается Cloudflare, Firefox, Chrome."),
    ("Что такое DPI?",
     "Deep Packet Inspection — глубокий анализ пакетов провайдерами/государствами. Может определять тип трафика (VPN, BitTorrent) по паттернам и блокировать."),
    ("Что такое DoH?",
     "DNS over HTTPS — шифрование DNS-запросов через HTTPS. Защищает от DNS-leak. Используют Cloudflare (1.1.1.1), Google (8.8.8.8)."),
    ("Что такое DoT?",
     "DNS over TLS — шифрование DNS через TLS на отдельном порту (853). Альтернатива DoH."),
    ("Что такое domain fronting?",
     "Маскировка реального домена. SNI указывает один домен (например google.com), HTTP Host header — другой (твой proxy). Cloudflare и AWS заблокировали это."),
    ("Что такое CDN?",
     "Content Delivery Network. Распределённая сеть серверов для быстрой доставки контента. Часто используется как VPN-bypass — CDN-домены трудно блокировать."),
    ("Что такое NAT?",
     "Network Address Translation — преобразование адресов. Один публичный IP на много локальных. Препятствует прямым соединениям, поэтому WG нужен NAT-T."),
    ("Что такое NAT-T?",
     "NAT Traversal — техника прохождения IPsec/IKE через NAT. UDP 4500. Без неё VPN-сервер за NAT не работает."),
    ("Что такое obfuscation?",
     "Обфускация трафика — маскировка VPN-трафика под обычный HTTPS/QUIC. Чтобы DPI не мог распознать. Примеры: v2ray-plugin, obfs4, Reality, Naive."),
    ("Что такое split tunneling?",
     "Раздельный туннель. Часть трафика идёт через VPN, часть напрямую. Полезно когда нужен доступ к локальным ресурсам ИЛИ только определённые сайты через VPN."),
    ("Что такое full tunnel?",
     "Весь трафик идёт через VPN (AllowedIPs = 0.0.0.0/0 в WG). Безопаснее, но локальные ресурсы недоступны без split."),
    ("Что такое перехват MITM?",
     "Man-in-the-Middle — атака с подменой сервера. Защита: TLS-сертификаты + правильная проверка. Атакующий не может имитировать сервер без приватного ключа."),
    ("Что такое certificate pinning?",
     "Закрепление сертификата. Клиент знает заранее какой сертификат должен быть у сервера, отвергает другие. Защита от подделок CA."),
    ("Что такое QUIC?",
     "Транспортный протокол на UDP. Используется в HTTP/3, Hysteria, TUIC. Быстрее TCP в плохих сетях, multiplexing, без блокировки head-of-line."),
    ("Что такое WebSocket?",
     "Протокол двунаправленной связи поверх HTTP. Используется в VMess/VLESS как обёртка для прохождения через прокси и CDN."),
    ("Что такое gRPC?",
     "Высокопроизводительный RPC от Google поверх HTTP/2. Может быть транспортом для VMess/VLESS — хорошо проходит через файрволы."),

    # Censorship
    ("Какие сервисы блокируют в России?",
     "Instagram, Facebook, Twitter/X, LinkedIn, медиа (Meduza, BBC RU), часть VPN-сервисов, OpenAI/ChatGPT, Discord (частично). Список меняется."),
    ("Какие протоколы блокирует РКН?",
     "Активно: OpenVPN, WireGuard в чистом виде, классический Shadowsocks, VMess. Сложно блокируются: VLESS+Reality, Hysteria, Naive (выглядят как HTTPS)."),
    ("Что такое Zapret?",
     "Утилита для обхода DPI на маршрутизаторе или ПК. Манипулирует пакетами (TCP segmentation, fake TLS) чтобы запутать DPI. Альтернатива VPN для определённых блокировок."),
    ("Что такое GoodbyeDPI?",
     "Утилита под Windows для обхода активного DPI. Принцип похож на Zapret — фрагментация пакетов, изменение SNI. Не VPN, работает локально."),
]


# =============================================================================
# 6. TROUBLESHOOTING — типичные проблемы
# =============================================================================

TROUBLESHOOT_QA = [
    ("VLESS не подключается, что делать?",
     "Проверь: 1) UUID совпадает. 2) Reality dest доступен с твоего IP. 3) Порт открыт (443 обычно). 4) SNI совпадает с serverName на сервере. 5) shortId совпадает или оба пустые. 6) Время на клиенте/сервере синхронизировано."),
    ("WireGuard не пингует через туннель.",
     "Проверь: 1) AllowedIPs правильные. 2) iptables MASQUERADE настроен на сервере. 3) IPv4 forwarding включён (sysctl net.ipv4.ip_forward=1). 4) Файрвол пропускает UDP 51820."),
    ("OpenVPN падает с TLS handshake failed.",
     "Возможно: 1) Сертификаты не совпадают. 2) Время рассинхронизировано (Sertifikate eingeschränkt по времени). 3) cipher или auth разные. 4) MTU слишком большой (попробуй mssfix 1300)."),
    ("Shadowsocks подключается но интернет не работает.",
     "Проверь: 1) Шифр совпадает с сервером. 2) Пароль точно тот же. 3) Mode установлен правильно (tcp_only/tcp_and_udp). 4) Сервер реально слушает на порту (ss -tlnp)."),
    ("Reality показывает 'TLS handshake failed'.",
     "Возможно dest-сервер не отвечает или серверный fingerprint не подходит. Поменяй dest на другой популярный домен (www.cloudflare.com, www.amazon.com). Проверь что privateKey/publicKey пара корректна."),
    ("Trojan возвращает 404.",
     "Это нормально для НЕправильного пароля — Trojan fallback на обычный HTTPS-ответ сервера. Проверь пароль точно. Также проверь sni."),
    ("Hysteria сильно лагает.",
     "1) Bandwidth установлен слишком высоким для канала. 2) UDP блокируется/ограничивается провайдером. 3) Попробуй congestion control bbr вместо brutal."),
    ("Скорость через VPN сильно меньше.",
     "Это норма — VPN всегда даёт overhead (~10-30%). Помогает: WireGuard (быстрее OpenVPN), сервер ближе географически, увеличение MTU."),
    ("Утечка DNS, что делать?",
     "В VPN-клиенте включи 'Use DNS from VPN'. Или явно укажи 1.1.1.1/8.8.8.8 в системных настройках. Проверь утечку: dnsleaktest.com."),
    ("Не работает на мобильном.",
     "1) Файрвол оператора блокирует UDP (для WG/Hysteria — попробуй TCP-обёртку). 2) Перегруженный 4G не пропускает большие пакеты — снизь MTU до 1280. 3) Проверь не блокирует ли антивирус приложение."),
]


# =============================================================================
# 7. «Я НЕ ЗНАЮ» PATTERNS — защита от галлюцинаций
# =============================================================================

IDK_QA = [
    ("Что такое ProtocolXYZ?",
     "Не знаю про ProtocolXYZ. Возможно это нишевая разработка или ты опечатался. Уточни источник или название."),
    ("Сколько пользователей у моего сервера?",
     "Не знаю — у меня нет доступа к твоему серверу. Проверь панель управления (3X-UI / Marzban) или логи xray."),
    ("Какой пароль у моего конфига?",
     "Не знаю и не должен знать. Пароли хранятся только в твоём конфиг-файле и на сервере. Я не имею доступа к ним."),
    ("Будет ли работать VPN через год?",
     "Не знаю. Это зависит от развития блокировок, изменений протоколов, действий регуляторов. Сейчас работает — что будет потом непредсказуемо."),
    ("Кто заблокирует Reality?",
     "Не знаю. Reality эффективен против текущих DPI, но это игра кошки и мышки — техники блокировок развиваются."),
    ("Какой VPN-сервер выбрать?",
     "Не могу советовать конкретного провайдера — это зависит от твоих требований (география, цена, скорость, политика логов). Сравнивай на vpncomparison.org или похожих сайтах."),
    ("Сколько стоит SaaS-VPN?",
     "Цены меняются. Сейчас обычно $3-12/месяц за легитимные сервисы (NordVPN, Mullvad, ProtonVPN). Точную цену смотри на сайтах провайдеров."),
    ("Когда выйдет следующая версия Xray?",
     "Не знаю — releases публикуются на github.com/XTLS/Xray-core. Смотри Releases там."),
    ("Что лучше для меня?",
     "Не могу решить за тебя. Скажи конкретный сценарий: какая ОС, какие задачи, какой бюджет, какие сайты нужно открыть — тогда дам рекомендацию."),
    ("Как взломать VPN-сервер?",
     "Я не помогаю с несанкционированным доступом. Это незаконно и не моя задача."),
    ("Дай мне готовый рабочий конфиг.",
     "Не могу — у меня нет реального сервера. Я могу показать ШАБЛОН конфига, ты подставишь свои IP, ключи, пароли."),
    ("Где найти бесплатный VLESS?",
     "Не знаю и не рекомендую — бесплатные VLESS-серверы часто перегружены, нестабильны или собирают данные. Подними свой за $3-5/мес на DigitalOcean/Vultr/Contabo."),
    ("Что значит ошибка 0x1234?",
     "Не знаю про конкретно такую ошибку. Покажи полный текст сообщения или скрин — отвечу точнее."),
]


# =============================================================================
# 8. CLARIFY — просьба уточнить вопрос
# =============================================================================

CLARIFY_QA = [
    ("Помоги.", "С чем именно? Опиши задачу — настройка сервера, выбор протокола, отладка ошибки?"),
    ("Не работает.", "Что именно не работает? Какой VPN-протокол, какой клиент, какая ошибка?"),
    ("Подскажи.", "Подскажу. О какой именно VPN/прокси теме вопрос?"),
    ("Что делать?", "Опиши ситуацию подробнее. Что хочешь настроить или какая проблема?"),
    ("Расскажи.", "О чём конкретно? VPN-протоколы, обход блокировок, настройка клиента, серверные конфиги?"),
    ("Как настроить?", "Что настроить? Сервер? Клиент? Какой протокол? Под какую ОС?"),
    ("Какой выбрать?", "Что выбрать — VPN-протокол, клиент, провайдера? Опиши сценарий — задача и ограничения."),
    ("Дай команду.", "Команду для чего? Установка, генерация ключей, проверка статуса? Опиши задачу."),
    ("Это работает?", "Что именно? Конкретный конфиг? Опиши что хочешь проверить."),
]


# =============================================================================
# 9. ПЕРЕФРАЗИРОВКИ
# =============================================================================

PREFIXES = [
    "{q}", "Пожалуйста, {q}", "Скажи, {q}",
    "А {q}", "Не знаешь {q}", "Подскажи: {q}",
    "Помоги: {q}", "Кстати, {q}",
    "Объясни: {q}", "Расскажи: {q}",
    "Не подскажешь {q}", "Хочу узнать: {q}",
    "Слушай, {q}", "Скажи мне {q}",
]


def lowfirst(s):
    return s[0].lower() + s[1:] if s else s


def make_paraphrases(q, a, n=40):
    variants = [(q, a)]
    base = q.rstrip("?.!").strip()
    base_low = lowfirst(base)

    # Универсальные префиксы
    for tpl in PREFIXES:
        suffix = "?" if q.rstrip().endswith("?") else "."
        new_q = tpl.format(q=base_low) + suffix
        variants.append((new_q, a))

    # «Что такое X»
    if base.lower().startswith("что такое "):
        topic = base[len("Что такое "):].lower()
        for c in [
            f"Объясни {topic}.",
            f"Расскажи про {topic}.",
            f"Поясни {topic}.",
            f"Что значит {topic}?",
            f"Опиши {topic}.",
            f"Дай определение: {topic}.",
            f"А {topic} — это что?",
            f"{topic[0].upper()}{topic[1:]} — что это?",
            f"Зачем нужен {topic}?",
        ]:
            variants.append((c, a))

    elif base.lower().startswith("чем "):
        variants.append((f"В чём разница, {base_low}?", a))
        variants.append((f"Объясни чем {base_low}?", a))

    elif base.lower().startswith("какая "):
        variants.append((f"Какой {base[6:].lower()}?", a))

    elif base.lower().startswith("какой "):
        variants.append((f"Какие {base[6:].lower()}?", a))

    elif base.lower().startswith("покажи "):
        rest = base[7:]
        variants.append((f"Можешь показать {rest.lower()}?", a))
        variants.append((f"Дай {rest.lower()}.", a))
        variants.append((f"Приведи пример: {rest.lower()}.", a))

    elif base.lower().startswith("как "):
        rest = base[4:]
        variants.append((f"Подскажи как {rest.lower()}?", a))
        variants.append((f"Расскажи как {rest.lower()}.", a))
        variants.append((f"Как правильно {rest.lower()}?", a))

    return variants[:n]


# =============================================================================
# СБОР
# =============================================================================

def generate(target=5000):
    base = []
    base.extend(VPN_QA)
    base.extend(PROXY_QA)
    base.extend(CONFIG_QA)
    base.extend(TOOLS_QA)
    base.extend(CONCEPTS_QA)
    base.extend(TROUBLESHOOT_QA)
    base.extend(IDK_QA)
    base.extend(CLARIFY_QA)

    expanded = []
    for q, a in base:
        expanded.extend(make_paraphrases(q, a))

    while len(expanded) < target * 2:
        b = random.choice(base)
        expanded.extend(make_paraphrases(*b, n=25))

    seen = set()
    unique = []
    for q, a in expanded:
        key = (q.strip(), a.strip())
        if key in seen or not q.strip() or not a.strip():
            continue
        seen.add(key)
        unique.append((q.strip(), a.strip()))

    random.shuffle(unique)
    return unique[:target]


def save(pairs, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for q, a in pairs:
            f.write(f"### Question: {q}\n### Answer: {a}\n\n")


if __name__ == "__main__":
    pairs = generate(5000)
    out = Path("data/texts/mi_vpn_training.txt")
    save(pairs, out)
    size_kb = out.stat().st_size / 1024
    print(f"OK: {len(pairs)} VPN Q&A pairs, {size_kb:.1f} KB")
    print(f"Saved: {out}")
