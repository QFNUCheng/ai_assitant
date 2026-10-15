#!/usr/bin/env bash
# 服务器端部署脚本（在服务器上的应用目录里执行）。
#
#   cd /opt/she_love_me && ./deploy/deploy.sh              # 构建并启动
#   cd /opt/she_love_me && ./deploy/deploy.sh --with-nginx # 顺带装宝塔 nginx 站点
#
# 部署形态遵循本机惯例：容器只监听 127.0.0.1:18120，HTTPS 由宝塔 nginx 负责。
# 不包含任何密钥；密钥一律放在同目录的 .env 里（不进 git、不进镜像）。
set -euo pipefail

APP_DIR="${APP_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DOMAIN="${DOMAIN:-ai.hunao.xin}"
APP_PORT="${APP_PORT:-18120}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-1200}"   # 首次建向量库可能较久
WITH_NGINX=0
[ "${1:-}" = "--with-nginx" ] && WITH_NGINX=1

cd "$APP_DIR"
log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[!] %s\033[0m\n' "$*"; }
die() { printf '\n\033[1;31m[✗] %s\033[0m\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- 0. 前置检查
preflight() {
  local avail_mb disk_avail
  avail_mb=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
  log "可用内存: ${avail_mb}MB"
  if [ "$avail_mb" -lt 500 ]; then
    warn "可用内存偏低（${avail_mb}MB）。构建可能变慢甚至被 OOM 杀掉。"
    warn "如构建失败，可先执行：docker builder prune -f   （只清构建缓存，不影响运行中的服务）"
  fi
  disk_avail=$(df -Pk "$APP_DIR" | awk 'NR==2{print int($4/1024/1024)}')
  log "可用磁盘: ${disk_avail}G"
  if [ "$disk_avail" -lt 3 ]; then
    warn "磁盘可用不足 3G，构建可能失败。"
  fi
  # 端口是否已被占用
  if (echo >/dev/tcp/127.0.0.1/"$APP_PORT") >/dev/null 2>&1; then
    warn "端口 ${APP_PORT} 已被占用，请确认是不是本项目已在运行；否则改 APP_PORT 后重试。"
  fi
}

# ---------------------------------------------------------------- 1. Docker
ensure_docker() {
  if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
    log "Docker 已就绪：$(docker --version)"
    return
  fi
  log "未检测到 Docker，开始安装（官方脚本）"
  if command -v dnf >/dev/null 2>&1; then
    dnf install -y curl
  elif command -v yum >/dev/null 2>&1; then
    yum install -y curl
  elif command -v apt-get >/dev/null 2>&1; then
    apt-get update -y && apt-get install -y curl
  fi
  curl -fsSL https://get.docker.com | sh
  systemctl enable --now docker
  command -v docker >/dev/null 2>&1 || die "Docker 安装失败"
  log "Docker 安装完成：$(docker --version)"
}

# ---------------------------------------------------------------- 2. 配置校验
check_env() {
  [ -f .env ] || die "缺少 .env —— 请先 cp deploy/.env.example .env 并填入真实值"
  local missing=()
  for key in DB_HOST DB_USER DB_PASSWORD DB_NAME LLM_API_KEY DASHSCOPE_API_KEY; do
    grep -qE "^${key}=.+" .env || missing+=("$key")
  done
  [ ${#missing[@]} -eq 0 ] || die ".env 里这些项为空：${missing[*]}"
  if grep -qE '^KB_SCOPE=' .env; then
    log "知识库范围：$(grep -E '^KB_SCOPE=' .env)（首次构建会按此范围调用 embedding）"
  fi
  log ".env 校验通过"
}

# ---------------------------------------------------------------- 3. 构建启动
build_and_start() {
  log "构建镜像并启动（首次安装依赖，请耐心）"
  docker compose up -d --build
  log "容器状态"
  docker compose ps
}

# ---------------------------------------------------------------- 4. 健康检查
wait_healthy() {
  log "等待应用就绪（最多 ${HEALTH_TIMEOUT}s；首次需构建向量库）"
  local waited=0
  while [ "$waited" -lt "$HEALTH_TIMEOUT" ]; do
    if docker compose exec -T app curl -fsS http://localhost:8000/health >/dev/null 2>&1; then
      log "应用已就绪 ✅"
      return 0
    fi
    sleep 10
    waited=$((waited + 10))
    printf '\r   已等待 %ss…' "$waited"
  done
  printf '\n'
  log "超时未就绪，最近日志："
  docker compose logs --tail 80 app
  die "健康检查超时"
}

# ---------------------------------------------------------------- 5. nginx 站点
install_nginx() {
  command -v nginx >/dev/null 2>&1 || { warn "未发现 nginx，跳过站点配置"; return 0; }
  local src="deploy/nginx/${DOMAIN}.conf"
  local dst="/www/server/panel/vhost/nginx/${DOMAIN}.conf"
  local certdir="/www/server/panel/vhost/cert/${DOMAIN}"
  [ -f "$src" ] || { warn "找不到 $src，跳过"; return 0; }

  log "配置宝塔 nginx 站点 ${DOMAIN}"
  if [ ! -f "${certdir}/fullchain.pem" ]; then
    warn "证书尚未就绪：${certdir}/fullchain.pem 不存在"
    warn "请先在宝塔面板 → 网站 → 添加站点 ${DOMAIN} → SSL 申请 Let's Encrypt 证书，"
    warn "或执行 deploy/nginx/${DOMAIN}.conf 末尾的 acme.sh 命令后再重跑本步骤。"
    warn "（先跳过 nginx 配置，应用本身已在本机 ${APP_PORT} 端口正常服务）"
    return 1
  fi
  [ -f "$dst" ] && cp -a "$dst" "${dst}.bak.$(date +%Y%m%d-%H%M%S)"
  cp -f "$src" "$dst"
  if nginx -t; then
    nginx -s reload
    log "nginx 已重载 ✅"
  else
    die "nginx 配置校验失败，已保留 ${dst}.bak.* 备份，请检查"
  fi
}

# ---------------------------------------------------------------- 6. 连通性
verify_public() {
  log "验证公网访问 https://${DOMAIN}/health"
  for i in $(seq 1 12); do
    if curl -fsS "https://${DOMAIN}/health" 2>/dev/null; then
      printf '\n✅ https://%s 已可访问\n' "$DOMAIN"
      return 0
    fi
    printf '\r   等待中… %s' "$i"
    sleep 10
  done
  printf '\n'
  warn "公网访问未通过。排查顺序："
  warn "  1) 域名是否解析到本机公网 IP"
  warn "  2) 宝塔是否已配置 ${DOMAIN} 站点并签发证书"
  warn "  3) 云服务器安全组是否放行 80/443"
  return 1
}

ensure_docker
preflight
check_env
build_and_start
wait_healthy

if [ "$WITH_NGINX" = "1" ]; then
  install_nginx && verify_public || true
else
  log "跳过 nginx 配置（加 --with-nginx 可在证书就绪后一并配置站点）"
  log "本地自测：curl -s http://127.0.0.1:${APP_PORT}/health"
fi

log "完成。常用命令：
  docker compose logs -f app     查看应用日志
  docker compose restart app     重启应用
  docker compose down            停止"
