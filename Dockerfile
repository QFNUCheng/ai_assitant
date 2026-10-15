FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Asia/Shanghai

WORKDIR /app

# 本机在腾讯云，改用腾讯云镜像源，避免默认 deb.debian.org 从国内访问过慢。
# 只装 curl（健康检查要用）；其余依赖均有预编译 wheel，无需 gcc/g++，
# 省掉 apt 约 292MB 的编译工具链，显著缩短构建时间。
RUN set -eux; \
    sed -i 's|deb.debian.org|mirrors.cloud.tencent.com|g; s|security.debian.org|mirrors.cloud.tencent.com|g' \
        /etc/apt/sources.list /etc/apt/sources.list.d/*.sources 2>/dev/null || true; \
    apt-get update; \
    apt-get install -y --no-install-recommends curl; \
    rm -rf /var/lib/apt/lists/*

# 先装依赖，让这层能被缓存（改代码不必重装依赖）。同样走腾讯云 PyPI 镜像。
COPY requirements.txt .
RUN pip install --upgrade pip -i https://mirrors.cloud.tencent.com/pypi/simple \
        --trusted-host mirrors.cloud.tencent.com \
    && pip install -r requirements.txt -i https://mirrors.cloud.tencent.com/pypi/simple \
        --trusted-host mirrors.cloud.tencent.com

COPY . .

# 仓库在 Windows 上开发，shell 脚本可能是 CRLF 换行，会导致
# "exec ./entrypoint.sh: no such file or directory"（shebang 变成 #!/bin/bash\r）。
# 这里统一归一到 LF，并确保可执行。
RUN find . -name '*.sh' -type f -exec sed -i 's/\r$//' {} + \
    && chmod +x entrypoint.sh

EXPOSE 8000

# entrypoint.sh 负责：等 MySQL → 校验/构建知识库向量 → 起 uvicorn
ENTRYPOINT ["./entrypoint.sh"]
