ARG BASE_IMAGE=mcr.microsoft.com/devcontainers/base:noble
FROM ${BASE_IMAGE}

# Extra Ubuntu packages for the devcontainer:
#   inetutils-ping, mc  — network debugging / file management
#   tmux                — terminal multiplexer
# hadolint ignore=DL3008
RUN apt-get update \
	&& apt-get install -y --no-install-recommends \
		inetutils-ping \
		mc \
		tmux \
	&& apt-get clean \
	&& rm -rf /var/lib/apt/lists/*

# EOF
