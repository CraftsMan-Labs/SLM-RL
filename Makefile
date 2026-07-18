# Workshop stack. `make` → OrbStack if present, else Docker; size RAM for GRPO.
# GRPO needs ~5–6GiB; compose mem_limit is useless if the VM is smaller.

.DEFAULT_GOAL := up

# Host RAM (MiB): macOS sysctl, else /proc, else assume 8GiB.
HOST_MIB := $(shell \
	if sysctl -n hw.memsize >/dev/null 2>&1; then \
		sysctl -n hw.memsize | awk '{print int($$1/1024/1024)}'; \
	elif [ -r /proc/meminfo ]; then \
		awk '/^MemTotal:/{print int($$2/1024)}' /proc/meminfo; \
	else echo 8192; fi)

# Ideal VM: 8GiB workshop floor, but leave 4GiB for the host UI.
MEM_MIB := $(shell awk -v h='$(HOST_MIB)' 'BEGIN{ \
	t=8192; if (h > 0 && h - 4096 < t) t = h - 4096; \
	if (t < 2048) t = 2048; print t }')

MEM_LIMIT    ?= $(shell awk -v m=$(MEM_MIB) 'BEGIN{ printf "%dg", int((m + 512) / 1024) }')
MEMSWAP_LIMIT ?= $(shell awk -v m=$(MEM_MIB) 'BEGIN{ printf "%dg", int((m * 3 / 2 + 512) / 1024) }')
export MEM_LIMIT MEMSWAP_LIMIT

.PHONY: up down ensure-runtime

ensure-runtime:
	@command -v docker >/dev/null || { echo "install Docker or OrbStack"; exit 1; }
	@if command -v orb >/dev/null 2>&1; then \
		echo "runtime: OrbStack  mem_limit=$(MEM_LIMIT)  target_vm=$(MEM_MIB)MiB  host=$(HOST_MIB)MiB"; \
		docker context use orbstack >/dev/null 2>&1 || true; \
		cur=$$(orb config show 2>/dev/null | awk -F': ' '/^memory_mib:/{print $$2}'); \
		if [ "$${cur:-0}" -lt $(MEM_MIB) ]; then \
			echo "raising OrbStack memory_mib $${cur:-0} → $(MEM_MIB) (restarting OrbStack)"; \
			orb config set memory_mib $(MEM_MIB); \
			orb stop; orb start; \
			i=0; while ! docker info >/dev/null 2>&1; do \
				i=$$((i+1)); [ $$i -gt 90 ] && { echo "OrbStack did not come back"; exit 1; }; \
				sleep 1; \
			done; \
		else \
			echo "OrbStack memory_mib=$$cur (ok)"; \
		fi; \
	else \
		echo "runtime: Docker  mem_limit=$(MEM_LIMIT)  host=$(HOST_MIB)MiB"; \
		avail=$$(docker info --format '{{.MemTotal}}' 2>/dev/null || echo 0); \
		need=$$(( $(MEM_MIB) * 1024 * 1024 )); \
		if [ "$$avail" -lt "$$need" ]; then \
			echo "Docker VM has $$avail bytes; need ≥$(MEM_MIB)MiB."; \
			echo "Docker Desktop → Settings → Resources → Memory, then retry."; \
			exit 1; \
		fi; \
	fi

up: ensure-runtime
	docker compose up --build

down:
	docker compose down
