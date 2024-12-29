SHELL = /bin/sh
UID := $(shell id -u)
GID := $(shell id -g)

export UID
export GID

init:
	docker compose build
	docker compose up -d
	docker compose exec -it python poetry install

start:
	docker compose up -d

run:
	docker compose exec -it python poetry run python gothic_zoologist

shell:
	docker compose exec -it python bash

.PHONY: init, start, run, shell