.PHONY: help run test test-all test-read test-preprocess test-model test-gradcam test-integrator test-gui commit clean
SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
export PATH := $(HOME)/.local/bin:$(PATH)
TIMESTAMP := $(shell date +"%Y-%m-%d_%H-%M-%S")

MODULE ?= all


help:
	@echo "Comandos disponibles:"
	@echo "  make run              - Ejecuta la aplicación de detección"
	@echo "  make test             - Corre todas las pruebas unitarias (test_all_...log)"
	@echo "  make test-read        - Prueba Módulo 1 (read_img)"
	@echo "  make test-preprocess  - Prueba Módulo 2 (preprocess_img)"
	@echo "  make test-model       - Prueba Módulo 3 (load_model)"
	@echo "  make test-gradcam     - Prueba Módulo 4 (grad_cam)"
	@echo "  make test-integrator  - Prueba Módulo 5 (integrator)"
	@echo "  make test-gui         - Prueba la interfaz gráfica (detector_neumonia)"
	@echo "  make clean            - Limpia archivos temporales y cachés"
	@echo "  make commit           - Guarda los cambios en Git (opcional: m=\"mensaje\")"

run:
	PYTHONPATH=. uv run --no-project src/detector_neumonia.py


test:
	@mkdir -p reports
ifeq ($(MODULE),all)
	PYTHONPATH=. uv run --no-project pytest test/ -v | tee reports/test_all_$(TIMESTAMP).log
	@cp reports/test_all_$(TIMESTAMP).log reports/latest.log
	@echo "📁 Reporte guardado: reports/test_all_$(TIMESTAMP).log"
else
	PYTHONPATH=. uv run --no-project pytest test/test_$(MODULE).py -v | tee reports/test_$(MODULE)_$(TIMESTAMP).log
	@cp reports/test_$(MODULE)_$(TIMESTAMP).log reports/latest.log
	@echo "📁 Reporte guardado: reports/test_$(MODULE)_$(TIMESTAMP).log"
endif


test-all:
	@$(MAKE) test MODULE=all

test-read:
	@$(MAKE) test MODULE=read_img

test-preprocess:
	@$(MAKE) test MODULE=preprocess_img

test-model:
	@$(MAKE) test MODULE=load_model

test-gradcam:
	@$(MAKE) test MODULE=grad_cam

test-integrator:
	@$(MAKE) test MODULE=integrator

clean:
	rm -rf __pycache__ .pytest_cache src/__pycache__ test/__pycache__ *.pyc

commit:
	@git status -s
	@echo "----------------------------------------"
	@if [ -z "$(m)" ]; then \
		read -p "Escribe el mensaje del commit: " msg; \
		git add . && git commit -m "$$msg"; \
	else \
		git add . && git commit -m "$(m)"; \
	fi

test-gui:
	@$(MAKE) test MODULE=detector_neumonia

