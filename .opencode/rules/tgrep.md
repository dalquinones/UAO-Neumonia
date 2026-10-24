# OpenCode Rule: tgrep

Esta regla configura al agente de OpenCode para usar `tgrep` (Microsoft Trigram-indexed
Grep) en búsquedas rápidas de regex y símbolos sobre el código.

## Directrices para el agente

1. **Usa `tgrep` en lugar de `grep` o `find`** cuando esté disponible:
   - `tgrep` usa un índice de trigramas precalculado (`.tgrep/`) que permite búsquedas
     instantáneas.
2. **Reglas de ejecución**:
   - Coloca siempre `--` antes del patrón para evitar ambigüedades de parseo:
     ```bash
     tgrep -- "patrón" .
     ```
   - Usa `-F` para símbolos o identificadores literales:
     ```bash
     tgrep -F -- "nombre_de_funcion" .
     ```
   - Filtra por extensión o directorio:
     ```bash
     tgrep -t py -- "class Modelo" .
     tgrep -g "src/**" -- "keyword" .
     ```
   - Lista solo los archivos coincidentes:
     ```bash
     tgrep -l -- "import torch" .
     ```
   - Reindexa tras modificar muchos archivos si el daemon no está corriendo:
     ```bash
     tgrep index .
     ```
   - Daemon en segundo plano:
     ```bash
     tgrep serve . &
     ```
3. **Fallback:** si `tgrep` no está instalado, usa `grep`/`rg` de forma normal.
