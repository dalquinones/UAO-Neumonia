# Plantilla de Módulo de Inteligencia Artificial (Template)

> **Uso de esta plantilla:** Utiliza este archivo como guía base cada vez que vayas a crear un nuevo módulo dentro de la carpeta `src/`.

---

## 📌 Propósito del Módulo
Define claramente cuál es la **Responsabilidad Única** de este script.

- **Objetivo principal:** (Ej. Extraer características de la imagen, limpiar datos, etc.)
- **Dependencias clave:** (Ej. `numpy`, `tensorflow`, `pydicom`)

---

## 📐 Diseño Clean Architecture

- **Alta Cohesión:** El módulo no debe hacer más de una cosa. Si hace preprocesamiento, no debe guardar archivos en disco.
- **Bajo Acoplamiento:** Este módulo **NO** debe importar librerías de interfaz gráfica (`tkinter`, `streamlit`, etc.). Toda la comunicación con el exterior se hace recibiendo parámetros en sus funciones y retornando valores o tuplas.

---

## 📝 Estructura Base de Código (Ejemplo)

```python
import numpy as np
from typing import Any

def procesar_datos(entrada: np.ndarray, configuracion: dict = None) -> np.ndarray:
    """
    Descripción breve de lo que hace la función.
    
    Args:
        entrada (np.ndarray): Descripción detallada del parámetro y su formato esperado.
        configuracion (dict, optional): Parámetros adicionales de configuración.
        
    Returns:
        np.ndarray: Descripción de lo que se devuelve.
        
    Raises:
        ValueError: Si los datos de entrada están corruptos.
        TypeError: Si la entrada no es del tipo esperado.
    """
    # 1. Validación de entradas
    if not isinstance(entrada, np.ndarray):
        raise TypeError("Se esperaba un arreglo NumPy.")
        
    # 2. Lógica principal
    resultado = entrada * 2.0  # Ejemplo
    
    # 3. Retorno
    return resultado
```

---

## ✅ Lista de Chequeo antes de enviar a PR
- [ ] La función tiene Type Hints (Ej. `-> np.ndarray:`).
- [ ] La función tiene un Docstring siguiendo el estándar PEP8 (Args, Returns, Raises).
- [ ] Las variables tienen nombres descriptivos en inglés o español técnico (no `x`, `y`, `var1`).
- [ ] Se crearon al menos 2 pruebas unitarias en la carpeta `test/` (una de éxito y una de fallo esperado).
