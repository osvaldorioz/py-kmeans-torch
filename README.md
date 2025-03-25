
#### **¿Qué es K-Means?**
K-Means es un algoritmo de clustering no supervisado que agrupa datos en **k** clusters basados en la minimización de la distancia intra-cluster. Funciona iterativamente siguiendo estos pasos:
1. Se seleccionan **k** centroides aleatorios.
2. Se asigna cada punto de datos al centroide más cercano.
3. Se recalculan los centroides como la media de los puntos asignados a cada cluster.
4. Se repiten los pasos 2 y 3 hasta que los centroides dejen de cambiar significativamente.

#### **Implementación en C++**
1. **Clase `KMeans` en C++**  
   - Implementada con `torch::Tensor` para manejar datos y cálculos con PyTorch.
   - Utiliza `torch::cdist` para calcular distancias entre puntos y centroides.
   - Entrenamiento basado en la actualización iterativa de los centroides hasta la convergencia.
   - Usa `Pybind11` para exponer la funcionalidad a Python.

2. **TorchScript para Exportación**  
   - Se entrena el modelo en Python.
   - Se convierte en un módulo `torch.jit.trace` para guardarlo y cargarlo sin necesidad del código fuente original.

#### **Implementación en Python**
1. **Generación de datos sintéticos**  
   - Se generan **300 puntos** en **2 dimensiones** con **3 clusters** usando `numpy`.

2. **Uso del Modelo K-Means**  
   - Se carga la implementación de C++ en Python con `pybind11`.
   - Se entrena el modelo con `model.fit(X_tensor)`.
   - Se predicen etiquetas con `model.predict(X_tensor)`.
   - Se extraen centroides con `model.get_centroids()`.

3. **Guardado con TorchScript**  
   - Se usa `torch.jit.trace` para convertir el modelo entrenado en un archivo `.pt`.

4. **Visualización de Resultados**  
   - **Gráfica de dispersión**: muestra los datos clasificados con sus centroides en rojo.
   - **Gráfica de pérdida**: muestra la distancia promedio dentro de cada cluster.

---
