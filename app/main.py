from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
from typing import List
import matplotlib
import matplotlib.pyplot as plt
import torch
import kmeans
import json

matplotlib.use('Agg')  # Usar backend no interactivo
app = FastAPI()

# Generar datos de prueba
def generate_data(n_samples=300, n_features=2, n_clusters=3):
    data = []
    centers = np.random.rand(n_clusters, n_features) * 10
    for center in centers:
        cluster_data = center + np.random.randn(n_samples // n_clusters, n_features)
        data.append(cluster_data)
    return np.vstack(data), centers

# Definir el modelo para el vector
class VectorF(BaseModel):
    vector: List[float]
    
@app.post("/kmeans-torch")
def calculo(k: int, max_iters: int):
    output_file_1 = 'scatter_plot.png'
    output_file_2 = "loss_plot.png"
    # Datos
    X, true_centers = generate_data()
    X = np.array(X, dtype=np.float32)  # Convertir a float32
    X_tensor = torch.tensor(X)

    # Entrenar modelo
    #k=3
    #max_iters=100
    model = kmeans.KMeans(k, max_iters)
    model.fit(X_tensor)
    labels = model.predict(X_tensor).numpy()
    centroids = model.get_centroids().numpy().astype(np.float32)  # Convertir a float32

    # Gráfica de dispersión
    plt.figure(figsize=(6, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, label='Centroides')
    plt.legend()
    plt.title("Clustering K-Means")
    plt.savefig(output_file_1)
    #plt.show()

    # Gráfica de pérdida (distancia intra-cluster)
    intra_cluster_distances = [torch.cdist(torch.tensor(X[labels == i], dtype=torch.float32), torch.tensor([centroids[i]], dtype=torch.float32)).mean().item() for i in range(3)]
    plt.figure(figsize=(6, 4))
    plt.bar(range(3), intra_cluster_distances, color='blue')
    plt.xlabel("Cluster")
    plt.ylabel("Distancia media")
    plt.title("Distancia intra-cluster")
    plt.savefig(output_file_2)
    plt.close()
    
    j1 = {
        "Scatter plot": output_file_1, 
        "Loss plot": output_file_2
    }
    jj = json.dumps(str(j1))

    return jj

@app.get("/kmeans-torch-graph")
def getGraph(output_file: str):
    return FileResponse(output_file, media_type="image/png", filename=output_file)