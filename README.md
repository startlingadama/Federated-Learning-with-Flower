
# Federated Learning with Flower, TensorFlow & PyTorch

Ce projet montre comment déployer un système de **Federated Learning (FL)** à l'aide de [Flower](https://flower.dev/), en intégrant à la fois des clients **TensorFlow** et **PyTorch**, avec une architecture entièrement **dockerisée**.

---

## Objectifs

- Démontrer l'entraînement collaboratif de modèles IA sur des clients hétérogènes.
- Supporter à la fois TensorFlow et PyTorch dans un système FL.
- Simuler des clients FL dans des conteneurs Docker.
- Utiliser Flower pour orchestrer les communications FL.


---

## Prérequis

- Docker
- Docker Compose

---

## Lancer le projet

```bash
docker-compose up --build
```

### Dashboard:
```link
http://localhost:8080/dashboard
```

Cela démarre :

* 1 serveur Flower (`server.py`)
* 1 client TensorFlow (`client_tf`)
* 1 client PyTorch (`client_pt`)

Les clients se connecteront automatiquement au serveur pour participer à 3 rounds de fédération.

---

## Communication

* Tous les composants communiquent sur le port `8080`.
* `NumPyClient` est utilisé pour faciliter l'intégration entre Flower et les frameworks IA.

---


## Références

* [Flower Docs](https://flower.dev/docs/)
* [TensorFlow](https://www.tensorflow.org/)
* [PyTorch](https://pytorch.org/)
* [Federated Learning Overview](https://en.wikipedia.org/wiki/Federated_learning)

---

## On peut aussi faire: 

* Intégration de vrais datasets (e.g. MNIST).
* Ajout de clients mobiles ou edge.
* Intégration avec des APIs gRPC ou GraphQL pour le contrôle externe.

---

## Developper:
- **Adama Coulibaly**: AI/ML Developer
