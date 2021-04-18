import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader, WeightedRandomSampler
# mis modulos
from src.galaxiesConfig import arguments_parser
from src.nn import GalaxiesNet
from src.datasets import ClassifierDataset
from src.preprocessing_module import (convert_df_float,
                                      get_galaxies_distribution)
from src.early_stopping import EarlyStopping
from src.metrics import multi_acc
from src.visualizations import (watch_galaxies_distributiions,
                                torch_classification_visualizer,
                                plot_confusion_matrix)
sns.set(font_scale=1.5)
plt.style.use('dark_background')

# gpu o cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device, "para hacer entrenamiento")

# argumentos de entrada
args = arguments_parser()

# leer datos
df = pd.read_csv("data/glaxies_featured.csv")

# convertir a float
df = convert_df_float(df)

# distribución de las clases
sns.countplot(x='class', data=df)

# class hacia indice e indice hacia class [para los reportes de clasificación]
class2idx = {"spiral": 0, "elliptical": 1, "uncertain": 2}
idx2class = {v: k for k, v in class2idx.items()}
df['class'].replace(class2idx, inplace=True)

# features y targets
targets = ["class"]
features = list(df.columns)
for tar in targets:
    features.remove(tar)
x = df[features]
y = df["class"]

# dividir en train + val y test
x_trainval, x_test, y_trainval, y_test = train_test_split(
    x, y, test_size=args.validation_size,
    stratify=y, random_state=args.random_state)
# dividir train y val
x_train, x_val, y_train, y_val = train_test_split(
    x_trainval, y_trainval, test_size=0.1, stratify=y_trainval,
    random_state=args.random_state)

# estadarizar: [dado los polinomyal features agregados]
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

# pasar a numpy
x_train, y_train = np.array(x_train), np.array(y_train)
x_val, y_val = np.array(x_val), np.array(y_val)
x_test, y_test = np.array(x_test), np.array(y_test)

# features y targets
num_features = x_train.shape[1]
num_classes = pd.DataFrame(y_train, columns=["y"])["y"].nunique()

# mirar las distribuciones de las clases [ --> es similar en ambos conjuntos]
watch_galaxies_distributiions(y_train, y_val, y_test)

# dataset para pytorch
train_dataset = ClassifierDataset(
    torch.from_numpy(x_train).float(), torch.from_numpy(y_train).long())
val_dataset = ClassifierDataset(
    torch.from_numpy(x_val).float(), torch.from_numpy(y_val).long())
test_dataset = ClassifierDataset(
    torch.from_numpy(x_test).float(), torch.from_numpy(y_test).long())

# lista de tensores a computar
target_list = []
for _, counter in train_dataset:
    target_list.append(counter)
print(torch.tensor(target_list).size())
target_list = torch.tensor(target_list)
target_list = target_list[torch.randperm(len(target_list))]

# conteo de las clases disponibles ver el desbalanceamiento de la data
class_count = [i for i in get_galaxies_distribution(y_train).values()]
class_weights = 1./torch.tensor(class_count, dtype=torch.float)

# ver las clases disponibles
class_weights_all = class_weights[target_list]
print(class_weights_all.size())

# balancer conjuntos de training [teniendo en cuenta la distr]
weighted_sampler = WeightedRandomSampler(
    weights=class_weights_all,
    num_samples=len(class_weights_all),
    replacement=True)

# torch datasets
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=args.batch_size,
                          sampler=weighted_sampler)
val_loader = DataLoader(dataset=val_dataset, batch_size=1)
test_loader = DataLoader(dataset=test_dataset, batch_size=1)

# modelo
model = GalaxiesNet(num_feature=num_features,
                    num_class=num_classes)
model.to(device)
print(model)

# función de costos
criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
# optimizador

# optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
optimizer = torch.optim.Adamax(model.parameters(), lr=args.learning_rate)


# callbacks [uno de torch optim y el otro por mi]
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, "min",
    factor=args.lr_factor,
    patience=args.lr_patience,
    verbose=True)
early_stopping = EarlyStopping(patience=args.patience, verbose=True,
                               delta=args.min_delta,
                               path='models/checkpoint.pt')

# listas de metricas de evaluación
accuracy_stats = {
    'train': [],
    "val": []}

loss_stats = {
    'train': [],
    "val": []}

print("Empezemos el entrenamiento ...")

for e in range(1, args.epochs+1):

    # Entrenamiento
    train_epoch_loss = 0
    train_epoch_acc = 0
    model.train()

    for x_train_batch, y_train_batch in train_loader:

        # batches de data
        x_train_batch, y_train_batch = x_train_batch.to(
            device), y_train_batch.to(device)

        # setiar a cero los gradientes [en pytorch son acumulativos]
        optimizer.zero_grad()

        # fordward hacia adelante
        y_train_pred = model(x_train_batch)

        # error de entrenamiento
        train_loss = criterion(y_train_pred, y_train_batch)
        # acc de entrenamiento
        train_acc = multi_acc(y_train_pred, y_train_batch)

        # backpropragation de la perdida
        train_loss.backward()
        optimizer.step()

        # loss y acc
        train_epoch_loss += train_loss.item()
        train_epoch_acc += train_acc.item()

    # Validación [torch.no_grad() no hacer backpropagation]
    with torch.no_grad():
        model.eval()

        val_epoch_loss = 0
        val_epoch_acc = 0

        for x_val_batch, y_val_batch in val_loader:
            x_val_batch, y_val_batch = x_val_batch.to(
                device), y_val_batch.to(device)

            y_val_pred = model(x_val_batch)

            val_loss = criterion(y_val_pred, y_val_batch)
            val_acc = multi_acc(y_val_pred, y_val_batch)

            val_epoch_loss += val_loss.item()
            val_epoch_acc += val_acc.item()

    loss_stats['train'].append(train_epoch_loss/len(train_loader))
    loss_stats['val'].append(val_epoch_loss/len(val_loader))
    accuracy_stats['train'].append(train_epoch_acc/len(train_loader))
    accuracy_stats['val'].append(val_epoch_acc/len(val_loader))

    # computo de metricas promedio por epoca
    train_loss = train_epoch_loss / len(train_loader)
    val_loss = val_epoch_loss / len(val_loader)
    train_acc = train_epoch_acc / len(train_loader)
    val_acc = val_epoch_acc / len(val_loader)

    # learning rate bajada
    scheduler.step(val_loss)

    # early stopping
    early_stopping(val_loss, model)
    if early_stopping.early_stop:
        break

    print(f"Epoca {e+0: 03}: | Train Loss: {train_loss: .5f}",
          f"| Val Loss: {val_loss:.5f}",
          f"| Train Acc: {train_acc: .3f}"
          f"| Val Acc: {val_acc: .3f}")

# ver resultados
torch_classification_visualizer(loss_stats, accuracy_stats)

# testear los resultados
y_pred_list = []
with torch.no_grad():
    model.eval()
    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(device)

        y_test_pred = model(x_batch)
        print(y_test_pred.size())
        _, y_pred_tags = torch.max(y_test_pred, dim=1)
        y_pred_list.append(y_pred_tags.cpu().numpy())
y_pred_list = [a.squeeze().tolist() for a in y_pred_list]


# matriz de confusión
confusion_matrix_df = pd.DataFrame(confusion_matrix(
    y_test, y_pred_list)).rename(columns=idx2class, index=idx2class)

plot_confusion_matrix(confusion_matrix_df, cmap=plt.cm.hot)

# reporte de clasificación
print(classification_report(y_test, y_pred_list))
