# Convertir les données en tensors PyTorch
water_ride_train_data.fillna(water_ride_train_data.median(numeric_only=True), inplace=True)
X = water_ride_train_data[['HEURE_JOURNEE', 'CURRENT_WAIT_TIME', 'TIME_TO_PARADE_1', 'TIME_TO_PARADE_2', 'TIME_TO_NIGHT_SHOW', 'ADJUST_CAPACITY',  'DOWNTIME']]
y = water_ride_train_data['WAIT_TIME_IN_2H']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Convertir les données au format approprié pour LSTM
X_train = np.reshape(X_train.values, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test.values, (X_test.shape[0], 1, X_test.shape[1]))
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
# Définir le modèle de réseau de neurones plus complexe

import torch.nn.init as init

class ComplexLSTM(nn.Module):
    def __init__(self):
        super(ComplexLSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_size=X_train.shape[2], hidden_size=50, num_layers=3, batch_first=True)
        self.dropout1 = nn.Dropout(p=0.2)
        self.lstm2 = nn.LSTM(input_size=50, hidden_size=50, num_layers=2, batch_first=True)
        self.dropout2 = nn.Dropout(p=0.2)
        self.fc = nn.Linear(50, 1)

        # Initialisation des poids avec la méthode de He
        for name, param in self.named_parameters():
            if 'weight' in name:
                init.kaiming_normal_(param)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out)
        out = self.fc(out[:, -1, :])  # Utiliser uniquement la dernière séquence de sortie
        return out


# Initialiser le modèle
model = ComplexLSTM()

# Définir la fonction de perte et l'optimiseur
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)

# Entraîner le modèle
num_epochs = 3000
losses = []

for epoch in range(num_epochs):
    # Calculer les prédictions du modèle
    predictions = model(X_train)

    # Calculer la perte
    loss = criterion(predictions, y_train)

    # Optimiser le modèle
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Enregistrer la perte pour l'affichage
    losses.append(loss.item())

    # Afficher la perte tous les 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f'Époque [{epoch + 1}/{num_epochs}], Perte : {loss.item():.4f}')

# Afficher la courbe de perte au fil des itérations
plt.plot(losses)
plt.xlabel('Époque')
plt.ylabel('Perte')
plt.title('Évolution de la perte au fil des époques')
plt.show()

# Afficher les labels prédits par le modèle
with torch.no_grad():
    model.eval()
    predicted_labels_flying_coaster = model(X_train)
    print("Labels Prédits :", predicted_labels_flying_coaster.flatten().numpy())
