import joblib
from sklearn.ensemble import RandomForestClassifier

# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense

# import torch
# import torch.nn as nn
# import torch.optim as optim

from mockdata import get_mock_data

# Get the mock data
X_train, X_test, y_train, y_test = get_mock_data()



# Train a simple model
clf = RandomForestClassifier()
clf.fit(X_test, y_test)

# Save the model
joblib.dump(clf, "models/sklearn_model.pkl")



# # Build a simple model
# model = Sequential([
#     Dense(10, activation='relu', input_shape=(10,)),
#     Dense(1, activation='sigmoid')
# ])
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# model.fit(X_test, y_test, epochs=5)

# # Save the model
# model.save("models/tensorflow_model.keras")



# # Define a simple model
# class SimpleNN(nn.Module):
#     def __init__(self):
#         super(SimpleNN, self).__init__()
#         self.fc1 = nn.Linear(10, 10)
#         self.fc2 = nn.Linear(10, 1)

#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.sigmoid(self.fc2(x))
#         return x

# # Train the model
# model = SimpleNN()
# criterion = nn.BCELoss()
# optimizer = optim.Adam(model.parameters())

# X_tensor = torch.tensor(X_test, dtype=torch.float32)
# y_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# for epoch in range(5):
#     optimizer.zero_grad()
#     outputs = model(X_tensor)
#     loss = criterion(outputs, y_tensor)
#     loss.backward()
#     optimizer.step()

# # Save the model
# torch.save(model, "models/pytorch_model.pt")