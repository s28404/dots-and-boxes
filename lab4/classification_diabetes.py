#############################
#   Autorzy:
#   Kajetan Frąckowiak s28404
#   Marek Walkowski    s25378
#############################

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.decomposition import PCA

# Link do danych: https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
column_names = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome",
]
data = pd.read_csv(url, names=column_names)

print("--- Dane załadowane ---")
print(data.head())

# Podział na X (cechy) i y (etykiety)
# data.shape = (768, 9), gdzie 768 to liczba próbek, a 9 to liczba kolumn (cech + etykieta)
# usuwamy kolumnę 'Outcome' z danych wejściowych X która jest etykietą (y)
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Podział na zbiór treningowy i testowy
# X_shape: (768, 8), y_shape: (768,)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Skalowanie danych (ważne dla SVM)
scaler = StandardScaler()
# Przekształcamy dane -> [mean=0, std=1]
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n--- Generowanie wizualizacji... ---")
# Używamy PCA do wizualizacji 2D (2 główne komponenty)
pca = PCA(n_components=2)
# X_pca shape: (614, 2) gdzie 614 to liczba próbek w X_train a 2 to liczba komponentów
X_pca = pca.fit_transform(X_train_scaled)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    # y_train zawiera etykiety 0 lub 1
    # c to kolor punktów 0 to kolor ciemny fiolet, 1 to jasny żółty
    X_pca[:, 0], X_pca[:, 1], c=y_train,
)
plt.title("Wizualizacja danych (PCA) - Cukrzyca (0=Nie, 1=Tak)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(scatter, label="Outcome")
plt.show()

print("\n--- Drzewo Decyzyjne ---")
# Max_depth to ile poziomów ma mieć drzewo
dt_clf = DecisionTreeClassifier(random_state=42, max_depth=5)
dt_clf.fit(X_train, y_train)  # Drzewa nie wymagają skalowania, ale można użyć
y_pred_dt = dt_clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Raport klasyfikacji:\n", classification_report(y_test, y_pred_dt))

# Używamy do wizualizacji, później trenujemy jeszcze raz na czterech różnych kernelach
print("\n--- SVM (Domyślny RBF) ---")
svm_clf = SVC(kernel="rbf", random_state=42)
svm_clf.fit(X_train_scaled, y_train)
y_pred_svm = svm_clf.predict(X_test_scaled)

# Accuracy = TP+TN / (TP+TN+FP+FN)
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Raport klasyfikacji:\n", classification_report(y_test, y_pred_svm))

print("\n--- Przykładowa predykcja ---")
# values zawraca numpy array zamiast pandas Series, a my potrzebujemy 2D array dla predict
# X_test.shape = (154, 8), X_test.iloc[0] shape = (8,), X_test.iloc[0].values.reshape(1, -1) shape = (1, 8)
sample_data = X_test.iloc[0].values.reshape(1, -1)
sample_data_scaled = scaler.transform(sample_data)

print(f"Dane wejściowe: {sample_data}")
# sample_data shape = (1, 8), gdzie indeks 0 to liczba próbek, a 8 to liczba cech
# predict zwraca 0 lub 1
pred_dt = dt_clf.predict(sample_data)[0] # chcemy pojedynczą wartość która jest na przedziale 0 lub 1
# sample_data_scaled shape = (1, 8), gdzie indeks 0 to liczba próbek, a 8 to liczba cech
pred_svm = svm_clf.predict(sample_data_scaled)[0]
print(f"Predykcja Drzewa: {pred_dt} ({'Cukrzyca' if pred_dt == 1 else 'Zdrowy'})")
print(f"Predykcja SVM:    {pred_svm} ({'Cukrzyca' if pred_svm == 1 else 'Zdrowy'})")
# y_test.shape = (154,)
# wybieramy iloc[0], czyli pierwszą wartość z y_test, iloc[1] to druga wartość itd.
print(f"Prawdziwa wartość: {y_test.iloc[0]}")

print("\n--- Eksperyment z Kernelami SVM ---")
kernels = ["linear", "poly", "rbf", "sigmoid"]
results = []

for k in kernels:
    model = SVC(kernel=k, random_state=42)
    model.fit(X_train_scaled, y_train)
    # model.score zwraca accuracy TP+TN / (TP+TN+FP+FN)
    acc = model.score(X_test_scaled, y_test)
    results.append((k, acc))
    print(f"Kernel: {k:10} | Accuracy: {acc:.4f}")

print("\nPodsumowanie wpływu kerneli:")
# results.shape = (4, 2) -> [(kernel1, acc1), (kernel2, acc2), ...]
# bierzemy x[1] a nie x[0], bo x[1] to accuracy, a mu chcemy z najlepszym accuracy
# best_kernel to krotka (kernel, accuracy)
# x[0] to nazwa kernela
best_kernel = max(results, key=lambda x: x[1])
print(
    f"Najlepszy wynik dał kernel: '{best_kernel[0]}' z dokładnością {best_kernel[1]:.4f}."
)

