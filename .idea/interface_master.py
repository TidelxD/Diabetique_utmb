import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import tkinter as tk
from tkinter import messagebox

# Charger les données
data = pd.read_csv('diabetes.csv')


# fuzzification
def fuzzification(value, variable):
    value = float(value)  # Conversion de la valeur en flottant
    if variable == "Pregnancies":
        return "normal" if value <= 17 else "high"
    elif variable == "Glucose":
        if value <= 99:
            return "Bas"
        elif 100 <= value <= 125:
            return "Normal"
        else:
            return "Élevé"
    elif variable == "BloodPressure":
        if value <= 79:
            return "Basse"
        elif 80 <= value <= 89:
            return "Normale"
        else:
            return "Élevée"
    elif variable == "SkinThickness":
        if value <= 9:
            return "Faible"
        elif 10 <= value <= 29:
            return "Moyenne"
        else:
            return "Élevée"
    elif variable == "Insulin":
        if value <= 79:
            return "Faible"
        elif 80 <= value <= 159:
            return "Moyenne"
        else:
            return "Élevée"
    elif variable == "BMI":
        if value <= 18.4:
            return "Maigre"
        elif 18.5 <= value <= 24.9:
            return "Normal"
        elif 25 <= value <= 29.9:
            return "Surpoids"
        else:
            return "Obèse"
    elif variable == "DiabetesPedigreeFunction":
        if value <= 0.5:
            return "Faible"
        elif 0.51 <= value <= 1.0:
            return "Moyenne"
        else:
            return "Forte"
    elif variable == "Age":
        if value <= 39:
            return "Jeune"
        elif 40 <= value <= 59:
            return "Moyen"
        else:
            return "Âgé"
    elif variable == "Outcome":
        return "Non-diabétique" if value == 0 else "Diabétique"
    else:
        return "Variable inconnue"


def fuzzification_global(data):
    for column in data.columns:
        data[column] = data[column].apply(lambda x: fuzzification(x, column))
    return data


# défzzification
def defuzzify(fuzzy_value, variable):
    if variable == "Outcome":
        return "Diabétique" if fuzzy_value == 0 else "Non-diabétique"
    #
    else:
        return fuzzy_value


# Fuzzifier les données
fuzzified_data = fuzzification_global(data.copy())

# Encoder les données fuzzifiées
label_encoders = {}
for column in fuzzified_data.columns:
    le = LabelEncoder()
    fuzzified_data[column] = le.fit_transform(fuzzified_data[column])
    label_encoders[column] = le

# Définition des univers de discours pour chaque variable
grossesses = ctrl.Antecedent(np.arange(0, 18, 1), 'grossesses')
glucose = ctrl.Antecedent(np.arange(0, 200, 1), 'glucose')
pression = ctrl.Antecedent(np.arange(0, 130, 1), 'pression')
epaisseur = ctrl.Antecedent(np.arange(0, 100, 1), 'epaisseur')
insuline = ctrl.Antecedent(np.arange(0, 900, 1), 'insuline')
imc = ctrl.Antecedent(np.arange(0, 50, 1), 'imc')
généalogie = ctrl.Antecedent(np.arange(0, 2, 0.1), 'généalogie')
âge = ctrl.Antecedent(np.arange(20, 100, 1), 'âge')
Outcome = ctrl.Consequent(np.arange(0, 2, 1), 'Outcome')

# Définition des fonctions d'appartenance
grossesses['normale'] = fuzz.trapmf(grossesses.universe, [0, 0, 10, 17])
grossesses['anormale'] = fuzz.trapmf(grossesses.universe, [17, 17, 18, 18])

glucose['bas'] = fuzz.trapmf(glucose.universe, [0, 0, 70, 99])
glucose['normal'] = fuzz.trapmf(glucose.universe, [100, 100, 120, 125])
glucose['élevé'] = fuzz.trapmf(glucose.universe, [126, 126, 200, 200])

pression['basse'] = fuzz.trapmf(pression.universe, [0, 0, 60, 79])
pression['normale'] = fuzz.trapmf(pression.universe, [80, 80, 85, 89])
pression['élevée'] = fuzz.trapmf(pression.universe, [90, 90, 130, 130])

epaisseur['Faible'] = fuzz.trapmf(epaisseur.universe, [0, 0, 9, 29])
epaisseur['Moyenne'] = fuzz.trapmf(epaisseur.universe, [10, 10, 30, 50])
epaisseur['Élevée'] = fuzz.trapmf(epaisseur.universe, [30, 30, 100, 100])

insuline['Faible'] = fuzz.trapmf(insuline.universe, [0, 0, 79, 159])
insuline['Moyenne'] = fuzz.trapmf(insuline.universe, [80, 80, 160, 320])
insuline['Élevée'] = fuzz.trapmf(insuline.universe, [160, 160, 900, 900])

imc['Maigre'] = fuzz.trapmf(imc.universe, [0, 0, 18.4, 24.9])
imc['Normal'] = fuzz.trapmf(imc.universe, [18.5, 18.5, 25, 29.9])
imc['Surpoids'] = fuzz.trapmf(imc.universe, [25, 25, 30, 34.9])
imc['Obèse'] = fuzz.trapmf(imc.universe, [30, 30, 50, 50])

généalogie['Faible'] = fuzz.trapmf(généalogie.universe, [0, 0, 0.5, 1])
généalogie['Moyenne'] = fuzz.trapmf(généalogie.universe, [0.51, 0.51, 1, 1.5])
généalogie['Forte'] = fuzz.trapmf(généalogie.universe, [1.01, 1.01, 2, 2])

âge['Jeune'] = fuzz.trapmf(âge.universe, [20, 20, 39, 49])
âge['Moyen'] = fuzz.trapmf(âge.universe, [40, 40, 59, 69])
âge['Âgé'] = fuzz.trapmf(âge.universe, [60, 60, 100, 100])

Outcome['Non-diabétique'] = fuzz.trimf(Outcome.universe, [0, 0, 1])
Outcome['Diabétique'] = fuzz.trimf(Outcome.universe, [0, 1, 1])

# Séparation des données
train_data, test_data = train_test_split(fuzzified_data, test_size=0.2, random_state=42)

# Préparation des caractéristiques et de la cible
X_train = train_data.drop(columns=['Outcome'])

y_train = train_data['Outcome']
X_test = test_data.drop(columns=['Outcome'])
y_test = test_data['Outcome']

X = fuzzified_data.drop(columns=['Outcome'])
y = fuzzified_data['Outcome']

# Utiliser SMOTE pour équilibrer les classes
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Création et entraînement du modèle
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Défluzzification des résultats de test
predicted = model.predict(X_test)
defuzzified_predicted = [defuzzify(value, "Outcome") for value in predicted]

# Evaluation d’éfficacité (Matrice de confusion)
# Calcul des métriques de performance (defuzzified_)
conf_matrix = confusion_matrix(y_test, predicted)
accuracy = accuracy_score(y_test, predicted)
precision = precision_score(y_test, predicted, zero_division=1)
recall = recall_score(y_test, predicted, zero_division=1)
f1 = f1_score(y_test, predicted, zero_division=1)

# Affichage des résultats
print("Matrice de confusion:")
print(conf_matrix)
print(f"Exactitude: {accuracy}")
print(f"Précision: {precision}")
print(f"Sensibilité: {recall}")
print(f"Score F1: {f1}")


# Fonction de prédiction
def predict():

    try:
        # Récupérer les valeurs des champs d'entrée
        input_data = {
            "Pregnancies": pregnancies_entry.get(),
            "Glucose": glucose_entry.get(),
            "BloodPressure": bp_entry.get(),
            "SkinThickness": skin_entry.get(),
            "Insulin": insulin_entry.get(),
            "BMI": bmi_entry.get(),
            "DiabetesPedigreeFunction": dpf_entry.get(),
            "Age": age_entry.get()
        }

        # Convertir les données en format requis
        input_df = pd.DataFrame([input_data])
        fuzzified_input = fuzzification_global(input_df)

        # Encoder les données fuzzifiées
        for column in fuzzified_input.columns:
            le = label_encoders[column]

            fuzzified_input[column] = le.transform(fuzzified_input[column])

        prediction = model.predict(fuzzified_input)
        print(f"prediction ==>: {prediction[0]}")
        # print(f"value ==>: {prediction[0]}")

        result = defuzzify(prediction[0], "Outcome")
        print(f"result ==>: {result}")
    except ValueError as e:
        messagebox.showerror("Erreur d'entrée", f"Veuillez entrer des valeurs valides.\n{e}")

    resultMessage = ""
    print(f" before resultMessage ==>: {resultMessage}")
    if (result == 1):
       resultMessage = "Diabétique"
    else:
       resultMessage = "Non-diabétique"

    print(f"resultMessage ==>: {resultMessage}")
    # Afficher le résultat
    messagebox.showinfo("Résultat de la prédiction", result)


# Création de l'interface
root = tk.Tk()
root.title("Prédiction du diabète")

# Champs d'entrée
fields = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction",
          "Age"]
entries = {}

for field in fields:
    row = tk.Frame(root)
    label = tk.Label(row, width=25, text=field, anchor='w')
    entry = tk.Entry(row)
    row.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
    label.pack(side=tk.LEFT)
    entry.pack(side=tk.RIGHT, expand=tk.YES, fill=tk.X)
    entries[field] = entry

pregnancies_entry = entries["Pregnancies"]
glucose_entry = entries["Glucose"]
bp_entry = entries["BloodPressure"]
skin_entry = entries["SkinThickness"]
insulin_entry = entries["Insulin"]
bmi_entry = entries["BMI"]
dpf_entry = entries["DiabetesPedigreeFunction"]
age_entry = entries["Age"]

# Bouton prédire
predict_button = tk.Button(root, text="Prédire", command=predict)
predict_button.pack(side=tk.BOTTOM, padx=5, pady=5)

root.mainloop()
