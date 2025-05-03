import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data_folder = r"C:\Users\manya\Downloads\archive (2)\SMNI_CMI_TEST"

dic = []
for file in os.listdir(data_folder):
    if file.endswith(".csv"):
        file_path = os.path.join(data_folder, file)
        try:
            df = pd.read_csv(file_path)
            dic.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")

if not dic:
    raise ValueError("No valid data found!")

combined_data = pd.concat(dic, ignore_index=True)

combined_data['sensor value'] = pd.to_numeric(combined_data['sensor value'], errors='coerce')
combined_data.dropna(subset=['sensor value'], inplace=True)

alcoholic_subjects = ["a"]

combined_data['is_alcoholic'] = combined_data['subject identifier'].apply(
    lambda x: 1 if x in alcoholic_subjects else 0
)

average_sensor_values = combined_data.groupby(['sensor position', 'is_alcoholic'])['sensor value'].mean().reset_index()

thresholds = (
    average_sensor_values.groupby('sensor position')['sensor value']
    .mean()
    .reset_index(name='threshold')
)

average_sensor_values = average_sensor_values.merge(thresholds, on='sensor position')

X = average_sensor_values[['sensor value']]  
y = average_sensor_values['is_alcoholic']   

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
# print(f"Model Accuracy: {accuracy:.4f}")

def predict(sensor_values, threshold_dict):
    eye_states = {}
    for electrode, value in sensor_values.items():
        threshold = threshold_dict.get(electrode, None)
        if threshold is None:
            eye_states[electrode] = "Unknown (No threshold data)"
        else:
            eye_state = "Open" if value > threshold else "Closed"
            eye_states[electrode] = eye_state

    closed_count = sum(1 for state in eye_states.values() if state == "Closed")
    open_count = sum(1 for state in eye_states.values() if state == "Open")
    alcoholic_state = "Alcoholic" if closed_count > open_count else "Non-Alcoholic"

    return eye_states, alcoholic_state

threshold_dict = dict(zip(thresholds['sensor position'], thresholds['threshold']))

plt.figure(figsize=(12, 6))

for is_alcoholic, label in zip([0, 1], ["Non-Alcoholic", "Alcoholic"]):
    subset = average_sensor_values[average_sensor_values['is_alcoholic'] == is_alcoholic]
    plt.plot(
        subset['sensor position'],
        subset['sensor value'],
        marker='o',
        label=f'{label} (Average)',
    )

plt.plot(
    thresholds['sensor position'],
    thresholds['threshold'],
    linestyle='--',
    color='red',
    label='Threshold',
)

plt.title('Average Sensor Values by Electrode Position')
plt.xlabel('Sensor Position')
plt.ylabel('Average Sensor Value')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
data_list = []
for file in os.listdir(data_folder):
    if file.endswith(".csv"):
        file_path = os.path.join(data_folder, file)
        df = pd.read_csv(file_path)
        df['subject identifier'] = file.split('.')[0]  
        data_list.append(df)
data = pd.concat(data_list, ignore_index=True)

data['sensor_value'] = pd.to_numeric(data['sensor value'], errors='coerce')
data.dropna(subset=['sensor_value'], inplace=True)

noise_factor = 0.1
data['sensor_value'] += np.random.normal(loc=0.0, scale=noise_factor, size=data['sensor_value'].shape)

alcoholic_ids = ["a"]  
data['is_alcoholic'] = data['subject identifier'].apply(
    lambda x: 1 if x in alcoholic_ids else 0
)

features = data.pivot_table(index='subject identifier', columns='sensor position', values='sensor_value').dropna()
labels = data.groupby('subject identifier')['is_alcoholic'].first()

rf_model = RandomForestClassifier(random_state=42)
# cross_val_scores = cross_val_score(rf_model, features, labels, cv=5)
# print(f"Cross-Validation Accuracy: {np.mean(cross_val_scores):.2f} ± {np.std(cross_val_scores):.2f}")

rf_model.fit(features, labels)

feature_scores = pd.Series(rf_model.feature_importances_, index=features.columns).sort_values(ascending=False)
important_sensors = feature_scores.head(5).index.tolist()
print("\nImportant Sensors:")
print(important_sensors)

sensor_avg = data.groupby(['sensor position', 'is_alcoholic'])['sensor_value'].mean().reset_index()
thresholds = sensor_avg[sensor_avg['sensor position'].isin(important_sensors)]
thresholds = thresholds.groupby('sensor position')['sensor_value'].mean().to_dict()

print("\nThresholds for Important Sensors:")
for sensor, threshold in thresholds.items():
    print(f"{sensor}: {threshold:.2f}")

def predict_states(sensor_data, thresholds):
    states = {}
    for sensor, value in sensor_data.items():
        if sensor in thresholds:
            states[sensor] = "Open" if value > thresholds[sensor] else "Closed"

    closed = sum(1 for state in states.values() if state == "Closed")
    open_ = sum(1 for state in states.values() if state == "Open")
    alcoholic = "Alcoholic" if closed > open_ else "Non-Alcoholic"

    return states, alcoholic

averaged_data = []
for file in os.listdir(data_folder):
    if file.endswith(".csv"):
        file_path = os.path.join(data_folder, file)
        df = pd.read_csv(file_path)
        avg_values = {sensor: df[df['sensor position'] == sensor]['sensor value'].mean() for sensor in important_sensors}
        averaged_data.append({'subject identifier': file.split('.')[0], **avg_values}) 
averaged_df = pd.DataFrame(averaged_data)

avg_values_for_prediction = averaged_df[important_sensors].mean().to_dict()
eye_states, alcoholic_status = predict_states(avg_values_for_prediction, thresholds)

print("\nAveraged Values for Important Electrodes:")
for sensor, avg_value in avg_values_for_prediction.items():
    print(f"{sensor}: {avg_value:.2f}")

print("\nPredicted Eye States:")
for sensor, state in eye_states.items():
    print(f"{sensor}: {state}")

print(f"\nPredicted Alcoholic Status: {alcoholic_status}")

plt.figure(figsize=(10, 6))

plt.plot(
    avg_values_for_prediction.keys(),
    avg_values_for_prediction.values(),
    marker='o',
    color='skyblue',
    label='Average Values'
)

for sensor in important_sensors:
    if sensor in thresholds:
        plt.axhline(
            y=thresholds[sensor],
            color='r',
            linestyle='--',
            label=f'Threshold for {sensor}'
        )
    else:
        print(f"Warning: Missing threshold for sensor {sensor}")

plt.title("Average Sensor Values for Important Electrodes", fontsize=16)
plt.xlabel("Sensor Position", fontsize=14)
plt.ylabel("Average Sensor Value", fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show() 