import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load the training data
train_df = pd.read_csv('train.csv')

# Split features and target
X_train = train_df.drop(columns=['id', 'target'])
y_train = train_df['target']

# Train a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Load the test data
test_df = pd.read_csv('test.csv')

# Make predictions
predictions = clf.predict(test_df.drop(columns=['id']))

# Create a submission file
submission = pd.DataFrame({'id': test_df['id'], 'target': predictions})
submission.to_csv('submission.csv', index=False)
