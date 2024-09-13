# Step 1: Import the required libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Step 2: Load the Excel file
# Replace 'iris_dataset.xlsx' with the path to your file
df = pd.read_excel('iris_dataset.xlsx')

# Step 3: Select the first 4 rows and the relevant columns (excluding non-numerical columns if any)
# Replace 'column1', 'column2', etc., with actual column names in your dataset
data = df.iloc[:4, :].select_dtypes(include=[float, int])

# Step 4: Standardize the features
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Step 5: Perform PCA with 2 components
pca = PCA(n_components=2)
principal_components = pca.fit_transform(data_scaled)

# Step 6: Create a DataFrame for the results
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

# Step 7: Visualize the PCA result
plt.scatter(pca_df['PC1'], pca_df['PC2'], color='blue')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of First 4 Rows')
plt.show()
