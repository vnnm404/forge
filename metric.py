import pandas as pd
import glob

# Folder containing the CSV files
folder_path = "metrics"

# Get a list of all CSV files in the folder
csv_files = glob.glob(folder_path + "/*.csv")

# Initialize a list to store the percentage differences for each file
percentage_differences = []

final_data = []

# Iterate over each CSV file
for file in csv_files:
    print(file)
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file)
    
    # Extract the Jaccard scores
    jaccard_scores = df[df['Metric'] == 'jaccard']
    
    # print(jaccard_scores)
    
    # Identify the best performing propagation method based on Jaccard score ignoring the "graph" method
    best_method = jaccard_scores.loc[jaccard_scores[jaccard_scores['Propagation Method'] != 'graph']['Average Score'].idxmax()]

    best_score = best_method['Average Score']
    
    # Extract the Jaccard score for the "graph" method
    graph_score = jaccard_scores[jaccard_scores['Propagation Method'] == 'graph']['Average Score'].values[0]
    
    print(f"best method: {best_method['Propagation Method']}, best score: {best_score}, graph score: {graph_score}")
    
    final_data.append({
        'File': "_".join(file.split("/")[-1].split(".")[0].split("_")[1:-1]),
        'Normal': graph_score,
        'HOGE': best_score,
    })
    
    # Calculate the percentage difference
    # best_score = best_method['Average Score']
    # percentage_difference = ((best_score - graph_score) / graph_score) * 100
    
    # # Append the percentage difference to the list
    # percentage_differences.append(percentage_difference)

# save final data
df = pd.DataFrame(final_data)
df.to_csv('summary.csv', index=False)

# print df sorted by dataset
df = df.sort_values(by=['File'])

# save final data
df.to_csv('summary.csv', index=False)

# Calculate the average percentage
# increase/decrease
# average_percentage_difference = sum(percentage_differences) / len(percentage_differences)

# # Report the average percentage increase/decrease
# print(f"Average % increase/decrease over 'graph' propagation method Jaccard score: {average_percentage_difference:.2f}%")
