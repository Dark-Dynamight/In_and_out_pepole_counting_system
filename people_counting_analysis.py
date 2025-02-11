import pandas as pd
import matplotlib.pyplot as plt

# Load the Excel file
file_path = r'C:\Users\user\Desktop\Mini Project\people_counting_system\people_counting_analysis.xlsx'

# Read the data from the Excel file
data = pd.read_excel(file_path)

# Display the first few rows of the dataframe and the column names
print(data.head())
print("Column names:", data.columns.tolist())  # Print the column names

# Extracting the relevant columns using the correct names
try:
    frame_number = data['Frame Number']  # Corrected column name
    up_count = data['Up Count']          # Corrected column name
    down_count = data['Down Count']      # Corrected column name
except KeyError as e:
    print(f"KeyError: {e}. Please check the column names.")

# Plotting the data if columns are found
if 'Frame Number' in data.columns and 'Up Count' in data.columns and 'Down Count' in data.columns:
    plt.figure(figsize=(12, 6))
    plt.plot(frame_number, up_count, label='Up Count', color='blue', marker='o')
    plt.plot(frame_number, down_count, label='Down Count', color='red', marker='x')

    # Adding titles and labels
    plt.title('Up Count and Down Count Over Frames')
    plt.xlabel('Frame Number')
    plt.ylabel('Count')
    plt.legend()
    plt.grid()

    # Show the plot
    plt.show()
