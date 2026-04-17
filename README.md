# ForeSight
An AI Web-Dashboard for Market Trend Prediction and Inventory Intelligence

# How to run the app locally





# How to add your own api key

Step 1: Open the "api_key.ipynb" notebook. 


Step 2: Add the following code in a new cell.

with.open(".env.example", "w") as f:
  f.write("OPENAI_API_KEY=your_api_key_here\n")
print(".env.example created")

Step 3: Configurate the ".env.example" with ".env"

Step 4: Replace "your_api_key_here" with your actual API key

Step 5: Save and restart the dashboard.



