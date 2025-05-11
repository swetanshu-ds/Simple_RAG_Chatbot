import sqlite3

# Connect to the database
conn = sqlite3.connect('complaints.db')
cursor = conn.cursor()

# Execute a query to select all rows from the 'complaints' table
cursor.execute("SELECT * FROM complaints")

# Fetch all results and print them
rows = cursor.fetchall()
for row in rows:
    print(row)
