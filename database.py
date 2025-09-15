import pymysql
from datetime import datetime


class ALPRDatabase:
    def __init__(self, host="localhost", user="root", password="", database="alpr_db"):
        self.db = pymysql.connect(
            host=host,
            user=user,
            password=password,
            database=database,
            cursorclass=pymysql.cursors.DictCursor
        )
        self.cursor = self.db.cursor()
        self.create_table()

    def create_table(self):
        query = """
        CREATE TABLE IF NOT EXISTS plates (
            id INT AUTO_INCREMENT PRIMARY KEY,
            plate_number VARCHAR(20) NOT NULL,
            owner_name VARCHAR(100),
            date_time DATETIME NOT NULL
        )
        """
        self.cursor.execute(query)
        self.db.commit()

    def insert_plate(self, plate_number, owner_name=None):
        """Insert recognized plate into database."""
        now = datetime.now()
        query = "INSERT INTO plates (plate_number, owner_name, date_time) VALUES (%s, %s, %s)"
        self.cursor.execute(query, (plate_number, owner_name, now))
        self.db.commit()
        print(f"[DB] Inserted {plate_number} at {now}")

    def fetch_all(self):
        """Fetch all stored plate records."""
        query = "SELECT * FROM plates ORDER BY date_time DESC"
        self.cursor.execute(query)
        return self.cursor.fetchall()

    def close(self):
        """Close database connection."""
        self.db.close()


# Example usage
if __name__ == "__main__":
    db = ALPRDatabase(host="localhost", user="root", password="yourpassword", database="alpr_db")

    # Insert sample record
    db.insert_plate("AP29BM1234", "Kedar")

    # Fetch all records
    records = db.fetch_all()
    for row in records:
        print(row)

    db.close()
