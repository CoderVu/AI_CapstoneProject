import mysql.connector
from mysql.connector import Error
from typing import Optional
import logging
import threading
from config import DB_HOST, DB_PORT, DB_NAME, DB_USERNAME, DB_PASSWORD

logger = logging.getLogger(__name__)

class DatabaseConfig:
    """Database configuration for MySQL connection with improved connection management"""
    
    def __init__(self):
        self.host = DB_HOST
        self.port = DB_PORT
        self.database = DB_NAME
        self.username = DB_USERNAME
        self.password = DB_PASSWORD
        self.connection = None
        self.lock = threading.Lock()
    
    def get_connection(self):
        """Get database connection with improved error handling"""
        with self.lock:
            try:
                # Check if connection exists and is still valid
                if self.connection is not None:
                    try:
                        # Test if connection is still alive
                        self.connection.ping(reconnect=True, attempts=3, delay=0.2)
                        if self.connection.is_connected():
                            return self.connection
                    except Error:
                        logger.warning("Existing connection is invalid, creating new one")
                        self.connection = None
                
                # Create new connection
                self.connection = mysql.connector.connect(
                    host=self.host,
                    port=self.port,
                    database=self.database,
                    user=self.username,
                    password=self.password,
                    charset='utf8mb4',
                    autocommit=True,
                    pool_name='ai_image_pool',
                    pool_size=5,
                    pool_reset_session=True,
                    connection_timeout=30,
                    use_pure=True
                )
                logger.info("Database connection established successfully")
                return self.connection
                
            except Error as e:
                logger.error(f"Error connecting to database: {e}")
                self.connection = None
                return None
    
    def close_connection(self):
        """Close database connection"""
        with self.lock:
            try:
                if self.connection and self.connection.is_connected():
                    self.connection.close()
                    logger.info("Database connection closed")
                    self.connection = None
            except Error as e:
                logger.error(f"Error closing database connection: {e}")
    
    def execute_query(self, query: str, params: tuple = None):
        """Execute a query and return results with improved error handling"""
        try:
            connection = self.get_connection()
            if not connection:
                logger.error("No database connection available")
                return None
            
            cursor = connection.cursor(dictionary=True)
            cursor.execute(query, params or ())
            results = cursor.fetchall()
            cursor.close()
            return results
            
        except Error as e:
            logger.error(f"Error executing query: {e}")
            # Try to reconnect on connection error
            if "MySQL Connection not available" in str(e):
                logger.info("Attempting to reconnect to database...")
                self.connection = None
                connection = self.get_connection()
                if connection:
                    try:
                        cursor = connection.cursor(dictionary=True)
                        cursor.execute(query, params or ())
                        results = cursor.fetchall()
                        cursor.close()
                        return results
                    except Error as retry_error:
                        logger.error(f"Retry query failed: {retry_error}")
            return None
    
    def execute_update(self, query: str, params: tuple = None):
        """Execute an update query and return affected rows with improved error handling"""
        try:
            connection = self.get_connection()
            if not connection:
                logger.error("No database connection available")
                return 0
            
            cursor = connection.cursor()
            cursor.execute(query, params or ())
            affected_rows = cursor.rowcount
            connection.commit()
            cursor.close()
            return affected_rows
            
        except Error as e:
            logger.error(f"Error executing update: {e}")
            # Try to reconnect on connection error
            if "MySQL Connection not available" in str(e):
                logger.info("Attempting to reconnect to database...")
                self.connection = None
                connection = self.get_connection()
                if connection:
                    try:
                        cursor = connection.cursor()
                        cursor.execute(query, params or ())
                        affected_rows = cursor.rowcount
                        connection.commit()
                        cursor.close()
                        return affected_rows
                    except Error as retry_error:
                        logger.error(f"Retry update failed: {retry_error}")
                        if connection:
                            connection.rollback()
            return 0

# Global database config instance
db_config = DatabaseConfig() 