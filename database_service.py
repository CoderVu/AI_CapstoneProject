from typing import List, Dict, Optional
import logging
from database_config import db_config

logger = logging.getLogger(__name__)

class DatabaseService:
    """Service for database operations"""
    
    def __init__(self):
        self.db = db_config
    
    def get_all_images(self) -> Optional[List[Dict]]:
        """Get all images from database"""
        try:
            query = """
                SELECT 
                    i.id,
                    i.url,
                    i.vector_features,
                    i.product_id,
                    i.color,
                    p.product_name,
                    p.id as product_id
                FROM image i
                LEFT JOIN product p ON i.product_id = p.id
                WHERE i.url IS NOT NULL
                ORDER BY i.id
            """
            results = self.db.execute_query(query)
            if results:
                # Convert to expected format
                images = []
                for row in results:
                    image_data = {
                        'id': row['id'],
                        'url': row['url'],
                        'path': row['url'],  # Use url as path for compatibility
                        'vectorFeatures': row['vector_features'],
                        'productId': row['product_id'],
                        'productName': row['product_name'],
                        'color': row['color']
                    }
                    images.append(image_data)
                logger.info(f"Retrieved {len(images)} images from database")
                return images
            else:
                logger.warning("No images found in database")
                return []
        except Exception as e:
            logger.error(f"Error getting images from database: {e}")
            return None
    
    def get_images_by_product(self, product_id: str) -> Optional[List[Dict]]:
        """Get all images for a specific product"""
        try:
            query = """
                SELECT 
                    i.id,
                    i.url,
                    i.vector_features,
                    i.product_id,
                    i.color,
                    p.product_name
                FROM image i
                LEFT JOIN product p ON i.product_id = p.id
                WHERE i.product_id = %s AND i.url IS NOT NULL
                ORDER BY i.id
            """
            results = self.db.execute_query(query, (product_id,))
            if results:
                images = []
                for row in results:
                    image_data = {
                        'id': row['id'],
                        'url': row['url'],
                        'path': row['url'],
                        'vectorFeatures': row['vector_features'],
                        'productId': row['product_id'],
                        'productName': row['product_name'],
                        'color': row['color']
                    }
                    images.append(image_data)
                return images
            else:
                return []
        except Exception as e:
            logger.error(f"Error getting images for product {product_id}: {e}")
            return None
    
    def get_image_by_id(self, image_id: str) -> Optional[Dict]:
        """Get image by ID"""
        try:
            query = """
                SELECT 
                    i.id,
                    i.url,
                    i.vector_features,
                    i.product_id,
                    i.color,
                    p.product_name
                FROM image i
                LEFT JOIN product p ON i.product_id = p.id
                WHERE i.id = %s
            """
            results = self.db.execute_query(query, (image_id,))
            if results:
                row = results[0]
                # Handle null values properly
                vector_features = row['vector_features'] if row['vector_features'] is not None else None
                
                return {
                    'id': row['id'],
                    'url': row['url'],
                    'path': row['url'],
                    'vectorFeatures': vector_features,
                    'productId': row['product_id'],
                    'productName': row['product_name'],
                    'color': row['color']
                }
            return None
        except Exception as e:
            logger.error(f"Error getting image {image_id}: {e}")
            import traceback
            logger.error(f"Stack trace: {traceback.format_exc()}")
            return None
    
    def update_vector_features(self, image_id: str, vector_features: str) -> bool:
        """Update vector features for an image"""
        try:
            # Validate input
            if not image_id or not vector_features:
                logger.error(f"Invalid input: image_id={image_id}, vector_features length={len(vector_features) if vector_features else 0}")
                return False
            
            query = """
                UPDATE image 
                SET vector_features = %s 
                WHERE id = %s
            """
            affected_rows = self.db.execute_update(query, (vector_features, image_id))
            if affected_rows > 0:
                logger.info(f"Successfully updated vector features for image {image_id}")
                return True
            else:
                logger.warning(f"No rows updated for image {image_id} - image may not exist")
                return False
        except Exception as e:
            logger.error(f"Error updating vector features for image {image_id}: {e}")
            import traceback
            logger.error(f"Stack trace: {traceback.format_exc()}")
            return False
    
    def get_products_with_stats(self) -> Optional[List[Dict]]:
        """Get all products with image statistics and detailed image information"""
        try:
            # First get product statistics
            stats_query = """
                SELECT 
                    p.id,
                    p.product_name,
                    COUNT(i.id) as total_images,
                    COUNT(CASE WHEN i.vector_features IS NOT NULL AND i.vector_features != '' THEN 1 END) as with_features,
                    COUNT(CASE WHEN i.vector_features IS NULL OR i.vector_features = '' THEN 1 END) as without_features
                FROM product p
                LEFT JOIN image i ON p.id = i.product_id
                WHERE i.url IS NOT NULL
                GROUP BY p.id, p.product_name
                ORDER BY total_images DESC
            """
            stats_results = self.db.execute_query(stats_query)
            
            if not stats_results:
                return []
            
            products = []
            for row in stats_results:
                total_images = row['total_images'] or 0
                with_features = row['with_features'] or 0
                without_features = row['without_features'] or 0
                product_id = row['id']
                
                # Get detailed image information for this product
                images_query = """
                    SELECT 
                        i.id,
                        i.url,
                        i.vector_features,
                        i.color
                    FROM image i
                    WHERE i.product_id = %s AND i.url IS NOT NULL
                    ORDER BY i.id
                """
                images_results = self.db.execute_query(images_query, (product_id,))
                
                # Format image data
                images = []
                for img_row in images_results:
                    image_data = {
                        'id': img_row['id'],
                        'url': img_row['url'],
                        'path': img_row['url'],  # Use url as path
                        'has_features': bool(img_row['vector_features'] and img_row['vector_features'].strip()),
                        'vector_features': img_row['vector_features'],
                        'color': img_row['color']
                    }
                    images.append(image_data)
                
                product_data = {
                    'id': product_id,
                    'productName': row['product_name'],
                    'total_images': total_images,
                    'with_features': with_features,
                    'without_features': without_features,
                    'completion_percentage': (with_features / total_images * 100) if total_images > 0 else 0,
                    'images': images  # Include detailed image information
                }
                products.append(product_data)
            
            return products
            
        except Exception as e:
            logger.error(f"Error getting products with stats: {e}")
            return None
    
    def get_images_without_features(self) -> Optional[List[Dict]]:
        """Get images that don't have vector features"""
        try:
            query = """
                SELECT 
                    i.id,
                    i.url,
                    i.vector_features,
                    i.product_id,
                    i.color,
                    p.product_name
                FROM image i
                LEFT JOIN product p ON i.product_id = p.id
                WHERE (i.vector_features IS NULL OR i.vector_features = '') 
                AND i.url IS NOT NULL
                ORDER BY i.id
            """
            results = self.db.execute_query(query)
            if results:
                images = []
                for row in results:
                    image_data = {
                        'id': row['id'],
                        'url': row['url'],
                        'path': row['url'],
                        'vectorFeatures': row['vector_features'],
                        'productId': row['product_id'],
                        'productName': row['product_name'],
                        'color': row['color']
                    }
                    images.append(image_data)
                return images
            else:
                return []
        except Exception as e:
            logger.error(f"Error getting images without features: {e}")
            return None
    
    def get_extraction_stats(self) -> Dict:
        """Get overall extraction statistics"""
        try:
            # Get total images
            total_query = """
                SELECT COUNT(*) as total
                FROM image 
                WHERE url IS NOT NULL
            """
            total_result = self.db.execute_query(total_query)
            total_images = total_result[0]['total'] if total_result else 0
            
            # Get images with features
            with_features_query = """
                SELECT COUNT(*) as count
                FROM image 
                WHERE vector_features IS NOT NULL 
                AND vector_features != '' 
                AND url IS NOT NULL
            """
            with_features_result = self.db.execute_query(with_features_query)
            images_with_features = with_features_result[0]['count'] if with_features_result else 0
            
            # Get product statistics
            product_stats_query = """
                SELECT 
                    p.id,
                    p.product_name,
                    COUNT(i.id) as total,
                    COUNT(CASE WHEN i.vector_features IS NOT NULL AND i.vector_features != '' THEN 1 END) as with_features
                FROM product p
                LEFT JOIN image i ON p.id = i.product_id
                WHERE i.url IS NOT NULL
                GROUP BY p.id, p.product_name
            """
            product_results = self.db.execute_query(product_stats_query)
            
            product_stats = {}
            for row in product_results:
                product_id = row['id']
                product_stats[product_id] = {
                    'total': row['total'],
                    'with_features': row['with_features']
                }
            
            return {
                'total_images': total_images,
                'images_with_features': images_with_features,
                'images_without_features': total_images - images_with_features,
                'completion_percentage': (images_with_features / total_images * 100) if total_images > 0 else 0,
                'product_stats': product_stats
            }
        except Exception as e:
            logger.error(f"Error getting extraction stats: {e}")
            return {
                'error': str(e),
                'total_images': 0,
                'images_with_features': 0,
                'images_without_features': 0,
                'completion_percentage': 0,
                'product_stats': {}
            }
    
    def health_check(self) -> bool:
        """Check if database is accessible with detailed status"""
        try:
            # Test basic connection
            query = "SELECT 1 as test"
            result = self.db.execute_query(query)
            
            if result is None:
                logger.error("Database health check failed: No result from test query")
                return False
            
            # Test if we can access the image table
            image_query = "SELECT COUNT(*) as count FROM image LIMIT 1"
            image_result = self.db.execute_query(image_query)
            
            if image_result is None:
                logger.error("Database health check failed: Cannot access image table")
                return False
            
            logger.info("Database health check passed")
            return True
            
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False

# Global database service instance
database_service = DatabaseService() 