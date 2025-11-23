from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from config import Config
from datetime import datetime

class MongoDB:
    """MongoDB handler for storing detection results"""
    
    def __init__(self):
        self.config = Config()
        self.client = None
        self.db = None
        self.collection = None
        self._connect()
    
    def _connect(self):
        """Establish connection to MongoDB"""
        try:
            # Create MongoDB client
            self.client = MongoClient(
                self.config.MONGO_URI,
                serverSelectionTimeoutMS=5000  # 5 second timeout
            )
            
            # Test connection
            self.client.admin.command('ping')
            
            # Get database and collection
            self.db = self.client[self.config.MONGO_DB_NAME]
            self.collection = self.db[self.config.MONGO_COLLECTION_NAME]
            
            # Create indexes for better query performance
            self._create_indexes()
            
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            raise Exception(f"Failed to connect to MongoDB: {e}")
        except Exception as e:
            raise Exception(f"MongoDB initialization error: {e}")
    
    def _create_indexes(self):
        """Create indexes for optimized queries"""
        try:
            # Index on timestamp for time-based queries
            self.collection.create_index([("timestamp", -1)])
            
            # Index on video source
            self.collection.create_index([("video_source", 1)])
            
            # Index on frame number
            self.collection.create_index([("frame_number", 1)])
            
            # Compound index for common queries
            self.collection.create_index([
                ("video_source", 1),
                ("timestamp", -1)
            ])
            
        except Exception as e:
            print(f"Warning: Could not create indexes: {e}")
    
    def insert_detection(self, document):
        """
        Insert a detection document into MongoDB
        
        Args:
            document (dict): Detection data to insert
        
        Returns:
            ObjectId: Inserted document ID
        """
        try:
            result = self.collection.insert_one(document)
            return result.inserted_id
        except Exception as e:
            raise Exception(f"Failed to insert document: {e}")
    
    def insert_many_detections(self, documents):
        """
        Insert multiple detection documents
        
        Args:
            documents (list): List of detection documents
        
        Returns:
            list: List of inserted document IDs
        """
        try:
            result = self.collection.insert_many(documents)
            return result.inserted_ids
        except Exception as e:
            raise Exception(f"Failed to insert documents: {e}")
    
    def get_detections_by_source(self, video_source, limit=100):
        """
        Get detections for a specific video source
        
        Args:
            video_source (str): Video source identifier
            limit (int): Maximum number of documents to return
        
        Returns:
            list: List of detection documents
        """
        try:
            return list(self.collection.find(
                {"video_source": video_source}
            ).sort("timestamp", -1).limit(limit))
        except Exception as e:
            raise Exception(f"Failed to query detections: {e}")
    
    def get_detections_by_time_range(self, start_time, end_time, video_source=None):
        """
        Get detections within a time range
        
        Args:
            start_time (datetime): Start time
            end_time (datetime): End time
            video_source (str): Optional video source filter
        
        Returns:
            list: List of detection documents
        """
        try:
            query = {
                "timestamp": {
                    "$gte": start_time,
                    "$lte": end_time
                }
            }
            
            if video_source:
                query["video_source"] = video_source
            
            return list(self.collection.find(query).sort("timestamp", -1))
        except Exception as e:
            raise Exception(f"Failed to query detections: {e}")
    
    def get_object_statistics(self, video_source=None):
        """
        Get aggregated statistics of detected objects
        
        Args:
            video_source (str): Optional video source filter
        
        Returns:
            list: Aggregated statistics
        """
        try:
            match_stage = {}
            if video_source:
                match_stage = {"$match": {"video_source": video_source}}
            
            pipeline = [
                match_stage,
                {"$unwind": "$detections"},
                {"$group": {
                    "_id": "$detections.class_name",
                    "total_count": {"$sum": 1},
                    "avg_confidence": {"$avg": "$detections.confidence"},
                    "max_confidence": {"$max": "$detections.confidence"},
                    "min_confidence": {"$min": "$detections.confidence"}
                }},
                {"$sort": {"total_count": -1}}
            ]
            
            # Remove empty match stage if no video_source
            if not video_source:
                pipeline = pipeline[1:]
            
            return list(self.collection.aggregate(pipeline))
        except Exception as e:
            raise Exception(f"Failed to get statistics: {e}")
    
    def delete_old_detections(self, days=7):
        """
        Delete detections older than specified days
        
        Args:
            days (int): Number of days to keep
        
        Returns:
            int: Number of deleted documents
        """
        try:
            from datetime import timedelta
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            result = self.collection.delete_many({
                "timestamp": {"$lt": cutoff_date}
            })
            
            return result.deleted_count
        except Exception as e:
            raise Exception(f"Failed to delete old detections: {e}")
    
    def get_collection_stats(self):
        """
        Get collection statistics
        
        Returns:
            dict: Collection statistics
        """
        try:
            stats = {
                'total_documents': self.collection.count_documents({}),
                'database_name': self.config.MONGO_DB_NAME,
                'collection_name': self.config.MONGO_COLLECTION_NAME,
                'indexes': self.collection.index_information()
            }
            return stats
        except Exception as e:
            raise Exception(f"Failed to get collection stats: {e}")
    
    def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()