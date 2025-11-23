from database import MongoDB
from datetime import datetime, timedelta
import json

def print_separator(char="=", length=60):
    print(char * length)

def display_recent_detections(limit=10):
    """Display recent detection records"""
    with MongoDB() as db:
        print_separator()
        print(f"RECENT DETECTIONS (Last {limit})")
        print_separator()
        
        detections = list(db.collection.find().sort("timestamp", -1).limit(limit))
        
        if not detections:
            print("No detections found in database.")
            return
        
        for i, det in enumerate(detections, 1):
            print(f"\n[{i}] Detection ID: {det['_id']}")
            print(f"    Timestamp: {det['timestamp']}")
            print(f"    Video Source: {det['video_source']}")
            print(f"    Frame: {det['frame_number']} (Processed: {det['processed_frame_number']})")
            print(f"    Total Objects: {det['total_objects_detected']}")
            print(f"    Object Counts: {det['object_counts']}")
        
        print_separator()

def display_object_statistics(video_source=None):
    """Display aggregated object statistics"""
    with MongoDB() as db:
        print_separator()
        print("OBJECT DETECTION STATISTICS")
        if video_source:
            print(f"Video Source: {video_source}")
        else:
            print("All Video Sources")
        print_separator()
        
        stats = db.get_object_statistics(video_source)
        
        if not stats:
            print("No statistics available.")
            return
        
        print(f"\n{'Class Name':<20} {'Count':<10} {'Avg Conf':<12} {'Min Conf':<12} {'Max Conf':<12}")
        print("-" * 66)
        
        for stat in stats:
            print(f"{stat['_id']:<20} {stat['total_count']:<10} "
                  f"{stat['avg_confidence']:<12.4f} {stat['min_confidence']:<12.4f} "
                  f"{stat['max_confidence']:<12.4f}")
        
        print_separator()

def display_collection_info():
    """Display collection information"""
    with MongoDB() as db:
        print_separator()
        print("COLLECTION INFORMATION")
        print_separator()
        
        stats = db.get_collection_stats()
        
        print(f"\nDatabase: {stats['database_name']}")
        print(f"Collection: {stats['collection_name']}")
        print(f"Total Documents: {stats['total_documents']}")
        print(f"\nIndexes:")
        for index_name, index_info in stats['indexes'].items():
            print(f"  - {index_name}: {index_info['key']}")
        
        print_separator()

def display_detections_by_time_range(hours=1):
    """Display detections from last N hours"""
    with MongoDB() as db:
        print_separator()
        print(f"DETECTIONS FROM LAST {hours} HOUR(S)")
        print_separator()
        
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        detections = db.get_detections_by_time_range(start_time, end_time)
        
        if not detections:
            print(f"No detections found in the last {hours} hour(s).")
            return
        
        print(f"\nTotal Detections: {len(detections)}")
        
        # Aggregate counts
        total_objects = {}
        for det in detections:
            for cls_name, count in det['object_counts'].items():
                total_objects[cls_name] = total_objects.get(cls_name, 0) + count
        
        print(f"\nAggregated Object Counts:")
        for cls_name, count in sorted(total_objects.items(), key=lambda x: x[1], reverse=True):
            print(f"  {cls_name}: {count}")
        
        print_separator()

def export_to_json(output_file='detections_export.json', limit=100):
    """Export detections to JSON file"""
    with MongoDB() as db:
        print_separator()
        print(f"EXPORTING DETECTIONS TO JSON")
        print_separator()
        
        detections = list(db.collection.find().sort("timestamp", -1).limit(limit))
        
        if not detections:
            print("No detections to export.")
            return
        
        # Convert ObjectId to string for JSON serialization
        for det in detections:
            det['_id'] = str(det['_id'])
            det['timestamp'] = det['timestamp'].isoformat()
        
        with open(output_file, 'w') as f:
            json.dump(detections, f, indent=2)
        
        print(f"✓ Exported {len(detections)} detections to {output_file}")
        print_separator()

def cleanup_old_data(days=7):
    """Delete old detection records"""
    with MongoDB() as db:
        print_separator()
        print(f"CLEANING UP DATA OLDER THAN {days} DAYS")
        print_separator()
        
        deleted_count = db.delete_old_detections(days)
        print(f"✓ Deleted {deleted_count} old detection records")
        print_separator()

def main():
    """Main menu for database queries"""
    print("\n" + "="*60)
    print("YOLO Detection Database Query Tool")
    print("="*60)
    
    while True:
        print("\nSelect an option:")
        print("1. View recent detections")
        print("2. View object statistics")
        print("3. View detections from last hour")
        print("4. View collection information")
        print("5. Export detections to JSON")
        print("6. Cleanup old data")
        print("7. Exit")
        
        choice = input("\nEnter choice (1-7): ").strip()
        
        try:
            if choice == '1':
                limit = input("Number of records to display (default 10): ").strip()
                limit = int(limit) if limit else 10
                display_recent_detections(limit)
            
            elif choice == '2':
                video_source = input("Filter by video source (press Enter for all): ").strip()
                display_object_statistics(video_source if video_source else None)
            
            elif choice == '3':
                hours = input("Number of hours to look back (default 1): ").strip()
                hours = int(hours) if hours else 1
                display_detections_by_time_range(hours)
            
            elif choice == '4':
                display_collection_info()
            
            elif choice == '5':
                output_file = input("Output filename (default 'detections_export.json'): ").strip()
                output_file = output_file if output_file else 'detections_export.json'
                limit = input("Number of records to export (default 100): ").strip()
                limit = int(limit) if limit else 100
                export_to_json(output_file, limit)
            
            elif choice == '6':
                days = input("Delete data older than N days (default 7): ").strip()
                days = int(days) if days else 7
                confirm = input(f"Are you sure you want to delete data older than {days} days? (yes/no): ").strip().lower()
                if confirm == 'yes':
                    cleanup_old_data(days)
                else:
                    print("Cleanup cancelled.")
            
            elif choice == '7':
                print("\nExiting...")
                break
            
            else:
                print("Invalid choice. Please select 1-7.")
        
        except Exception as e:
            print(f"\n✗ Error: {e}")

if __name__ == "__main__":
    main()