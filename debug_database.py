#!/usr/bin/env python3
"""
Database Diagnostic Script
Check what's in your content_chunks table
"""

import os
from dotenv import load_dotenv
from supabase import create_client

# Load environment variables
load_dotenv()

def check_database():
    """Check the content_chunks table"""
    try:
        # Initialize Supabase client
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_SERVICE_KEY')
        
        if not supabase_url or not supabase_key:
            print("❌ Environment variables not set")
            return
            
        supabase = create_client(supabase_url, supabase_key)
        
        print("🔍 Checking your content_chunks table...")
        print("=" * 50)
        
        # Check total rows and embeddings
        print("1. Checking row counts:")
        result = supabase.table('content_chunks').select('id, embedding').execute()
        total_rows = len(result.data)
        rows_with_embeddings = len([row for row in result.data if row.get('embedding')])
        
        print(f"   📊 Total rows: {total_rows}")
        print(f"   🧠 Rows with embeddings: {rows_with_embeddings}")
        
        if total_rows == 0:
            print("❌ No data in content_chunks table!")
            return
            
        if rows_with_embeddings == 0:
            print("❌ No embeddings found! You need to generate embeddings first.")
            return
            
        # Show sample content
        print("\n2. Sample content:")
        sample_result = supabase.table('content_chunks').select('content, context, video_id').limit(3).execute()
        for i, row in enumerate(sample_result.data, 1):
            content = row.get('content', '')[:100]
            video_id = row.get('video_id', 'N/A')
            context = row.get('context', 'N/A')
            print(f"   Row {i}:")
            print(f"     Video: {video_id}")
            print(f"     Content: {content}...")
            print(f"     Context: {context}")
            print()
            
        # Test the search function
        print("3. Testing search function:")
        try:
            # Create a simple test embedding (all zeros for testing)
            test_embedding = [0.0] * 1536
            
            search_result = supabase.rpc('match_content_chunks', {
                'query_embedding': test_embedding,
                'match_threshold': 0.0,  # Very low threshold
                'match_count': 5
            }).execute()
            
            print(f"   ✅ Search function works! Found {len(search_result.data)} results with test embedding")
            
            # Show what topics are actually in your database
            print("\n4. Available topics in your database:")
            all_content = supabase.table('content_chunks').select('content').execute()
            
            # Extract keywords from content
            all_text = ' '.join([row.get('content', '') for row in all_content.data])
            words = all_text.lower().split()
            
            # Count common words (excluding common English words)
            exclude_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
            
            word_count = {}
            for word in words:
                clean_word = word.strip('.,!?;:"()[]{}').lower()
                if len(clean_word) > 3 and clean_word not in exclude_words:
                    word_count[clean_word] = word_count.get(clean_word, 0) + 1
            
            # Show top 10 words
            top_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)[:10]
            print("   Top words in your content:")
            for word, count in top_words:
                print(f"     • {word} (appears {count} times)")
                
        except Exception as e:
            print(f"   ❌ Search function error: {str(e)}")
            print("   💡 You may need to run the database setup SQL first")
            
    except Exception as e:
        print(f"❌ Database connection error: {str(e)}")

if __name__ == "__main__":
    check_database()