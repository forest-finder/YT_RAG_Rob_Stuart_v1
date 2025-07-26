#!/usr/bin/env python3
"""
Test the search functionality step by step
"""

import os
from dotenv import load_dotenv
from supabase import create_client
import openai

# Load environment variables
load_dotenv()

def test_search():
    try:
        # Initialize clients
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_SERVICE_KEY')
        openai_key = os.getenv('OPENAI_API_KEY')
        
        supabase = create_client(supabase_url, supabase_key)
        openai_client = openai.OpenAI(api_key=openai_key)
        
        print("🔍 Testing search functionality...")
        print("=" * 50)
        
        # Step 1: Test if function exists
        print("1. Checking if match_content_chunks function exists...")
        try:
            # Try to call the function with dummy data
            test_embedding = [0.0] * 1536
            result = supabase.rpc('match_content_chunks', {
                'query_embedding': test_embedding,
                'match_threshold': 0.0,
                'match_count': 1
            }).execute()
            print("   ✅ Function exists and can be called")
        except Exception as e:
            print(f"   ❌ Function error: {str(e)}")
            print("   💡 You need to run the database setup SQL first!")
            return
        
        # Step 2: Test embedding generation
        print("\n2. Testing embedding generation...")
        try:
            response = openai_client.embeddings.create(
                model="text-embedding-3-small",
                input="biomarkers"
            )
            query_embedding = response.data[0].embedding
            print(f"   ✅ Generated embedding for 'biomarkers' (length: {len(query_embedding)})")
        except Exception as e:
            print(f"   ❌ Embedding error: {str(e)}")
            return
        
        # Step 3: Test actual search
        print("\n3. Testing search with real embedding...")
        try:
            result = supabase.rpc('match_content_chunks', {
                'query_embedding': query_embedding,
                'match_threshold': 0.5,
                'match_count': 5
            }).execute()
            
            print(f"   📊 Search returned {len(result.data)} results")
            
            if len(result.data) > 0:
                print("   ✅ Search is working! Top result:")
                top_result = result.data[0]
                print(f"     Content: {top_result.get('content', '')[:100]}...")
                print(f"     Similarity: {top_result.get('similarity', 0):.3f}")
                print(f"     Video: {top_result.get('video_id', 'N/A')}")
            else:
                print("   ⚠️  No results found. Let's try with lower threshold...")
                
                # Try with very low threshold
                result2 = supabase.rpc('match_content_chunks', {
                    'query_embedding': query_embedding,
                    'match_threshold': 0.0,
                    'match_count': 5
                }).execute()
                
                print(f"   📊 With threshold 0.0: {len(result2.data)} results")
                
                if len(result2.data) > 0:
                    print("   💡 Results found with lower threshold!")
                    top_result = result2.data[0]
                    print(f"     Content: {top_result.get('content', '')[:100]}...")
                    print(f"     Similarity: {top_result.get('similarity', 0):.3f}")
                else:
                    print("   ❌ Still no results. There may be an issue with embeddings.")
        
        except Exception as e:
            print(f"   ❌ Search error: {str(e)}")
        
        print("\n" + "=" * 50)
        print("🔍 Diagnosis complete!")
        
    except Exception as e:
        print(f"❌ Setup error: {str(e)}")

if __name__ == "__main__":
    test_search()