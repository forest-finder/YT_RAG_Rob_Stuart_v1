#!/usr/bin/env python3
"""
Terminal RAG System with Memory
A command-line interface for retrieving and answering questions from your database
with conversation memory capabilities.
"""

import os
import sys
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json

try:
    from pydantic_ai import Agent
    from pydantic_ai.models.openai import OpenAIModel
    from supabase import create_client, Client
    from dotenv import load_dotenv
except ImportError as e:
    print(f"❌ Missing required package: {e}")
    print("Please run: pip install pydantic-ai supabase python-dotenv")
    sys.exit(1)

@dataclass
class ConversationMessage:
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime
    
    def to_dict(self) -> Dict:
        return {
            'role': self.role,
            'content': self.content,
            'timestamp': self.timestamp.isoformat()
        }

class TerminalRAGSystem:
    def __init__(self, max_history: int = 10):
        """Initialize the Terminal RAG System with conversation memory."""
        # Load environment variables
        load_dotenv()
        
        # Configuration
        self.max_history = max_history
        self.conversation_history: List[ConversationMessage] = []
        
        # Initialize clients
        self._initialize_clients()
        
        # Search configuration
        self.similarity_threshold = 0.5  # Lowered for better results
        self.max_results = 8
        self.embedding_model = "text-embedding-3-small"
        self.chat_model = "gpt-3.5-turbo"
        
    def _initialize_clients(self):
        """Initialize OpenAI and Supabase clients."""
        # OpenAI configuration
        openai_key = os.getenv('OPENAI_API_KEY')
        if not openai_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        self.embedding_model_client = OpenAIModel('text-embedding-3-small')
        self.chat_model_client = OpenAIModel('gpt-3.5-turbo')
        self.chat_agent = Agent(self.chat_model_client)
        
        # Supabase configuration
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_SERVICE_KEY')
        
        if not supabase_url or not supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY environment variables are required")
        
        self.supabase: Client = create_client(supabase_url, supabase_key)
    
    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for the given text."""
        try:
            # PydanticAI doesn't have a direct embedding method, so we'll use the underlying OpenAI client
            import openai
            openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            response = openai_client.embeddings.create(
                model=self.embedding_model,
                input=text.replace("\n", " ")
            )
            return response.data[0].embedding
        except Exception as e:
            raise Exception(f"Error generating embedding: {str(e)}")
    
    def keyword_search(self, query: str, limit: int = 5) -> List[Dict]:
        """Search for documents using keyword matching with video metadata."""
        try:
            # Create a text search query using Supabase's text search
            # Join with video_metadata to get video information
            result = self.supabase.table('content_chunks').select(
                '*, video_metadata(video_url, title, channel_name, channel_handle, view_count, likes, keywords, is_shorts, comments)'
            ).text_search(
                'content', 
                query,
                type='websearch'
            ).limit(limit).execute()
            
            return result.data
            
        except Exception as e:
            print(f"❌ Error in keyword search: {str(e)}")
            return []
    
    def hybrid_search(self, query: str, limit: int = 5) -> List[Dict]:
        """Combine vector similarity AND keyword matching for better results."""
        try:
            print(f"🔍 Hybrid searching for: {query}")
            
            # Get vector search results
            vector_results = self.vector_search(query, limit)
            
            # Get keyword search results  
            keyword_results = self.keyword_search(query, limit)
            
            # Combine and rank results
            combined_results = self.merge_and_rank_results(vector_results, keyword_results)
            
            print(f"✅ Found {len(combined_results)} relevant documents (hybrid)")
            return combined_results[:self.max_results]
            
        except Exception as e:
            print(f"❌ Error in hybrid search: {str(e)}")
            return []
    
    def vector_search(self, query: str, limit: int = 5) -> List[Dict]:
        """Search for documents using vector similarity with video metadata."""
        try:
            # Generate query embedding
            query_embedding = self.get_embedding(query)
            
            # Search using the stored function and join with video metadata
            result = self.supabase.rpc('match_content_chunks_with_metadata', {
                'query_embedding': query_embedding,
                'match_threshold': self.similarity_threshold,
                'match_count': limit
            }).execute()
            
            return result.data
            
        except Exception as e:
            print(f"❌ Error in vector search: {str(e)}")
            return []
    
    def merge_and_rank_results(self, vector_results: List[Dict], keyword_results: List[Dict]) -> List[Dict]:
        """Merge vector and keyword results, removing duplicates and ranking by relevance."""
        # Create a dictionary to store unique results by content chunk ID
        merged = {}
        
        # Add vector results with vector score
        for doc in vector_results:
            doc_id = doc.get('id') or doc.get('content', '')[:50]  # Use ID or content snippet as key
            doc['vector_score'] = doc.get('similarity', 0.0)
            doc['keyword_score'] = 0.0
            doc['search_source'] = 'vector'
            merged[doc_id] = doc
        
        # Add keyword results, combining scores if already exists
        for doc in keyword_results:
            doc_id = doc.get('id') or doc.get('content', '')[:50]
            keyword_score = 0.8  # Default keyword match score
            
            if doc_id in merged:
                # Combine scores for documents found in both searches
                merged[doc_id]['keyword_score'] = keyword_score
                merged[doc_id]['search_source'] = 'both'
                # Boost combined score
                merged[doc_id]['combined_score'] = (
                    merged[doc_id]['vector_score'] * 0.6 + 
                    keyword_score * 0.4 + 
                    0.2  # Bonus for appearing in both
                )
            else:
                # Add new keyword-only result
                doc['vector_score'] = 0.0
                doc['keyword_score'] = keyword_score
                doc['search_source'] = 'keyword'
                doc['combined_score'] = keyword_score * 0.8
                merged[doc_id] = doc
        
        # Convert back to list and sort by combined score
        results = list(merged.values())
        
        # Add combined score for vector-only results
        for doc in results:
            if 'combined_score' not in doc:
                doc['combined_score'] = doc['vector_score']
        
        # Sort by combined score (highest first)
        results.sort(key=lambda x: x.get('combined_score', 0), reverse=True)
        
        return results

    def search_documents(self, query: str) -> List[Dict]:
        """Search for relevant documents using hybrid search (vector + keyword)."""
        return self.hybrid_search(query, self.max_results)
    
    def add_to_conversation(self, role: str, content: str):
        """Add a message to conversation history."""
        message = ConversationMessage(
            role=role,
            content=content,
            timestamp=datetime.now()
        )
        self.conversation_history.append(message)
        
        # Keep only the last max_history messages
        if len(self.conversation_history) > self.max_history * 2:  # *2 for user+assistant pairs
            self.conversation_history = self.conversation_history[-(self.max_history * 2):]
    
    def format_conversation_context(self) -> str:
        """Format conversation history for context."""
        if not self.conversation_history:
            return ""
        
        context_lines = ["Previous conversation:"]
        for msg in self.conversation_history[-6:]:  # Last 6 messages
            role_emoji = "👤" if msg.role == "user" else "🤖"
            context_lines.append(f"{role_emoji} {msg.content}")
        
        return "\n".join(context_lines) + "\n\n"
    
    def generate_response(self, query: str, documents: List[Dict]) -> str:
        """Generate response using PydanticAI with context and conversation history."""
        try:
            print("🤖 Generating response...")
            
            # Format documents as context
            context_parts = []
            for i, doc in enumerate(documents, 1):
                content = doc.get('content', '')
                doc_context = doc.get('context', '')
                similarity = doc.get('similarity', 0)
                video_id = doc.get('video_id', '')
                start_time = doc.get('start_time', '')
                end_time = doc.get('end_time', '')
                
                # Video metadata (from JOIN or nested object)
                video_url = doc.get('video_url', '')
                title = doc.get('title', '')
                channel_name = doc.get('channel_name', '')
                
                # Handle nested video_metadata object structure
                if not video_url and 'video_metadata' in doc and doc['video_metadata']:
                    metadata = doc['video_metadata']
                    video_url = metadata.get('video_url', '')
                    title = metadata.get('title', '')
                    channel_name = metadata.get('channel_name', '')
                
                context_part = f"Chunk {i} (relevance: {similarity:.2f}):\n{content}"
                if doc_context:
                    context_part += f"\nContext: {doc_context}"
                if title:
                    context_part += f"\nVideo Title: {title}"
                if channel_name:
                    context_part += f"\nChannel: {channel_name}"
                if video_url:
                    context_part += f"\nVideo URL: {video_url}"
                if start_time and end_time:
                    context_part += f"\nTimestamp: {start_time} - {end_time}"
                context_parts.append(context_part)
            
            documents_context = "\n\n".join(context_parts)
            
            # Get conversation context
            conversation_context = self.format_conversation_context()
            
            # Create system prompt for PydanticAI
            system_prompt = f"""You are a helpful AI assistant with access to a knowledge base of video content chunks. Your job is to synthesize information from the chunks and provide intelligent, coherent answers.

{conversation_context}Available content chunks:
{documents_context}

CRITICAL INSTRUCTIONS:
- DO NOT just copy and paste chunk content
- SYNTHESIZE the information into a clear, intelligent response
- Use your own words to explain concepts based on the chunk information
- Provide specific examples and details from the chunks when relevant
- If multiple chunks cover the same topic, combine their insights
- If the chunks don't contain enough information, clearly state what's missing
- Be conversational, helpful, and provide actionable advice when possible
- Always provide a complete, well-structured answer

REQUIRED: End your response with a "Sources:" section that lists:
- Video title and URL for each video referenced
- Specific timestamps when available
- Format like: "Video Title (timestamp) - URL"
- Only include sources that were actually used in your answer"""
            
            # Create a temporary agent with system prompt for this specific query
            from pydantic_ai import Agent
            temp_agent = Agent(
                self.chat_model_client,
                system_prompt=system_prompt
            )
            
            # Generate response using PydanticAI
            result = temp_agent.run_sync(query)
            
            # Extract the actual text response from PydanticAI result
            if hasattr(result, 'data'):
                response_text = result.data
            elif hasattr(result, 'content'):
                response_text = result.content
            elif hasattr(result, 'message'):
                response_text = result.message
            else:
                response_text = str(result)
            
            # Ensure we return a string, not raw data
            if isinstance(response_text, str):
                return response_text
            else:
                return str(response_text)
            
        except Exception as e:
            return f"❌ Error generating response: {str(e)}"
    
    def clear_conversation(self):
        """Clear conversation history."""
        self.conversation_history = []
        print("🧹 Conversation history cleared!")
    
    def show_conversation_history(self):
        """Display conversation history."""
        if not self.conversation_history:
            print("📝 No conversation history yet.")
            return
        
        print("📝 Conversation History:")
        print("-" * 50)
        for msg in self.conversation_history:
            role_name = "You" if msg.role == "user" else "Assistant"
            timestamp = msg.timestamp.strftime("%H:%M:%S")
            print(f"[{timestamp}] {role_name}: {msg.content[:100]}{'...' if len(msg.content) > 100 else ''}")
        print("-" * 50)
    
    def show_help(self):
        """Display help information."""
        help_text = """
🤖 Terminal RAG System with Memory - Help

Commands:
  help              - Show this help message
  clear             - Clear conversation history
  history           - Show conversation history
  search <query>    - Search documents only (no AI response)
  quit, exit, q     - Exit the system

Features:
- 🧠 Remembers conversation context
- 🔍 Searches your document database
- 💬 Conversational AI responses
- 📝 Tracks conversation history

Usage Examples:
  💬 You: What is machine learning?
  💬 You: How do I get started with it?  (remembers "it" = machine learning)
  💬 You: search python programming     (search only, no AI response)
  💬 You: clear                         (clear conversation memory)

Configuration:
- Similarity threshold: {self.similarity_threshold}
- Max results: {self.max_results}
- Max conversation history: {self.max_history} exchanges
- Model: {self.chat_model}
        """
        print(help_text)
    
    def process_query(self, query: str) -> Optional[str]:
        """Process a user query and return response."""
        query = query.strip()
        
        # Handle special commands
        if query.lower() in ['quit', 'exit', 'q']:
            return None
        
        if query.lower() == 'help':
            self.show_help()
            return ""
        
        if query.lower() == 'clear':
            self.clear_conversation()
            return ""
        
        if query.lower() == 'history':
            self.show_conversation_history()
            return ""
        
        # Handle search-only command
        if query.lower().startswith('search '):
            search_query = query[7:].strip()
            if search_query:
                documents = self.search_documents(search_query)
                if documents:
                    print("\n📄 Search Results:")
                    for i, doc in enumerate(documents, 1):
                        content = doc.get('content', '')[:200]
                        similarity = doc.get('similarity', 0)
                        video_id = doc.get('video_id', '')
                        start_time = doc.get('start_time', '')
                        end_time = doc.get('end_time', '')
                        
                        # Video metadata
                        video_url = doc.get('video_url', '')
                        title = doc.get('title', '')
                        channel_name = doc.get('channel_name', '')
                        
                        # Handle nested video_metadata object structure
                        if not video_url and 'video_metadata' in doc and doc['video_metadata']:
                            metadata = doc['video_metadata']
                            video_url = metadata.get('video_url', '')
                            title = metadata.get('title', '')
                            channel_name = metadata.get('channel_name', '')
                        
                        result_line = f"{i}. (Score: {similarity:.3f}) {content}..."
                        if title:
                            result_line += f"\n   🎬 {title}"
                        if channel_name:
                            result_line += f"\n   📺 Channel: {channel_name}"
                        if video_url:
                            result_line += f"\n   🔗 URL: {video_url}"
                        if start_time and end_time:
                            result_line += f"\n   ⏱️ Timestamp: {start_time}-{end_time}"
                        print(result_line)
                else:
                    print("❌ No relevant content chunks found.")
            return ""
        
        if not query:
            return ""
        
        # Add user query to conversation
        self.add_to_conversation("user", query)
        
        # Search for relevant documents
        documents = self.search_documents(query)
        
        if not documents:
            response = "❌ No relevant content chunks found in the database. Please try a different search term or check if your database contains relevant information."
        else:
            # Generate AI response
            response = self.generate_response(query, documents)
        
        # Add assistant response to conversation
        self.add_to_conversation("assistant", response)
        
        return response
    
    def run(self):
        """Main interactive loop."""
        print("✅ Connected to OpenAI and Supabase")
        print("🧠 Conversation memory enabled")
        print("\n🤖 Terminal RAG System with Memory Ready!")
        print("Type 'quit', 'exit', or 'q' to exit")
        print("Type 'help' for available commands")
        print("-" * 50)
        
        try:
            while True:
                try:
                    user_input = input("\n💬 You: ").strip()
                    
                    if not user_input:
                        continue
                    
                    response = self.process_query(user_input)
                    
                    if response is None:  # Quit command
                        break
                    
                    if response:  # Only print if there's a response
                        print(f"\n🤖 Assistant: {response}")
                
                except KeyboardInterrupt:
                    print("\n\n👋 Goodbye!")
                    break
                except EOFError:
                    print("\n\n👋 Goodbye!")
                    break
                except Exception as e:
                    print(f"\n❌ Error: {str(e)}")
                    print("Please try again or type 'help' for available commands.")
        
        except Exception as e:
            print(f"❌ Fatal error: {str(e)}")
            sys.exit(1)

def main():
    """Entry point for the Terminal RAG System."""
    try:
        # Initialize and run the system
        rag_system = TerminalRAGSystem(max_history=10)
        rag_system.run()
        print("\n👋 Goodbye!")
        
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error starting Terminal RAG System: {str(e)}")
        print("\nPlease check:")
        print("1. Your .env file contains valid API keys")
        print("2. You've run the database setup SQL in Supabase")
        print("3. All required packages are installed: pip install -r requirements.txt")
        sys.exit(1)

if __name__ == "__main__":
    main()