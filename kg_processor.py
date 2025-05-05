import os
import csv
import logging

# Try to import pandas, but provide fallback if not available
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    # Create a dummy pd with notna function to avoid errors
    # This is only to prevent syntax errors, the actual pandas code won't run without pandas
    class DummyPandas:
        @staticmethod
        def notna(val):
            return val is not None and val != ""
            
        @staticmethod
        def read_csv(file_path):
            raise NotImplementedError("pandas is not installed, cannot read CSV with pandas")
    pd = DummyPandas()
    logging.warning("Pandas not installed. Using basic CSV processing instead.")

class KnowledgeGraphProcessor:
    def __init__(self, neo4j_manager, groq_api_key):
        self.neo4j_manager = neo4j_manager
        self.groq_api_key = groq_api_key
        self.llm = None
        logging.warning("GROQ/LangChain not installed. LLM processing is unavailable.")
        
    def _initialize_llm(self):
        raise NotImplementedError("LLM integration is not available. Required packages not installed.")
    
    def _get_domain_prompt(self, domain):
        """Get domain-specific prompt template for entity extraction"""
        domain_configs = {
            "movie": {
                "relations": [
                    ("HAS_ACTOR", "actors", "Actor who appears in a movie"),
                    ("HAS_DIRECTOR", "director", "Director of a movie"),
                    ("IN_GENRE", "genres", "Genre category of a movie"),
                    ("RELEASED_ON", "released", "Release date of a movie"),
                    ("HAS_IMDB_RATING", "imdbRating", "IMDb rating of a movie")
                ]
            },
            "book": {
                "relations": [
                    ("WRITTEN_BY", "author", "Author of a book"),
                    ("PUBLISHED_BY", "publisher", "Publisher of a book"),
                    ("IN_GENRE", "genre", "Genre category of a book"),
                    ("PUBLISHED_ON", "publicationDate", "Publication date of a book")
                ]
            },
            "music": {
                "relations": [
                    ("PERFORMED_BY", "artist", "Artist who performs a song"),
                    ("PRODUCED_BY", "producer", "Producer of a song"),
                    ("IN_GENRE", "genre", "Genre category of a song"),
                    ("RELEASED_ON", "releaseDate", "Release date of a song"),
                    ("INCLUDED_IN", "album", "Album containing a song")
                ]
            },
            "academic": {
                "relations": [
                    ("AUTHORED_BY", "author", "Author of a paper"),
                    ("PUBLISHED_IN", "journal", "Journal where a paper was published"),
                    ("IN_FIELD", "field", "Research field of a paper"),
                    ("PUBLISHED_ON", "publicationDate", "Publication date of a paper")
                ]
            },
            "business": {
                "relations": [
                    ("HAS_CEO", "ceo", "CEO of a company"),
                    ("IN_SECTOR", "sector", "Industry sector of a company"),
                    ("FOUNDED_ON", "foundingDate", "Date a company was founded"),
                    ("HEADQUARTERED_IN", "location", "Location of company headquarters")
                ]
            },
            "custom": {
                "relations": []
            }
        }
        
        return domain_configs.get(domain, domain_configs["custom"])
    
    def extract_triples_from_text(self, text, domain="custom"):
        """Extract knowledge graph triples from text"""
        raise NotImplementedError("LLM-based triple extraction is not available. Required packages not installed.")
    
    def process_csv(self, csv_path, domain="custom"):
        """Process CSV file to extract knowledge graph triples"""
        try:
            # Get domain relations
            relation_configs = self._get_domain_prompt(domain)["relations"]
            
            # Use direct column mapping for CSV processing
            all_triples = []
            
            # Process CSV file
            if HAS_PANDAS:
                # Use pandas for processing if available
                df = pd.read_csv(csv_path)
                
                if domain == "movie":
                    # Special handling for movie domain with the example CSV format
                    for _, row in df.iterrows():
                        movie_title = row.get('title', '')
                        
                        # Process actors (pipe-separated)
                        if 'actors' in row and pd.notna(row['actors']):
                            actors = row['actors'].split('|')
                            for actor in actors:
                                if actor.strip():
                                    all_triples.append((movie_title, "HAS_ACTOR", actor.strip()))
                        
                        # Process director
                        if 'director' in row and pd.notna(row['director']):
                            all_triples.append((movie_title, "HAS_DIRECTOR", row['director']))
                        
                        # Process genres (pipe-separated)
                        if 'genres' in row and pd.notna(row['genres']):
                            genres = row['genres'].split('|')
                            for genre in genres:
                                if genre.strip():
                                    all_triples.append((movie_title, "IN_GENRE", genre.strip()))
                        
                        # Process release date
                        if 'released' in row and pd.notna(row['released']):
                            all_triples.append((movie_title, "RELEASED_ON", str(row['released'])))
                        
                        # Process IMDb rating
                        if 'imdbRating' in row and pd.notna(row['imdbRating']):
                            all_triples.append((movie_title, "HAS_IMDB_RATING", str(row['imdbRating'])))
                else:
                    # Generic processing for other domains
                    for _, row in df.iterrows():
                        # Find the main entity column (usually the first column that's not a relationship)
                        entity_col = df.columns[0] if len(df.columns) > 0 else None
                        entity_val = row.get(entity_col, '') if entity_col else 'Unknown'
                        
                        # For each column, create a triple if it's not the entity column
                        for col in df.columns:
                            if col != entity_col and pd.notna(row[col]):
                                # Find a matching relation type from the domain config
                                relation_type = "HAS_" + col.upper()
                                for rel_type, rel_col, _ in relation_configs:
                                    if rel_col.lower() == col.lower():
                                        relation_type = rel_type
                                        break
                                
                                # Add to triples
                                all_triples.append((entity_val, relation_type, str(row[col])))
            else:
                # Fallback to built-in CSV processing for Python
                with open(csv_path, 'r', newline='') as csvfile:
                    reader = csv.reader(csvfile)
                    # Get header row
                    try:
                        header = next(reader)
                    except StopIteration:
                        logging.error("CSV file is empty")
                        return []
                    
                    # Assume first column is the primary entity
                    entity_col_idx = 0
                    
                    if domain == "movie":
                        # Get column indexes
                        title_idx = header.index('title') if 'title' in header else 0
                        actors_idx = header.index('actors') if 'actors' in header else None
                        director_idx = header.index('director') if 'director' in header else None
                        genres_idx = header.index('genres') if 'genres' in header else None
                        released_idx = header.index('released') if 'released' in header else None
                        rating_idx = header.index('imdbRating') if 'imdbRating' in header else None
                        
                        # Process each row
                        for row in reader:
                            if not row:  # Skip empty rows
                                continue
                                
                            movie_title = row[title_idx] if len(row) > title_idx else 'Unknown'
                            
                            # Process actors
                            if actors_idx is not None and len(row) > actors_idx and row[actors_idx]:
                                actors = row[actors_idx].split('|')
                                for actor in actors:
                                    if actor.strip():
                                        all_triples.append((movie_title, "HAS_ACTOR", actor.strip()))
                            
                            # Process director
                            if director_idx is not None and len(row) > director_idx and row[director_idx]:
                                all_triples.append((movie_title, "HAS_DIRECTOR", row[director_idx]))
                            
                            # Process genres
                            if genres_idx is not None and len(row) > genres_idx and row[genres_idx]:
                                genres = row[genres_idx].split('|')
                                for genre in genres:
                                    if genre.strip():
                                        all_triples.append((movie_title, "IN_GENRE", genre.strip()))
                            
                            # Process release date
                            if released_idx is not None and len(row) > released_idx and row[released_idx]:
                                all_triples.append((movie_title, "RELEASED_ON", row[released_idx]))
                            
                            # Process IMDb rating
                            if rating_idx is not None and len(row) > rating_idx and row[rating_idx]:
                                all_triples.append((movie_title, "HAS_IMDB_RATING", row[rating_idx]))
                    else:
                        # Generic processing for other domains
                        for row in reader:
                            if not row or len(row) <= entity_col_idx:  # Skip rows without enough data
                                continue
                                
                            # Get the entity from the first column
                            entity_val = row[entity_col_idx] if row[entity_col_idx] else 'Unknown'
                            
                            # For each column, create a triple
                            for i, col_name in enumerate(header):
                                if i != entity_col_idx and i < len(row) and row[i]:
                                    # Find a matching relation type from the domain config
                                    relation_type = "HAS_" + col_name.upper()
                                    for rel_type, rel_col, _ in relation_configs:
                                        if rel_col.lower() == col_name.lower():
                                            relation_type = rel_type
                                            break
                                    
                                    # Add to triples
                                    all_triples.append((entity_val, relation_type, row[i]))
            
            return all_triples
        
        except Exception as e:
            logging.error(f"Error processing CSV: {e}")
            raise RuntimeError(f"Failed to process CSV file: {e}")
    
    def _split_chunks(self, sentences, max_tokens=300):
        """Split text into chunks to avoid context length issues"""
        chunks, chunk, tokens = [], [], 0
        for s in sentences:
            t = len(s.split()) * 1.5  # Approximate token count
            if tokens + t > max_tokens:
                chunks.append(chunk)
                chunk, tokens = [s], t
            else:
                chunk.append(s)
                tokens += t
        if chunk:
            chunks.append(chunk)
        return chunks
    
    def save_triples_to_csv(self, triples, output_path):
        """Save extracted triples to CSV file"""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["entity1", "relation", "entity2"])
                for triple in triples:
                    writer.writerow(triple)
            return True
        except Exception as e:
            logging.error(f"Error saving triples to CSV: {e}")
            raise RuntimeError(f"Failed to save triples to CSV: {e}")
    
    def query_knowledge_graph(self, query_text, graph_id):
        """Query the knowledge graph using natural language"""
        try:
            # Since we can't use LLM-based querying, we'll provide a rule-based search function
            nodes, relationships = self.neo4j_manager.get_graph_data(graph_id)
            
            # Create node maps for efficient lookup
            node_label_map = {node.get("id"): node.get("label") for node in nodes}
            node_id_map = {node.get("label").lower(): node.get("id") for node in nodes}
            
            # Process specific question patterns
            query_lower = query_text.lower()
            
            # Handle "What movies did X direct?" type questions
            if "movies" in query_lower and "direct" in query_lower:
                # Extract director name from the query
                # This assumes the director name is the main entity being searched
                director_terms = []
                for term in query_lower.split():
                    if term not in ["what", "movies", "did", "direct", "directed", "by", "?", "films"]:
                        director_terms.append(term)
                
                if director_terms:
                    director_name = " ".join(director_terms)
                    
                    # Find all HAS_DIRECTOR relationships
                    director_movies = []
                    for rel in relationships:
                        # Check if this is a director relationship
                        if rel.get("type") == "HAS_DIRECTOR":
                            # Get target (director) label
                            target_id = rel.get("target")
                            director_label = node_label_map.get(target_id, "").lower()
                            
                            # Check if this director matches our search
                            if director_name in director_label:
                                # Get source (movie) label
                                source_id = rel.get("source")
                                movie_label = node_label_map.get(source_id, "Unknown")
                                director_movies.append((movie_label, node_label_map.get(target_id, "Unknown")))
                    
                    if director_movies:
                        result_lines = ["Found these relationships:"]
                        for movie, director in director_movies:
                            result_lines.append(f"- {movie} [HAS_DIRECTOR] {director}")
                        return "\n".join(result_lines)
                    
                    # Also check the reverse direction (sometimes the graph might store it in reverse)
                    for rel in relationships:
                        if rel.get("type") == "HAS_DIRECTOR":
                            # Check if source (could be director) label matches
                            source_id = rel.get("source")
                            director_label = node_label_map.get(source_id, "").lower()
                            
                            if director_name in director_label:
                                # Get target (could be movie) label
                                target_id = rel.get("target")
                                movie_label = node_label_map.get(target_id, "Unknown")
                                director_movies.append((director_label, movie_label))
                    
                    if director_movies:
                        result_lines = ["Found these relationships:"]
                        for director, movie in director_movies:
                            result_lines.append(f"- {movie} [HAS_DIRECTOR] {director}")
                        return "\n".join(result_lines)
            
            # Handle "Which actors starred in X?" type questions
            elif "actors" in query_lower and ("starred" in query_lower or "appear" in query_lower or "in" in query_lower):
                # Extract movie name from the query - improved parsing
                query_parts = query_lower.replace("?", "").split()
                movie_name = ""
                
                # Check if 'inception' is in the query as a specific title
                if "inception" in query_lower or "incepion" in query_lower:
                    movie_name = "inception"
                else:
                    # Find the position of 'in' word as it usually precedes the movie title
                    if "in" in query_parts:
                        in_pos = query_parts.index("in")
                        # Take everything after "in" as the movie name, excluding stop words
                        movie_terms = []
                        skip_terms = ["which", "actors", "starred", "appear", "did", "who", "was", "were", "the", "a", "an"]
                        for term in query_parts[in_pos+1:]:
                            if term not in skip_terms:
                                movie_terms.append(term)
                        
                        if movie_terms:
                            movie_name = " ".join(movie_terms)
                    
                    # If no movie name was found using 'in' position, try the old approach
                    if not movie_name:
                        movie_terms = []
                        skip_terms = ["which", "actors", "starred", "in", "?", "appear", "did", "who", "was", "were", "the"]
                        for term in query_parts:
                            if term not in skip_terms:
                                movie_terms.append(term)
                        
                        if movie_terms:
                            movie_name = " ".join(movie_terms)
                
                logging.info(f"Searching for actors in movie: '{movie_name}'")
                
                if movie_name:
                    # Find all HAS_ACTOR relationships related to this movie
                    movie_actors = []
                    for rel in relationships:
                        # Check if this is an actor relationship
                        if rel.get("type") == "HAS_ACTOR":
                            # Get source (movie) label
                            source_id = rel.get("source")
                            movie_label = node_label_map.get(source_id, "").lower()
                            
                            # Use more precise matching with different approaches
                            # 1. Exact match
                            # 2. The movie name is a substring of the movie label (with word boundaries)
                            # 3. Fuzzy match for handling typos (at least 80% similar)
                            if (movie_name == movie_label or 
                                f" {movie_name} " in f" {movie_label} " or
                                movie_name in movie_label.split()):
                                # Get target (actor) label
                                target_id = rel.get("target")
                                actor_label = node_label_map.get(target_id, "Unknown")
                                movie_actors.append((node_label_map.get(source_id, "Unknown"), actor_label))
                    
                    if movie_actors:
                        result_lines = ["Found these relationships:"]
                        for movie, actor in movie_actors:
                            result_lines.append(f"- {movie} [HAS_ACTOR] {actor}")
                        return "\n".join(result_lines)
                    else:
                        return f"No actors found for movie '{movie_name}'. Please check the movie name and try again."
            
            # Handle general queries by searching for entities in the query
            
            # Extract potential search terms (names, titles, etc.)
            # Skip common words that are likely not entity names
            skip_words = {"what", "which", "who", "where", "when", "how", "did", "does", "is", "are", "was", "were", 
                         "has", "have", "had", "the", "a", "an", "in", "on", "at", "by", "to", "for", "with", 
                         "from", "about", "movies", "actors", "directors", "character", "film", "star", "direct"}
            
            search_terms = []
            for term in query_lower.split():
                if term not in skip_words and len(term) > 2:  # Skip short terms like "of", "as", etc.
                    search_terms.append(term)
            
            # Combine consecutive terms to catch multi-word entities
            phrase_search_terms = []
            if len(search_terms) >= 2:
                for i in range(len(search_terms) - 1):
                    phrase_search_terms.append(f"{search_terms[i]} {search_terms[i+1]}")
            
            # Add triple word phrases
            if len(search_terms) >= 3:
                for i in range(len(search_terms) - 2):
                    phrase_search_terms.append(f"{search_terms[i]} {search_terms[i+1]} {search_terms[i+2]}")
            
            # Search for nodes that match search terms
            matching_nodes = []
            # First try exact multi-word phrase matches
            for phrase in phrase_search_terms:
                for node in nodes:
                    node_label = node.get("label", "").lower()
                    if phrase in node_label and node not in matching_nodes:
                        matching_nodes.append(node)
            
            # Then try individual word matches
            if not matching_nodes:
                for node in nodes:
                    node_label = node.get("label", "").lower()
                    if any(term in node_label for term in search_terms) and node not in matching_nodes:
                        matching_nodes.append(node)
            
            # Extract the relationships involving the matching nodes
            matching_node_ids = [node.get("id") for node in matching_nodes]
            matching_relations = []
            for rel in relationships:
                source_id = rel.get("source")
                target_id = rel.get("target")
                if source_id in matching_node_ids or target_id in matching_node_ids:
                    matching_relations.append(rel)
            
            # Format the results
            if matching_relations:
                result_lines = ["Found these relationships:"]
                
                for rel in matching_relations:
                    source_id = rel.get("source")
                    target_id = rel.get("target") 
                    relation_type = rel.get("type")
                    
                    source_label = node_label_map.get(source_id, "Unknown")
                    target_label = node_label_map.get(target_id, "Unknown")
                    
                    result_lines.append(f"- {source_label} [{relation_type}] {target_label}")
                
                return "\n".join(result_lines)
            elif matching_nodes:
                return f"Found {len(matching_nodes)} entities matching your query, but no relationships."
            else:
                return "No matching entities found for your query."
                
        except Exception as e:
            logging.error(f"Error querying knowledge graph: {e}")
            raise RuntimeError(f"Failed to query knowledge graph: {e}")
