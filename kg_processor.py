import os
import csv
import json
import logging
import re
import random
import string
from datetime import datetime

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

# Define entity types and their properties
ENTITY_TYPES = {
    "Movie": {
        "properties": ["title", "year", "runtime", "plot", "language", "country", "awards"],
        "data_types": {
            "title": "string",
            "year": "number",
            "runtime": "number",
            "plot": "string",
            "language": "string",
            "country": "string",
            "awards": "string"
        }
    },
    "Person": {
        "properties": ["name", "birthdate", "nationality", "gender"],
        "data_types": {
            "name": "string",
            "birthdate": "date",
            "nationality": "string",
            "gender": "string"
        }
    },
    "Genre": {
        "properties": ["name", "description"],
        "data_types": {
            "name": "string",
            "description": "string"
        }
    },
    "Book": {
        "properties": ["title", "isbn", "pages", "summary", "language", "publicationYear"],
        "data_types": {
            "title": "string",
            "isbn": "string",
            "pages": "number",
            "summary": "string",
            "language": "string",
            "publicationYear": "number"
        }
    },
    "Artist": {
        "properties": ["name", "formationYear", "disbandmentYear", "genre", "country"],
        "data_types": {
            "name": "string",
            "formationYear": "number",
            "disbandmentYear": "number",
            "genre": "string",
            "country": "string"
        }
    },
    "Song": {
        "properties": ["title", "duration", "releaseYear", "lyrics", "language"],
        "data_types": {
            "title": "string",
            "duration": "number",
            "releaseYear": "number",
            "lyrics": "string",
            "language": "string"
        }
    },
    "Institution": {
        "properties": ["name", "foundedYear", "type", "location", "description"],
        "data_types": {
            "name": "string",
            "foundedYear": "number",
            "type": "string",
            "location": "string",
            "description": "string"
        }
    }
}

# Define relationship types and their properties
RELATIONSHIP_TYPES = {
    "ACTED_IN": {
        "source": ["Person"],
        "target": ["Movie"],
        "properties": ["role", "screen_time"],
        "bidirectional": True,
        "reverse_name": "HAS_ACTOR"
    },
    "DIRECTED": {
        "source": ["Person"],
        "target": ["Movie"],
        "properties": ["year"],
        "bidirectional": True,
        "reverse_name": "DIRECTED_BY"
    },
    "BELONGS_TO": {
        "source": ["Movie", "Song"],
        "target": ["Genre"],
        "properties": [],
        "bidirectional": True,
        "reverse_name": "CONTAINS"
    },
    "WROTE": {
        "source": ["Person"],
        "target": ["Book", "Movie"],
        "properties": ["year"],
        "bidirectional": True,
        "reverse_name": "WRITTEN_BY"
    },
    "SIMILAR_TO": {
        "source": ["Movie", "Book", "Song"],
        "target": ["Movie", "Book", "Song"],
        "properties": ["similarity_score"],
        "bidirectional": True,
        "reverse_name": "SIMILAR_TO"
    },
    "PERFORMED": {
        "source": ["Artist", "Person"],
        "target": ["Song"],
        "properties": ["year", "location"],
        "bidirectional": True,
        "reverse_name": "PERFORMED_BY"
    },
    "AFFILIATED_WITH": {
        "source": ["Person"],
        "target": ["Institution"],
        "properties": ["start_year", "end_year", "role"],
        "bidirectional": True,
        "reverse_name": "HAS_MEMBER"
    }
}

class KnowledgeGraphProcessor:
    def __init__(self, neo4j_manager, groq_api_key=None):
        self.neo4j_manager = neo4j_manager
        self.groq_api_key = groq_api_key
        self.llm = None
        
        # Entity type recognition patterns
        self.entity_patterns = {
            "Person": [
                r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b',  # Simple name pattern
                r'\b(?:Mr|Mrs|Ms|Dr|Prof)\.? ([A-Z][a-z]+ [A-Z][a-z]+)\b'  # Name with title
            ],
            "Movie": [
                r'"([^"]+)"',  # Quoted title
                r'movie[s]? (?:called|titled|named) "([^"]+)"',
                r'(?:film|movie|documentary) "([^"]+)"'
            ],
            "Date": [
                r'\b(January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2},? \d{4}\b',
                r'\b\d{4}-\d{2}-\d{2}\b',
                r'\b\d{1,2}/\d{1,2}/\d{4}\b'
            ],
            "Year": [
                r'\b(19|20)\d{2}\b'  # Simple year pattern
            ],
            "Genre": [
                r'\b(Action|Adventure|Comedy|Drama|Horror|Sci-Fi|Thriller|Romance|Fantasy|Animation|Documentary|Biography)\b'
            ]
        }
        
        # Try importing more advanced NLP if available
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
            self.has_nlp = True
            logging.info("Spacy NLP loaded successfully")
        except:
            self.has_nlp = False
            logging.warning("Spacy NLP not available. Using basic pattern matching for entity extraction.")
            
        try:
            import nltk
            nltk.download('punkt', quiet=True)
            self.has_nltk = True
            logging.info("NLTK loaded successfully")
        except:
            self.has_nltk = False
            logging.warning("NLTK not available. Using basic sentence splitting.")
        
        if not groq_api_key:
            logging.warning("GROQ API key not provided. LLM processing is unavailable.")
        else:
            try:
                self._initialize_llm()
            except Exception as e:
                logging.warning(f"Failed to initialize LLM: {e}")
        
    def _initialize_llm(self):
        # Placeholder for LLM initialization
        try:
            # If LangChain is available, we would initialize the LLM here
            logging.info("LLM would be initialized here if dependencies were available.")
            return True
        except Exception as e:
            logging.error(f"Error initializing LLM: {e}")
            return False
    
    def _generate_unique_id(self, entity_type, label):
        """Generate a unique ID for an entity based on its type and label"""
        # Clean the label and combine with type
        clean_label = re.sub(r'[^a-zA-Z0-9]', '', label)
        prefix = entity_type[:3].upper()
        
        # Add a timestamp and random characters to ensure uniqueness
        timestamp = datetime.now().strftime('%y%m%d%H%M%S')
        random_chars = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
        
        return f"{prefix}-{clean_label[:10]}-{timestamp}-{random_chars}"
    
    def _get_domain_prompt(self, domain):
        """Get domain-specific prompt template for entity extraction"""
        domain_configs = {
            "movie": {
                "entity_types": ["Movie", "Person", "Genre"],
                "relations": [
                    ("HAS_ACTOR", "actors", "Actor who appears in a movie"),
                    ("DIRECTED_BY", "director", "Director of a movie"),
                    ("BELONGS_TO", "genres", "Genre category of a movie"),
                    ("RELEASED_ON", "released", "Release date of a movie"),
                    ("HAS_IMDB_RATING", "imdbRating", "IMDb rating of a movie"),
                    ("SIMILAR_TO", None, "Movie similar to another movie"),
                    ("WRITTEN_BY", "writers", "Writers of the movie screenplay")
                ]
            },
            "book": {
                "entity_types": ["Book", "Person", "Genre", "Institution"],
                "relations": [
                    ("WRITTEN_BY", "author", "Author of a book"),
                    ("PUBLISHED_BY", "publisher", "Publisher of a book"),
                    ("BELONGS_TO", "genre", "Genre category of a book"),
                    ("PUBLISHED_ON", "publicationDate", "Publication date of a book"),
                    ("SIMILAR_TO", None, "Book similar to another book"),
                    ("AFFILIATED_WITH", None, "Author affiliated with an institution")
                ]
            },
            "music": {
                "entity_types": ["Song", "Artist", "Person", "Genre"],
                "relations": [
                    ("PERFORMED_BY", "artist", "Artist who performs a song"),
                    ("WRITTEN_BY", "songwriter", "Person who wrote a song"),
                    ("PRODUCED_BY", "producer", "Producer of a song"),
                    ("BELONGS_TO", "genre", "Genre category of a song"),
                    ("RELEASED_ON", "releaseDate", "Release date of a song"),
                    ("INCLUDED_IN", "album", "Album containing a song"),
                    ("SIMILAR_TO", None, "Song similar to another song")
                ]
            },
            "academic": {
                "entity_types": ["Person", "Institution", "Genre"],
                "relations": [
                    ("WRITTEN_BY", "author", "Author of a paper"),
                    ("PUBLISHED_IN", "journal", "Journal where a paper was published"),
                    ("BELONGS_TO", "field", "Research field of a paper"),
                    ("PUBLISHED_ON", "publicationDate", "Publication date of a paper"),
                    ("AFFILIATED_WITH", "institution", "Institution affiliated with an author"),
                    ("COLLABORATED_WITH", None, "Authors who collaborated on research")
                ]
            },
            "business": {
                "entity_types": ["Institution", "Person"],
                "relations": [
                    ("LED_BY", "ceo", "CEO of a company"),
                    ("BELONGS_TO", "sector", "Industry sector of a company"),
                    ("FOUNDED_ON", "foundingDate", "Date a company was founded"),
                    ("HEADQUARTERED_IN", "location", "Location of company headquarters"),
                    ("AFFILIATED_WITH", "parent", "Parent company or organization"),
                    ("INVESTED_IN", None, "Company invested in by another company")
                ]
            },
            "custom": {
                "entity_types": ["Person", "Institution", "Genre"],
                "relations": []
            }
        }
        
        return domain_configs.get(domain, domain_configs["custom"])
    
    def extract_entities_with_patterns(self, text, domain="custom"):
        """Extract entities using regex patterns and rules"""
        entities = {}
        domain_config = self._get_domain_prompt(domain)
        entity_types = domain_config.get("entity_types", [])
        
        # Extract entities with patterns
        for entity_type, patterns in self.entity_patterns.items():
            if entity_type not in entity_types and entity_type not in ["Date", "Year"]:
                continue
                
            entities[entity_type] = []
            for pattern in patterns:
                matches = re.finditer(pattern, text)
                for match in matches:
                    entity_text = match.group(1) if len(match.groups()) > 0 else match.group(0)
                    entity_id = self._generate_unique_id(entity_type, entity_text)
                    
                    # Skip duplicates
                    if any(e['text'] == entity_text for e in entities[entity_type]):
                        continue
                        
                    entity_data = {'id': entity_id, 'text': entity_text, 'type': entity_type}
                    entities[entity_type].append(entity_data)
        
        return entities
    
    def extract_triples_from_text(self, text, domain="custom"):
        """Extract knowledge graph triples from text with enhanced NLP"""
        # Start with regex pattern extraction
        entities_by_type = self.extract_entities_with_patterns(text, domain)
        
        # Flatten entities
        entities = []
        for entity_type, entity_list in entities_by_type.items():
            entities.extend(entity_list)
        
        # Extract triples using basic heuristics
        triples = []
        domain_config = self._get_domain_prompt(domain)
        relation_configs = domain_config.get("relations", [])
        
        # Split text into sentences
        if self.has_nltk:
            import nltk
            sentences = nltk.sent_tokenize(text)
        else:
            # Basic sentence splitting
            sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Process each sentence to find entity pairs
        for sentence in sentences:
            sentence_entities = []
            for entity in entities:
                if entity['text'] in sentence:
                    sentence_entities.append(entity)
            
            # If we have at least 2 entities, try to create triples
            if len(sentence_entities) >= 2:
                # Check for relation patterns in the sentence
                for rel_type, rel_col, rel_desc in relation_configs:
                    relation_pattern = self._get_relation_pattern(rel_type)
                    if relation_pattern and re.search(relation_pattern, sentence.lower()):
                        # Find compatible entity pairs
                        for i, entity1 in enumerate(sentence_entities):
                            for entity2 in sentence_entities[i+1:]:
                                if self._are_entities_compatible(entity1['type'], entity2['type'], rel_type):
                                    # Create triple
                                    triples.append((entity1['text'], rel_type, entity2['text']))
                    
                    # If no specific pattern found, check proximity-based relations
                    if not triples and len(sentence_entities) == 2:
                        entity1, entity2 = sentence_entities[0], sentence_entities[1]
                        # Try to infer relation based on entity types
                        rel_type = self._infer_relation(entity1['type'], entity2['type'], domain)
                        if rel_type:
                            triples.append((entity1['text'], rel_type, entity2['text']))
        
        # If we have spaCy available, enhance triple extraction with dependency parsing
        if self.has_nlp and not triples:
            try:
                doc = self.nlp(text)
                # Extract subject-verb-object patterns
                for sentence in doc.sents:
                    subjects = []
                    objects = []
                    verbs = []
                    
                    for token in sentence:
                        # Find subjects
                        if token.dep_ in ("nsubj", "nsubjpass"):
                            subjects.append(token.text)
                            
                        # Find objects
                        if token.dep_ in ("dobj", "pobj", "attr"):
                            objects.append(token.text)
                            
                        # Find verbs
                        if token.pos_ == "VERB":
                            verbs.append(token.text)
                    
                    # Create triples if we have subject-verb-object
                    if subjects and verbs and objects:
                        relation = verbs[0].upper()
                        for subject in subjects:
                            for obj in objects:
                                triples.append((subject, relation, obj))
            except Exception as e:
                logging.error(f"Error in spaCy entity extraction: {e}")
        
        return triples
    
    def _get_relation_pattern(self, relation_type):
        """Get regex pattern for a relation type"""
        relation_patterns = {
            "HAS_ACTOR": r'(?:stars|features|has|with|starring)\s+(\w+)',
            "DIRECTED_BY": r'(?:directed|helmed|created)\s+by\s+(\w+)',
            "BELONGS_TO": r'(?:is|belongs to|categorized as|in)\s+(?:genre|category)',
            "RELEASED_ON": r'(?:released|came out|premiered)\s+(?:on|in)',
            "WRITTEN_BY": r'(?:written|authored|penned)\s+by\s+(\w+)',
            "PERFORMED_BY": r'(?:performed|sung|played)\s+by\s+(\w+)',
            "PRODUCED_BY": r'(?:produced|created)\s+by\s+(\w+)',
            "AFFILIATED_WITH": r'(?:affiliated|associated|connected|working)\s+with\s+(\w+)',
            "SIMILAR_TO": r'(?:similar|like|resembles|comparable)\s+to\s+(\w+)'
        }
        
        return relation_patterns.get(relation_type.upper())
    
    def _are_entities_compatible(self, entity1_type, entity2_type, relation_type):
        """Check if entities are compatible for a given relation"""
        # Convert to uppercase for comparison
        relation_type = relation_type.upper()
        
        # Look up relation in RELATIONSHIP_TYPES
        rel_config = None
        for rel, config in RELATIONSHIP_TYPES.items():
            if rel.upper() == relation_type or (config.get('reverse_name', '').upper() == relation_type):
                rel_config = config
                break
        
        if not rel_config:
            return False
            
        # Check if entity types are compatible with the relationship
        if relation_type == rel_config.get('reverse_name', '').upper():
            # Swap source and target for reverse relationship
            return entity1_type in rel_config['target'] and entity2_type in rel_config['source']
        else:
            return entity1_type in rel_config['source'] and entity2_type in rel_config['target']
    
    def _infer_relation(self, entity1_type, entity2_type, domain):
        """Infer a relation type based on entity types and domain"""
        domain_config = self._get_domain_prompt(domain)
        relation_configs = domain_config.get("relations", [])
        
        # Check which relations are compatible with the entity types
        compatible_relations = []
        for rel_type, _, _ in relation_configs:
            if self._are_entities_compatible(entity1_type, entity2_type, rel_type):
                compatible_relations.append(rel_type)
                
        # Return the first compatible relation, if any
        return compatible_relations[0] if compatible_relations else None
    
    def process_csv(self, csv_path, domain="custom"):
        """Process CSV file to extract knowledge graph triples with enhanced entity and relation detection"""
        try:
            # Get domain relations
            domain_config = self._get_domain_prompt(domain)
            relation_configs = domain_config.get("relations", [])
            entity_types = domain_config.get("entity_types", [])
            
            # Use direct column mapping for CSV processing
            all_triples = []
            all_entities = set()
            entity_properties = {}
            
            # Process CSV file
            if HAS_PANDAS:
                # Use pandas for processing if available
                df = pd.read_csv(csv_path)
                
                if domain == "movie":
                    # Special handling for movie domain with the example CSV format
                    for _, row in df.iterrows():
                        movie_title = row.get('title', '')
                        if not movie_title:
                            continue
                            
                        # Add movie entity
                        all_entities.add(("Movie", movie_title))
                        
                        # Store movie properties
                        movie_props = {}
                        for prop in ["year", "runtime", "plot", "language", "country", "awards"]:
                            if prop in row and pd.notna(row[prop]):
                                movie_props[prop] = str(row[prop])
                        if movie_props:
                            entity_properties[(movie_title, "Movie")] = movie_props
                        
                        # Process actors (pipe-separated)
                        if 'actors' in row and pd.notna(row['actors']):
                            actors = row['actors'].split('|')
                            for actor in actors:
                                actor = actor.strip()
                                if actor:
                                    # Add actor entity
                                    all_entities.add(("Person", actor))
                                    # Add relation
                                    all_triples.append((actor, "ACTED_IN", movie_title))
                                    all_triples.append((movie_title, "HAS_ACTOR", actor))
                        
                        # Process director
                        if 'director' in row and pd.notna(row['director']):
                            director = row['director'].strip()
                            # Add director entity
                            all_entities.add(("Person", director))
                            # Add relation
                            all_triples.append((director, "DIRECTED", movie_title))
                            all_triples.append((movie_title, "DIRECTED_BY", director))
                        
                        # Process genres (pipe-separated)
                        if 'genres' in row and pd.notna(row['genres']):
                            genres = row['genres'].split('|')
                            for genre in genres:
                                genre = genre.strip()
                                if genre:
                                    # Add genre entity
                                    all_entities.add(("Genre", genre))
                                    # Add relation
                                    all_triples.append((movie_title, "BELONGS_TO", genre))
                                    all_triples.append((genre, "CONTAINS", movie_title))
                        
                        # Process release date
                        if 'released' in row and pd.notna(row['released']):
                            released = str(row['released'])
                            all_triples.append((movie_title, "RELEASED_ON", released))
                        
                        # Process IMDb rating
                        if 'imdbRating' in row and pd.notna(row['imdbRating']):
                            rating = str(row['imdbRating'])
                            all_triples.append((movie_title, "HAS_IMDB_RATING", rating))
                else:
                    # Enhanced processing for other domains
                    # Try to identify the main entity and child entities
                    
                    # First pass: identify primary entity column and type
                    primary_entity_col = None
                    primary_entity_type = None
                    
                    # Check for domain-specific primary entity types
                    if domain == "book":
                        primary_entity_type = "Book"
                        title_columns = [col for col in df.columns if col.lower() in ["title", "name", "book_title"]]
                        if title_columns:
                            primary_entity_col = title_columns[0]
                    elif domain == "music":
                        primary_entity_type = "Song"
                        title_columns = [col for col in df.columns if col.lower() in ["title", "song", "track"]]
                        if title_columns:
                            primary_entity_col = title_columns[0]
                    elif domain == "academic":
                        primary_entity_type = "Person"
                        name_columns = [col for col in df.columns if col.lower() in ["author", "researcher", "name"]]
                        if name_columns:
                            primary_entity_col = name_columns[0]
                    elif domain == "business":
                        primary_entity_type = "Institution"
                        name_columns = [col for col in df.columns if col.lower() in ["company", "business", "name"]]
                        if name_columns:
                            primary_entity_col = name_columns[0]
                    
                    # If we couldn't determine the primary entity, use the first column
                    if not primary_entity_col and len(df.columns) > 0:
                        primary_entity_col = df.columns[0]
                        
                        # Guess entity type from column name
                        if "title" in primary_entity_col.lower():
                            if domain in ["movie", "book"]:
                                primary_entity_type = "Movie" if domain == "movie" else "Book"
                        elif "name" in primary_entity_col.lower():
                            primary_entity_type = "Person"
                    
                    # Default to first entity type from domain config if still not set
                    if not primary_entity_type and entity_types:
                        primary_entity_type = entity_types[0]
                    
                    # Process each row
                    for _, row in df.iterrows():
                        if not primary_entity_col or primary_entity_col not in row:
                            continue
                            
                        primary_entity = row[primary_entity_col]
                        if not pd.notna(primary_entity) or not primary_entity:
                            continue
                        
                        primary_entity = str(primary_entity).strip()
                        
                        # Add primary entity
                        all_entities.add((primary_entity_type, primary_entity))
                        
                        # Store primary entity properties
                        primary_props = {}
                        if primary_entity_type in ENTITY_TYPES:
                            for prop in ENTITY_TYPES[primary_entity_type]["properties"]:
                                if prop in row and pd.notna(row[prop]):
                                    primary_props[prop] = str(row[prop])
                        if primary_props:
                            entity_properties[(primary_entity, primary_entity_type)] = primary_props
                        
                        # For each column, check if it's a potential relationship
                        for col in df.columns:
                            if col == primary_entity_col or not pd.notna(row[col]) or not row[col]:
                                continue
                            
                            # Check if column matches any relation
                            matched_relation = None
                            related_entity_type = None
                            
                            for rel_type, rel_col, _ in relation_configs:
                                if rel_col and rel_col.lower() == col.lower():
                                    matched_relation = rel_type
                                    # Determine related entity type
                                    for rel_name, rel_config in RELATIONSHIP_TYPES.items():
                                        if rel_name == matched_relation or rel_config.get("reverse_name") == matched_relation:
                                            if primary_entity_type in rel_config["source"]:
                                                related_entity_type = rel_config["target"][0]
                                            else:
                                                related_entity_type = rel_config["source"][0]
                                            break
                                    break
                            
                            if not matched_relation:
                                # Use column name to infer relation
                                if "actor" in col.lower() or "cast" in col.lower():
                                    matched_relation = "HAS_ACTOR"
                                    related_entity_type = "Person"
                                elif "director" in col.lower():
                                    matched_relation = "DIRECTED_BY"
                                    related_entity_type = "Person"
                                elif "genre" in col.lower() or "category" in col.lower():
                                    matched_relation = "BELONGS_TO"
                                    related_entity_type = "Genre"
                                elif "author" in col.lower() or "writer" in col.lower():
                                    matched_relation = "WRITTEN_BY"
                                    related_entity_type = "Person"
                                elif "date" in col.lower() or "year" in col.lower():
                                    # This is likely a date property, not a relation
                                    continue
                                elif "rating" in col.lower() or "score" in col.lower():
                                    # This is likely a rating property, not a relation
                                    continue
                                else:
                                    # Default relation name
                                    matched_relation = "HAS_" + col.upper()
                            
                            # Process the related entity/entities
                            related_value = str(row[col])
                            
                            # Check if it's a multi-value field (comma or pipe separated)
                            if "|" in related_value or "," in related_value:
                                separator = "|" if "|" in related_value else ","
                                related_values = [v.strip() for v in related_value.split(separator) if v.strip()]
                                
                                for related_val in related_values:
                                    if related_entity_type:
                                        all_entities.add((related_entity_type, related_val))
                                    
                                    # Add relation
                                    all_triples.append((primary_entity, matched_relation, related_val))
                                    
                                    # Add reverse relation if applicable
                                    for rel_name, rel_config in RELATIONSHIP_TYPES.items():
                                        if rel_name == matched_relation and rel_config.get("bidirectional"):
                                            all_triples.append((related_val, rel_config["reverse_name"], primary_entity))
                                            break
                            else:
                                if related_entity_type:
                                    all_entities.add((related_entity_type, related_value))
                                
                                # Add relation
                                all_triples.append((primary_entity, matched_relation, related_value))
                                
                                # Add reverse relation if applicable
                                for rel_name, rel_config in RELATIONSHIP_TYPES.items():
                                    if rel_name == matched_relation and rel_config.get("bidirectional"):
                                        all_triples.append((related_value, rel_config["reverse_name"], primary_entity))
                                        break
            else:
                # Fallback to basic built-in CSV processing for Python
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
                            
                            # Add movie entity
                            all_entities.add(("Movie", movie_title))
                            
                            # Process actors
                            if actors_idx is not None and len(row) > actors_idx and row[actors_idx]:
                                actors = row[actors_idx].split('|')
                                for actor in actors:
                                    actor = actor.strip()
                                    if actor:
                                        # Add actor entity
                                        all_entities.add(("Person", actor))
                                        # Add relation
                                        all_triples.append((actor, "ACTED_IN", movie_title))
                                        all_triples.append((movie_title, "HAS_ACTOR", actor))
                            
                            # Process director
                            if director_idx is not None and len(row) > director_idx and row[director_idx]:
                                director = row[director_idx].strip()
                                # Add director entity
                                all_entities.add(("Person", director))
                                # Add relation
                                all_triples.append((director, "DIRECTED", movie_title))
                                all_triples.append((movie_title, "DIRECTED_BY", director))
                            
                            # Process genres
                            if genres_idx is not None and len(row) > genres_idx and row[genres_idx]:
                                genres = row[genres_idx].split('|')
                                for genre in genres:
                                    genre = genre.strip()
                                    if genre:
                                        # Add genre entity
                                        all_entities.add(("Genre", genre))
                                        # Add relation
                                        all_triples.append((movie_title, "BELONGS_TO", genre))
                                        all_triples.append((genre, "CONTAINS", movie_title))
                            
                            # Process release date
                            if released_idx is not None and len(row) > released_idx and row[released_idx]:
                                released = row[released_idx]
                                all_triples.append((movie_title, "RELEASED_ON", released))
                            
                            # Process IMDb rating
                            if rating_idx is not None and len(row) > rating_idx and row[rating_idx]:
                                rating = row[rating_idx]
                                all_triples.append((movie_title, "HAS_IMDB_RATING", rating))
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
                                        if rel_col and rel_col.lower() == col_name.lower():
                                            relation_type = rel_type
                                            break
                                    
                                    # Add to triples
                                    all_triples.append((entity_val, relation_type, row[i]))
            
            # Store entity information and properties in the triples metadata
            entity_meta = []
            for (entity_type, entity_label) in all_entities:
                entity_info = {
                    "type": entity_type,
                    "label": entity_label,
                    "id": self._generate_unique_id(entity_type, entity_label)
                }
                
                # Add properties if available
                if (entity_label, entity_type) in entity_properties:
                    entity_info["properties"] = entity_properties[(entity_label, entity_type)]
                
                entity_meta.append(entity_info)
            
            # Add the enhanced triples and entity metadata
            enhanced_data = {
                "triples": all_triples,
                "entities": entity_meta
            }
            
            # Save metadata to a json file with same name as CSV
            meta_path = os.path.splitext(csv_path)[0] + "_metadata.json"
            try:
                with open(meta_path, 'w') as f:
                    json.dump(enhanced_data, f, indent=2)
                logging.info(f"Saved enhanced metadata to {meta_path}")
            except Exception as e:
                logging.warning(f"Failed to save metadata: {e}")
            
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
        """Query the knowledge graph using natural language with enhanced matching"""
        try:
            # Get graph data
            nodes, relationships = self.neo4j_manager.get_graph_data(graph_id)
            
            # Extract query intents
            query_lower = query_text.lower()
            
            # Check for specific query types
            is_entity_query = any(term in query_lower for term in ["what is", "who is", "tell me about", "information on"])
            is_relationship_query = any(term in query_lower for term in ["related to", "connected to", "relationship", "related with"])
            is_count_query = any(term in query_lower for term in ["how many", "count", "number of"])
            is_similarity_query = any(term in query_lower for term in ["similar", "like", "related", "same genre as"])
            
            # Extract entity types being asked about
            entity_types_mentioned = []
            entity_type_patterns = {
                "movie": ["movie", "film", "documentary"],
                "person": ["actor", "director", "person", "people"],
                "genre": ["genre", "category", "type"],
                "book": ["book", "novel", "publication"],
                "artist": ["artist", "band", "musician", "singer"],
                "song": ["song", "track", "music"],
                "institution": ["company", "institution", "organization", "university"]
            }
            
            for entity_type, patterns in entity_type_patterns.items():
                if any(pattern in query_lower for pattern in patterns):
                    entity_types_mentioned.append(entity_type.capitalize())
            
            # Basic entity extraction from query
            query_entities = []
            
            # Use regex patterns to extract possible entity names
            name_patterns = [
                r'"([^"]+)"',  # Quoted names
                r"'([^']+)'",  # Single-quoted names
                r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b"  # Capitalized multi-word names
            ]
            
            for pattern in name_patterns:
                matches = re.finditer(pattern, query_text)
                for match in matches:
                    query_entities.append(match.group(1))
            
            # If no entities found with patterns, use basic keyword matching
            if not query_entities:
                # Split into words and look for non-stopwords
                stopwords = ["the", "a", "an", "in", "on", "at", "by", "for", "with", "about", 
                            "is", "are", "was", "were", "be", "been", "being", "have", "has", 
                            "had", "do", "does", "did", "will", "would", "shall", "should", 
                            "may", "might", "must", "can", "could"]
                
                query_words = [word for word in re.findall(r'\b\w+\b', query_lower) 
                              if word not in stopwords and len(word) > 2]
                
                # Use longest words as potential entities (excluding common question words)
                question_words = ["who", "what", "where", "when", "why", "how", "which", "whose"]
                potential_keywords = [word for word in query_words 
                                     if word not in question_words and word not in stopwords]
                
                # Sort by length and take the top 3
                potential_keywords.sort(key=len, reverse=True)
                query_entities.extend(potential_keywords[:3])
            
            # Search for matching nodes
            matching_nodes = []
            for node in nodes:
                node_label = node.get("label", "").lower()
                
                # Check for exact matches first
                if any(entity.lower() == node_label for entity in query_entities):
                    matching_nodes.append(node)
                    continue
                
                # Then check for substring matches
                if any(entity.lower() in node_label or node_label in entity.lower() 
                       for entity in query_entities):
                    matching_nodes.append(node)
                    continue
                
                # Finally check for word-level matches
                node_words = set(re.findall(r'\b\w+\b', node_label))
                if any(set(re.findall(r'\b\w+\b', entity.lower())) & node_words 
                       for entity in query_entities if len(entity) > 2):
                    matching_nodes.append(node)
            
            # Filter by entity type if mentioned
            if entity_types_mentioned and matching_nodes:
                type_filtered_nodes = [
                    node for node in matching_nodes 
                    if node.get("type", "").capitalize() in entity_types_mentioned
                ]
                
                # Only use type filtered results if we found some
                if type_filtered_nodes:
                    matching_nodes = type_filtered_nodes
            
            # Extract relationships
            if matching_nodes:
                node_label_map = {node.get("id"): node.get("label") for node in nodes}
                node_type_map = {node.get("id"): node.get("type", "Entity") for node in nodes}
                matching_node_ids = [node.get("id") for node in matching_nodes]
                
                if is_relationship_query:
                    # Find all relationships connected to the matching nodes
                    matching_relations = []
                    for rel in relationships:
                        source_id = rel.get("source")
                        target_id = rel.get("target")
                        
                        if source_id in matching_node_ids or target_id in matching_node_ids:
                            matching_relations.append(rel)
                    
                    if matching_relations:
                        # Format the results
                        result_lines = ["Found these relationships:"]
                        
                        for rel in matching_relations:
                            source_id = rel.get("source")
                            target_id = rel.get("target") 
                            relation_type = rel.get("type")
                            
                            source_label = node_label_map.get(source_id, "Unknown")
                            target_label = node_label_map.get(target_id, "Unknown")
                            source_type = node_type_map.get(source_id, "Entity")
                            target_type = node_type_map.get(target_id, "Entity")
                            
                            result_lines.append(f"- [{source_type}] {source_label} [{relation_type}] [{target_type}] {target_label}")
                        
                        return "\n".join(result_lines)
                    else:
                        return f"Found {len(matching_nodes)} entities matching your query, but no relationships."
                        
                elif is_entity_query:
                    # Return details about the entities
                    result_lines = ["Found these entities:"]
                    
                    for node in matching_nodes:
                        node_id = node.get("id")
                        node_label = node.get("label")
                        node_type = node.get("type", "Entity")
                        
                        # Find relationships for this node
                        node_relations = []
                        for rel in relationships:
                            if rel.get("source") == node_id:
                                target_label = node_label_map.get(rel.get("target"), "Unknown")
                                node_relations.append(f"[{rel.get('type')}] {target_label}")
                            elif rel.get("target") == node_id:
                                source_label = node_label_map.get(rel.get("source"), "Unknown")
                                node_relations.append(f"[{rel.get('type')}] from {source_label}")
                        
                        # Create entity description
                        entity_desc = [f"- [{node_type}] {node_label}"]
                        if node_relations:
                            entity_desc.append("  Related to:")
                            for rel in node_relations[:5]:  # Limit to 5 relations
                                entity_desc.append(f"  * {rel}")
                            
                            if len(node_relations) > 5:
                                entity_desc.append(f"  * ...and {len(node_relations) - 5} more relations")
                                
                        result_lines.extend(entity_desc)
                        
                    return "\n".join(result_lines)
                    
                elif is_count_query:
                    # Count entities and relationships
                    if "relationship" in query_lower or "relation" in query_lower:
                        # Count relationships
                        matching_relations = []
                        for rel in relationships:
                            source_id = rel.get("source")
                            target_id = rel.get("target")
                            
                            if source_id in matching_node_ids or target_id in matching_node_ids:
                                matching_relations.append(rel)
                                
                        return f"Found {len(matching_relations)} relationships connected to {len(matching_nodes)} matched entities."
                    else:
                        # Count entities by type
                        type_counts = {}
                        for node in matching_nodes:
                            node_type = node.get("type", "Entity")
                            type_counts[node_type] = type_counts.get(node_type, 0) + 1
                            
                        result_lines = [f"Found {len(matching_nodes)} entities matching your query:"]
                        for node_type, count in type_counts.items():
                            result_lines.append(f"- {count} {node_type}(s)")
                            
                        return "\n".join(result_lines)
                        
                elif is_similarity_query:
                    # Find similar entities
                    result_lines = ["Found entities that might be similar:"]
                    
                    for node in matching_nodes:
                        node_id = node.get("id")
                        node_label = node.get("label")
                        node_type = node.get("type", "Entity")
                        
                        # Find entities with shared relationships
                        similar_entities = set()
                        for rel in relationships:
                            if rel.get("source") == node_id:
                                # Find other entities related to the same target
                                target_id = rel.get("target")
                                rel_type = rel.get("type")
                                
                                for rel2 in relationships:
                                    if (rel2.get("target") == target_id and 
                                        rel2.get("type") == rel_type and 
                                        rel2.get("source") != node_id):
                                        similar_entities.add(rel2.get("source"))
                                        
                            elif rel.get("target") == node_id:
                                # Find other entities targeted by the same source
                                source_id = rel.get("source")
                                rel_type = rel.get("type")
                                
                                for rel2 in relationships:
                                    if (rel2.get("source") == source_id and 
                                        rel2.get("type") == rel_type and 
                                        rel2.get("target") != node_id):
                                        similar_entities.add(rel2.get("target"))
                        
                        # Format the results
                        entity_desc = [f"- Entities similar to [{node_type}] {node_label}:"]
                        if similar_entities:
                            for similar_id in list(similar_entities)[:5]:  # Limit to 5 similar entities
                                similar_label = node_label_map.get(similar_id, "Unknown")
                                similar_type = node_type_map.get(similar_id, "Entity")
                                entity_desc.append(f"  * [{similar_type}] {similar_label}")
                                
                            if len(similar_entities) > 5:
                                entity_desc.append(f"  * ...and {len(similar_entities) - 5} more similar entities")
                        else:
                            entity_desc.append("  * No similar entities found")
                            
                        result_lines.extend(entity_desc)
                        
                    return "\n".join(result_lines)
                
                else:
                    # General query - show both entities and relationships
                    result_lines = [f"Found {len(matching_nodes)} entities matching your query:"]
                    
                    # Show entity details
                    for node in matching_nodes[:3]:  # Limit to 3 entities
                        node_id = node.get("id")
                        node_label = node.get("label")
                        node_type = node.get("type", "Entity")
                        
                        # Find relationships for this node
                        node_relations = []
                        for rel in relationships:
                            if rel.get("source") == node_id:
                                target_label = node_label_map.get(rel.get("target"), "Unknown")
                                node_relations.append(f"[{rel.get('type')}] {target_label}")
                            elif rel.get("target") == node_id:
                                source_label = node_label_map.get(rel.get("source"), "Unknown")
                                node_relations.append(f"[{rel.get('type')}] from {source_label}")
                        
                        # Create entity description
                        entity_desc = [f"- [{node_type}] {node_label}"]
                        if node_relations:
                            entity_desc.append("  Related to:")
                            for rel in node_relations[:3]:  # Limit to 3 relations
                                entity_desc.append(f"  * {rel}")
                            
                            if len(node_relations) > 3:
                                entity_desc.append(f"  * ...and {len(node_relations) - 3} more relations")
                                
                        result_lines.extend(entity_desc)
                    
                    if len(matching_nodes) > 3:
                        result_lines.append(f"...and {len(matching_nodes) - 3} more entities")
                        
                    return "\n".join(result_lines)
            else:
                return "No matching entities found for your query. Please try different keywords or be more specific."
                
        except Exception as e:
            logging.error(f"Error querying knowledge graph: {e}")
            error_msg = f"Failed to query knowledge graph: {str(e)}"
            logging.exception(error_msg)
            return f"Error: {error_msg}"
