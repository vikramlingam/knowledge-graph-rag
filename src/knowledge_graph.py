import logging
from typing import List, Dict, Tuple, Set
from collections import defaultdict
import json # Added for the new get_vis_js_html

logger = logging.getLogger(__name__)

# Professional Light Blue/Cyan Palette (User Requested)
BLUE_PALETTE = [
    "#00CCFF", # Vivid Cyan
    "#3399FF", # Sky Blue
    "#00BFFF", # Deep Sky Blue
    "#00FFFF", # Aqua
    "#87CEEB", # Sky Blue Light
    "#40E0D1", # Turquoise
    "#1E90FF", # Dodger Blue
    "#ADD8E6", # Light Blue
]

# Assign colors based on entity label
def get_blue_color(label):
    hash_val = sum(ord(c) for c in label)
    return BLUE_PALETTE[hash_val % len(BLUE_PALETTE)]


def extract_entities(text: str) -> List[Dict]:
    """Extract named entities from text using spaCy."""
    try:
        import spacy
        
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text[:50000])  # Limit text length
        
        entities = []
        seen = set()
        
        for ent in doc.ents:
            # Deduplicate
            key = (ent.text.lower().strip(), ent.label_)
            if key in seen or len(ent.text.strip()) < 2:
                continue
            seen.add(key)
            
            entities.append({
                "id": len(entities),
                "label": ent.text.strip(),
                "type": ent.label_,
                "color": get_blue_color(ent.label_),
            })
        
        logger.info(f"Extracted {len(entities)} entities")
        return entities
        
    except Exception as e:
        logger.error(f"Entity extraction error: {e}")
        return []


def extract_relationships(entities: List[Dict], text: str) -> List[Dict]:
    """Extract relationships between entities using co-occurrence."""
    if len(entities) < 2:
        return []
    
    relationships = []
    text_lower = text.lower()
    
    # Find co-occurring entities (within proximity)
    entity_positions = {}
    for ent in entities:
        label_lower = ent["label"].lower()
        pos = text_lower.find(label_lower)
        if pos >= 0:
            entity_positions[ent["id"]] = pos
    
    # Create edges for entities appearing close together
    entity_ids = list(entity_positions.keys())
    for i, id1 in enumerate(entity_ids):
        for id2 in entity_ids[i+1:]:
            pos1 = entity_positions[id1]
            pos2 = entity_positions[id2]
            distance = abs(pos1 - pos2)
            
            # If within 500 chars, consider related
            if distance < 500:
                relationships.append({
                    "from": id1,
                    "to": id2,
                    "strength": max(0.3, 1 - distance / 500),
                })
    
    # Limit to top 500 strongest relationships for performance
    relationships.sort(key=lambda x: x["strength"], reverse=True)
    relationships = relationships[:500]
    
    logger.info(f"Found {len(relationships)} relationships (limited to top 500)")
    return relationships


def build_graph_data(chunks: List[Dict]) -> Dict:
    """Build complete graph data from document chunks."""
    # Combine all text
    all_text = " ".join([c.get("text", "") for c in chunks])
    
    # Extract entities
    entities = extract_entities(all_text)
    
    # Calculate entity frequency across chunks for node sizing
    # This addresses User request: "size these nodes based on number of chunks it has"
    entity_counts = {ent["id"]: 0 for ent in entities}
    
    for ent in entities:
        label_lower = ent["label"].lower()
        count = 0
        for chunk in chunks:
            if label_lower in chunk.get("text", "").lower():
                count += 1
        entity_counts[ent["id"]] = count
    
    # Extract relationships
    relationships = extract_relationships(entities, all_text)
    
    # Format for vis.js
    nodes = []
    for ent in entities:
        # Base size 10, +2 for each chunk occurrence, max cap reasonable
        freq = entity_counts.get(ent["id"], 1)
        # Logarithmic-ish scaling or linear? Linear is fine for small chunk counts.
        # User said "higher chunks - higher size".
        size = 10 + (freq * 3) 
        if size > 40: size = 40 # Cap max size
        
        nodes.append({
            "id": ent["id"],
            "label": ent["label"][:30],  
            "title": f"{ent['type']}: {ent['label']} ({freq} documents)",
            "color": ent["color"],
            "size": size,
        })
    
    edges = []
    for rel in relationships:
        edges.append({
            "from": rel["from"],
            "to": rel["to"],
            "width": 1, 
            "color": {"opacity": rel["strength"]},
        })
    
    return {"nodes": nodes, "edges": edges}


def get_vis_js_html(graph_data: Dict) -> str:
    """Generate the vis.js HTML for the knowledge graph."""
    import json
    
    nodes_json = json.dumps(graph_data.get("nodes", []))
    edges_json = json.dumps(graph_data.get("edges", []))
    
    # Return only JSON data that will be used by JavaScript
    return json.dumps({"nodes": nodes_json, "edges": edges_json})


# Alias for compatibility
build_graph = build_graph_data
