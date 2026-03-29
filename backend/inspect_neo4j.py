import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv("/home/lph/Documents/GitHub/AIRAG/.env")

uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
user = os.getenv("NEO4J_USERNAME", "neo4j")
password = os.getenv("NEO4J_PASSWORD", "password")

driver = GraphDatabase.driver(uri, auth=(user, password))

def run():
    with driver.session() as session:
        print("======== NODES ========")
        result = session.run("MATCH (n) RETURN labels(n) AS labels, n.entity_id AS entity_id, n.display_name AS name LIMIT 20")
        for record in result:
            print(f"[{record['labels']}] {record['entity_id']} (Name: {record['name']})")
        
        print("\n======== RELATIONS ========")
        result = session.run("MATCH (a)-[r]->(b) RETURN labels(a)[1] as a_label, a.entity_id as a_id, type(r) as rel_type, labels(b)[1] as b_label, b.entity_id as b_id LIMIT 30")
        for record in result:
            print(f"({record['a_label']}: {record['a_id']}) -[{record['rel_type']}]-> ({record['b_label']}: {record['b_id']})")
            
        print("\n======== PART_OF RELATIONS ========")
        result = session.run("MATCH (a)-[r:PART_OF]->(b) RETURN labels(a)[1] as a_label, a.entity_id as a_id, labels(b)[1] as b_label, b.entity_id as b_id LIMIT 20")
        for record in result:
            print(f"({record['a_label']}: {record['a_id']}) -[PART_OF]-> ({record['b_label']}: {record['b_id']})")

try:
    run()
finally:
    driver.close()
