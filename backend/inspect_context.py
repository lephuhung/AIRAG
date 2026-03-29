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
        print("\n======== ALL LOCATIONS ========")
        result = session.run("MATCH (n:Location) RETURN n.entity_id AS entity_id")
        for record in result:
            print(f"Location: {record['entity_id']}")
            
        print("\n======== ALL ORGANIZATIONS ========")
        result = session.run("MATCH (n:Organization) RETURN n.entity_id AS entity_id LIMIT 10")
        for record in result:
            print(f"Organization: {record['entity_id']}")

        print("\n======== ALL DOCUMENTS ========")
        result = session.run("MATCH (n:Document) RETURN n.entity_id AS entity_id")
        for record in result:
            print(f"Document: {record['entity_id']}")

try:
    run()
finally:
    driver.close()
