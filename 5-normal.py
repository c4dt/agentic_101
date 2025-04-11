import httpx
import os
from functools import lru_cache
import json

def cache_to_file(func):
    """Decorator to cache function results to a local file."""
    cache_file = f"{func.__name__}_cache.json"
    
    def wrapper(*args):
        # Load cache from file if it exists
        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                cache = json.load(f)
        else:
            cache = {}
        
        # Check if result is already cached
        key = str(args)
        if key in cache:
            return cache[key]
        
        # Call the function and cache the result
        result = func(*args)
        cache[key] = result
        with open(cache_file, "w") as f:
            json.dump(cache, f)
        return result
    
    return wrapper

@cache_to_file
def get_person(email: str) -> str:
    """This function returns the description of this person in the EPFL database.
    It returns the full description, without any differentiation.

    Args:
        email (str): the email of the person

    Returns:
        str: json of the person"""
        
    url = f"https://search-api.epfl.ch/api/ldap?q={email}&hl=en"
    response = httpx.get(url)
    response.raise_for_status()
    return response.json()

@cache_to_file
def get_unit(name: str) -> str:
    """This function returns the description of this unit in the EPFL database.
    It returns the full description, without any differentiation.
    For some of the units, the name needs to end in "-GE" to get the list of employees. 

    Args:
        name (str): the name of the unit

    Returns:
        str: json of the unit, including people"""

    url = f"https://search-api.epfl.ch/api/unit?q={name}&hl=en"
    response = httpx.get(url)
    response.raise_for_status()
    return response.json()

with open("emails.txt", "r") as file:
    emails = file.readlines()
    
positions = set()
rse_positions = {
    'Trainee Computer Scientist', 'ETS/HES Engineer', 'Head of Engineering', 
    'Scientific Staff Member', 'HPC Application expert', 'Principal scientist', 
    'HPC application expert', 'Research Associate', 'HPC System Manager', 
    'Technical Specialist', 'System specialist', 'Systems Engineer', 
    'Scientific Advisor', 'Senior Scientist', 'Postdoctoral Researcher', 
    'Scientist', 'Information Specialist', 'Technical Employee', 'Engineer', 
    'Research Software Engineer', 'Head of IT', 'Scientific Assistant', 
    'Executive Assistant', 'DevOps Specialist', 'Computer Scientist', 
    'Data Scientist'
}
emails_seen = set()
units_seen = set()

for email in emails[1:]:
    email = email.strip()
    results = get_person(email)
    if len(results) > 1:
        print(f"1 - Multiple entries for {email}:")
        for result in results:
            print(result["sciper"])
        continue
    
    result = results[0]
    for accred in result["accreds"]:
        path = accred["path"].split("/")
        if path[1] == "ETU":
            continue
        
        if path[-1] in units_seen:
            print("1 - Already seen", path[-1])
            continue

        units_seen.add(path[-1])
        unit = get_unit(path[-1])
        
        if not "position" in accred:
            print("1 - No position for", result["email"], accred)
            continue
        
        if not "head" in unit or not "email" in unit["head"]:
            print("1 - No head for", result["email"], accred)
            continue

        if not result["email"] in emails_seen:
            emails_seen.add(result["email"])
            print(", ".join([path[1], path[-1], unit["head"]["email"], result["email"], accred["position"]]))

        for person in unit["people"]:
            positions.add(person["position"])
            if person["position"] in rse_positions:
                if "email" in person and "position" in person and \
                    person["email"] != None and person["position"] != None:
                    if not person["email"] in emails_seen:
                        emails_seen.add(person["email"])
                        print(", ".join([path[1], path[-1], unit["head"]["email"], person["email"], person["position"]]))
                    else:
                        print("1 - Already printed", person["email"])

# print("Set of all positions:", positions)