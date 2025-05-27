import ast

def parse_genres(genre_str):
    try:
        genres = ast.literal_eval(genre_str)
        return [genre['name'] for genre in genres if 'name' in genre]
    except:
        return []
