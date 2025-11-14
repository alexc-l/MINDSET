# constants.py
all_process_desc = [
    {
        "Process_name": "Extraverted Sensing",
        "Abbreviation": "Se",
        "Description": "Acts on concrete data in the here and now. Likes to experience the world—active, talkative, and social. Trusts the present, what is tangible and real. Keyword: Experiencing.",
        "Neg_desc": "Overindulgence, hyperactive, overly talkative"
    },
    {
        "Process_name": "Introverted Sensing",
        "Abbreviation": "Si",
        "Description": "Compares present facts and situations to past experience. Excellent recall for specific details. Trusts and remembers the past. Stores sensory data that is important to them for future use. Keyword: Remembering.",
        "Neg_desc": "Dogmatic, obsess about unimportant data, withdraw"
    },
    {
        "Process_name": "Extraverted Intuition",
        "Abbreviation": "Ne",
        "Description": "Sees possibilities in the external world. Enthusiastic and enjoys networking. Trusts the big picture, forms patterns and connections, which can then be shared with others. Keyword: Brainstorming.",
        "Neg_desc": "Over the top, swamped with options, change for the sake of change"
    },
    {
        "Process_name": "Introverted Intuition",
        "Abbreviation": "Ni",
        "Description": "Sees the big picture and underlying patterns. Focuses on future implications and abstract meanings. Trusts insights and hunches. Keyword: Visioning.",
        "Neg_desc": "Unrealistic visions, only accept data that supports their theories, make things overcomplicated"
    },
    {
        "Process_name": "Extraverted Thinking",
        "Abbreviation": "Te",
        "Description": "Seeks logic and consistency in the outside world. Concern for external laws and rules. Logical, analytical decision makers who organize the environment to achieve goals. Keyword: Organizing.",
        "Neg_desc": "Detached, cold, overly rational, critique lack of logic in others"
    },
    {
        "Process_name": "Introverted Thinking",
        "Abbreviation": "Ti",
        "Description": "Seeks internal consistency and logic of ideas. Trusts internal framework, which may be difficult to explain to others. Experience a depth of concentration that is objective and analytical. Keyword: Analyzing.",
        "Neg_desc": "Obsessive search for the truth, detached, look only at cons, driven like a machine out of control"
    },
    {
        "Process_name": "Extraverted Feeling",
        "Abbreviation": "Fe",
        "Description": "Seeks harmony with and between people in the outside world. Interpersonal and cultural values are important. Encouraging and interested in others. Keyword: Harmonizing.",
        "Neg_desc": "Insistent they know what is best for everyone, intrusive, ignore problems, force superficial harmony"
    },
    {
        "Process_name": "Introverted Feeling",
        "Abbreviation": "Fi",
        "Description": "Seeks harmony of action and thoughts with personal values. May not always articulate those values. Empathetic, sensitive, and idealistic. Keyword: Valuing.",
        "Neg_desc": "Carry the weight of the world on their shoulders, hypersensitive, pompous, feel sorry for themselves"
    }
]

# Fast lookup: Process_name → dict
PROCESS_DESC_LOOKUP = {item["Abbreviation"]: item for item in all_process_desc}