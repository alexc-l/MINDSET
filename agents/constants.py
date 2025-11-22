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

# constants.py — Big Five descriptions with stress modulation
all_trait_desc = [
    # High poles
    {
        "Trait_name": "Openness (high)",
        "Level": "high",
        "Description": "Imaginative, curious, open to new experiences, appreciates art and abstract ideas.",
        "Stressed_desc": "Scattered, eccentric, detached from reality, prone to bizarre ideas or conspiracy thinking."
    },
    {
        "Trait_name": "Conscientiousness (high)",
        "Level": "high",
        "Description": "Organized, responsible, self-disciplined, dutiful, and goal-oriented.",
        "Stressed_desc": "Perfectionistic, compulsive, workaholic, rigid, unable to relax or delegate."
    },
    {
        "Trait_name": "Extraversion (high)",
        "Level": "high",
        "Description": "Outgoing, energetic, assertive, enjoys social stimulation and being the center of attention.",
        "Stressed_desc": "Attention-seeking, domineering, reckless, talks over others, cannot tolerate solitude."
    },
    {
        "Trait_name": "Agreeableness (high)",
        "Level": "high",
        "Description": "Compassionate, cooperative, trusting, modest, and altruistic.",
        "Stressed_desc": "People-pleasing, self-sacrificing, conflict-avoidant to a fault, suppresses own needs."
    },
    {
        "Trait_name": "Neuroticism (high)",
        "Level": "high",
        "Description": "Emotionally reactive, experiences intense feelings, sensitive to stress.",
        "Stressed_desc": "Overwhelmed, catastrophizing, panic-prone, emotionally flooded, paranoid."
    },

    # Low poles
    {
        "Trait_name": "Openness (low)",
        "Level": "low",
        "Description": "Conventional, prefers routine, practical, down-to-earth, resistant to change.",
        "Stressed_desc": "Dogmatic, authoritarian, fearful rejection of novelty, closed-minded rigidity."
    },
    {
        "Trait_name": "Conscientiousness (low)",
        "Level": "low",
        "Description": "Flexible, spontaneous, carefree, dislikes strict schedules.",
        "Stressed_desc": "Irresponsible, chaotic, self-sabotaging, avoids all obligations."
    },
    {
        "Trait_name": "Extraversion (low)",
        "Level": "low",
        "Description": "Reserved, reflective, enjoys solitude, low need for external stimulation.",
        "Stressed_desc": "Socially withdrawn, paralyzed in groups, extreme isolation, depressive shutdown."
    },
    {
        "Trait_name": "Agreeableness (low)",
        "Level": "low",
        "Description": "Competitive, skeptical, direct, prioritizes own interests.",
        "Stressed_desc": "Hostile, manipulative, vindictive, sees others as threats."
    },
    {
        "Trait_name": "Neuroticism (low)",
        "Level": "low",
        "Description": "Emotionally stable, calm under pressure, resilient, rarely upset.",
        "Stressed_desc": "Emotionally flat, dismissive of others’ suffering, reckless risk-taking due to lack of anxiety."
    },
]

# Fast lookup: "openness" → high/low entry based on predicted level
BIGFIVE_DESC_LOOKUP = {
    item["Trait_name"].split(" (")[0]: item for item in all_trait_desc
}
