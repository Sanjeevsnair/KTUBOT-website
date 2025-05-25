from google import genai

client = genai.Client(api_key="AIzaSyDiTLuSzvhRqTAC4WuTO_On8LdUoxtmL5Q")

paragraph= """• Von Neumann Architecture 

It has a single memory storage to hold both program instructions and data, i.e., common program and data space. 
The CPU can either read an instruction or data from the memory one at a time (or write data to memory) because instructions and data are accessed using same bus system. 
The Von Neumann Architecture is named after the mathematician and computer scientist John Von Neumann. 
The advantage of Von Neumann architecture is simple design of microcontroller chip because only 
one memory is to be implemented which in turn reduces required hardware. 
The disadvantage is slower execution of a program. 
It is also referred as Princeton architecture as it was developed at Princeton University. 
Motorola 68HC11 microco 
5.2 Harvard Architecture 
tecture. """

prompt = f"""
You are an expert academic editor for engineering content. Refine the following paragraph to make it accurate, easy to understand, and useful for students at any level.

Please apply the following steps internally:
1. Improve sentence structure and logical flow.
2. Use a clear and formal academic tone.
3. Correct all grammar and technical mistakes.
4. Remove irrelevant or incomplete sentences.
5. Simplify complex phrasing without losing important meaning.
6. Keep all essential technical concepts and definitions intact.

**Do not show the steps. Just return the final refined version in a clean, bullet-point format that is easy for all students to understand.**

Paragraph to refine:
\"\"\"{paragraph}\"\"\"

Only return the refined content as bullet points. Do not add explanations or extra comments.
"""


response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=prompt,
)

print(response.text)