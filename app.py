from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Initialize the LLM and chain outside the function
llm = ChatOllama(
    model="llama3.2:latest",
    temperature=0,
    seed=42,
    # other params if necessary...
)

# Define the LangChain prompt
prompt_template = ChatPromptTemplate([
    ("system", '''
        I want you to act as a song recommender. I will provide you with a song and you will create a playlist of 10 songs
        that are similar to the given song. And you will provide a playlist name and description for the playlist.
        Do not choose songs that are same name or artist. Do not write any explanations or other words,
        just reply with the playlist name, description and the songs.
    '''
    ),
    ("user", "My first song is {song}.")
])

# Create the chain
chain = prompt_template | llm

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    song = request.json.get('song')
    # Invoke the LangChain prompt with the song input
    response = chain.invoke({"song": song})
    return jsonify({"playlist": response.content})

if __name__ == '__main__':
    app.run(debug=True)
