from dotenv import load_dotenv
import os
import glob
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from langchain.vectorstores import Pinecone as LC_Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain, RetrievalQA
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.retrievers import MultiQueryRetriever
from ddgs import DDGS
from langchain.agents import initialize_agent, Tool, AgentType
from langsmith import Client
from langchain.callbacks import LangChainTracer
from langchain.schema import Document
from langchain.memory import ConversationBufferMemory
import streamlit as st
import uuid
import json

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "barcelona-kids"
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")

CACHE_FILE = "processed_cache.json"
client = Client(api_key=LANGSMITH_API_KEY)
tracer = LangChainTracer(client=client, project_name="Barcelona-Kids")

myLlmAt0 = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model="gpt-4o", 
    temperature=0,
    streaming=True,
    callbacks=[tracer]  
)
# ------------------ FUNÇÕES ------------------

def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "videos" in data and "sites" in data:
                    return data
        except (json.JSONDecodeError, ValueError):
            pass 
    return {"videos": [], "sites": []}

def save_cache(cache):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

def parse_vtt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    text_lines = []
    for line in lines:
        line = line.strip()
        if line and not line[0].isdigit() and "-->" not in line:
            text_lines.append(line)
    return " ".join(text_lines)

def prepare_documents(vtt_folder, sites):
    """Lê VTT e sites, retorna dataframe sem reprocessar já processados"""
    cache = load_cache()
    
    vtt_files = glob.glob(os.path.join(vtt_folder, "*.vtt"))
    video_data = []
    for file in vtt_files:
        title = os.path.splitext(os.path.basename(file))[0]
        if title in cache["videos"]:
            continue  
        content = parse_vtt(file)
        video_data.append({"title": title, "content": content, "source": "YouTube"})
        cache["videos"].append(title)

    df_videos = pd.DataFrame(video_data)
    print(f"{len(df_videos)} vídeos novos processados.")

    site_data = []
    for url in sites:
        if url in cache["sites"]:
            continue 
        r = requests.get(url)
        soup = BeautifulSoup(r.text, "html.parser")
        paragraphs = soup.find_all("p")
        content = " ".join([p.get_text() for p in paragraphs])
        title_tag = soup.find("title")
        title = title_tag.get_text() if title_tag else url
        site_data.append({"title": title, "content": content, "source": url})
        cache["sites"].append(url) 

    df_sites = pd.DataFrame(site_data)
    print(f"{len(df_sites)} sites novos processados.")

    df_all = pd.concat([df_videos, df_sites], ignore_index=True)

    if df_all.empty:
        df_all = pd.DataFrame(columns=["title", "content", "source"])
    else:
        if "content" not in df_all.columns:
            df_all["content"] = ""
        else:
            df_all["content"] = df_all["content"].fillna("")

    save_cache(cache)

    return df_all

def prepare_vectorstore(df_all, model_name="paraphrase-multilingual-MiniLM-L12-v2"):
    """Cria embeddings e inicializa Pinecone e retrievers"""
    embedder = SentenceTransformer(model_name)
    embeddings = embedder.encode(df_all["content"].tolist(), show_progress_bar=True, convert_to_numpy=True)
    df_all["embedding"] = list(embeddings)

    pc = Pinecone(api_key=PINECONE_API_KEY)
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    index = pc.Index(INDEX_NAME)
    to_upsert = []
    for i, row in df_all.iterrows():
        vec_id = str(i)
        embedding = row["embedding"].tolist()
        metadata = {
            "title": row.get("title", ""),
            "source": row.get("source", ""),
            "content": row.get("content", "")[:200]
        }
        to_upsert.append((vec_id, embedding, metadata))
        
    if len(to_upsert) == 0:
        print("Nenhum vetor novo para upsert. Pulando envio para Pinecone.")
    else:
        index.upsert(vectors=to_upsert, namespace="default")
        
    print(f"{len(to_upsert)} embeddings inseridos no Pinecone!")

    llm_embeddings = SentenceTransformerEmbeddings(model_name=model_name)
    vectorstore = LC_Pinecone.from_existing_index(
        index_name=INDEX_NAME,
        embedding=llm_embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    multiquery_retriever = MultiQueryRetriever.from_llm(
        retriever=retriever,
        llm=myLlmAt0
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=myLlmAt0,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": qa_prompt},
        return_source_documents=False
    )
    return vectorstore, llm_embeddings, retriever, multiquery_retriever, qa_chain

# ------------------ PROMPTS ------------------

multiquery_prompt_template = """
Tu és um especialista em viagens com crianças.
Recebeste esta pergunta do utilizador: "{pergunta}"

Divide-a em sub-queries úteis e específicas que ajudem a procurar respostas nos documentos.
Cria apenas as sub-queries relevantes para a pergunta, usa diferentes termos para maximizar as chances de encontrar uma resposta relevante, não acrescentes categorias extra.

Retorna cada sub-query numerada e clara.
"""
multiquery_prompt = PromptTemplate(template=multiquery_prompt_template, input_variables=["pergunta"])
multiquery_chain = LLMChain(
    llm=myLlmAt0,
    prompt=multiquery_prompt
)

qa_prompt_template = """
Tu és um assistente especializado em viagens para famílias com crianças.
Responde apenas à pergunta feita, usando exclusivamente a informação dos documentos abaixo.
Não cries listas fixas nem acrescentes tópicos extra se não forem pedidos.
Se não encontrares a resposta, diz claramente: "Não encontrei essa informação nos documentos."

Documentos recuperados:
{context}

Pergunta: {question}

Resposta:
"""
qa_prompt = PromptTemplate(input_variables=["context", "question"], template=qa_prompt_template)

agent_prompt = PromptTemplate(
    input_variables=["question", "agent_scratchpad"],
    template="""
És um agente de viagens especializado em famílias com crianças.

Tens acesso a ferramentas que te ajudam a responder:
- Base de dados local (Pinecone retriever)
- DuckDuckGo Search (DDGS) para pesquisar na web

A tua tarefa é:
1. Interpretar as perguntas do usuario filtrando apenas as informaçoes necessarias.
2. Sugerir atividades adequadas para crianças dessa idade na cidade indicada.
3. Indicar o que deve ser preparado/levado para a viagem.
4. Indicar o que pode ser alugado na cidade e onde.
5. Listar restaurantes ou locais adaptados a famílias.
6. Reforça items de primeira necessidade que sao comuns (por exemplo, levar protetor solar no verao). 

Regras importantes:
- Usa sempre as ferramentas para obter informação.
- Resume e organiza os resultados de forma clara e prática.
- Se não encontrares resultados relevantes, diz claramente que não há informação na base de dados. Complementa a informação com senso comum. 
- Usa perguntas para confirmar que o usuário tem os items necessarios e continue o planeamento da viagem. 
- usa optimizações especificas para o motor de pequisa duckduck go (escreve perguntas em inglês, usa palavras chave). Lê a resposta com atencao e adapta o teu próximo passo.
- Verifica sempre a tua propria base de dados antes de fazer uma pesquisa
- Responde semore em português

follow the format                                                

Question:input
Thought: I should think about this step by step
Action: [action to take]
Action Input: [input to the action] Observation: [result of the action] ... (repeat Thought/Action/Action Input/Observation as needed) 
Final Answer: [final answer to the question] 

begin!                                              

Pergunta do utilizador: {question}
{agent_scratchpad}                                            
"""
)

# ------------------ AGENT FUNÇÕES ------------------

def run_multiquery_prompt(pergunta, qa_chain, retriever):
    subqueries_text = multiquery_chain.run({"pergunta": pergunta})
    subqueries = [line.split(":",1)[1].strip() for line in subqueries_text.split("\n") if ":" in line]
    respostas = []
    encontrou_resposta = False
    for sq in subqueries:
        docs = retriever.get_relevant_documents(sq)
        context = "\n".join([d.page_content for d in docs])
        resposta = qa_chain.run({"context": context, "question": sq})
        respostas.append(f"➡️ Pergunta: {sq}\n{resposta}")
        if "Não encontrei" not in resposta:
            encontrou_resposta = True
    return "\n\n".join(respostas), encontrou_resposta

def search_ddgs_categorized(query: str, categoria: str, max_results: int = 10):
    ddgs = DDGS()
    query_modificada = f"{query} Barcelona Espanha"
    resultados = list(ddgs.text(query_modificada, max_results=max_results))
    categoria_keywords = {
        "atividades": [
            "parque", "atividade", "atração", "kids", "evento", "diversão", "coisas para fazer",
            "museum", "playground", "jogo", "show", "teatro", "espetáculo", "cultura", "familiar",
            "gratuito", "passeio", "caminhada", "outdoor", "interativo"
        ],
        "restaurantes": [
            "restaurante", "menu infantil", "comida", "food", "dining", "eat", "cafeteria",
            "family friendly", "criança", "kids menu", "play area", "buffet", "temático", "brincar",
            "parque infantil", "terraço"
        ],
        "o_que_levar": [
            "lista", "o que levar", "essenciais", "viagem", "criança", "bagagem", "checklist",
            "roupa", "protetor solar", "chapéu", "sapatos confortáveis", "lanches", "água",
            "documentos", "brinquedos", "fraldas", "medicamentos", "berço portátil", "carrinho"
        ],
        "o_que_alugar": [
            "aluguer", "rent", "rental", "equipamento", "carrinho de bebé", "cadeira auto", "gear",
            "bicicleta", "scooter", "brinquedos", "cadeira de praia", "berço", "carro", "equipamento infantil"
        ]
    }
    keywords = categoria_keywords.get(categoria, [])
    resultados_filtrados = [
        r for r in resultados
        if ("barcelona" in r["title"].lower() or "barcelona" in r["body"].lower())
        and any(k.lower() in (r["title"] + r["body"]).lower() for k in keywords)
    ]
    return resultados_filtrados

def summarize_ddgs_results(results):
    if not results:
        return "⚠️ Nenhum resultado relevante encontrado."
    textos = results
    prompt_template = ChatPromptTemplate.from_template("""
    Com base nestes resultados web, dá uma resposta curta e estruturada para o utilizador,
    focando apenas em Barcelona, Espanha. Não acrescentes tópicos extra:

    {textos}
    """)
    llm_chain = LLMChain(
        llm=myLlmAt0,
        prompt=prompt_template
    )
    return llm_chain.invoke({"textos": textos})

def add_ddgs_results_to_vectorstore(results, vectorstore):
    vectorstore.add_texts(results)
    print(f"✅ search results adicionados ao vectorstore a partir do DuckDuckGo.")

def duckduckgo_tool(query: str, vectorstore):
    with DDGS() as ddgs:
        results = ""
        for result in ddgs.text(query, max_results=30, backend="google"):
            results = f'{results}, {result["title"]}, {result["body"]}'
    if results:
        add_ddgs_results_to_vectorstore(results, vectorstore)
        return summarize_ddgs_results(results)
    return "⚠️ Nenhum resultado relevante encontrado."

def multiquery_qa_tool(query: str, multiquery_retriever):
    docs = multiquery_retriever.invoke(query)
    if not docs:
        return "Não encontrei informação relevante nos documentos."
    context = "\n\n".join([d.page_content for d in docs])
    llm = myLlmAt0
    chain = LLMChain(llm=llm, prompt=qa_prompt)
    return chain.invoke({"context": context, "question": query})["text"]

# ------------------ Leitura de ficheiros de video e sites ------------------

vtt_folder = r"C:\Users\megap\Desktop\Final Project\Scraps"
sites = [
    "https://www.tripadvisor.com/Restaurants-g187497-zfp5-Barcelona_Catalonia.html",
    "https://annetravelfoodie.com/barcelona-with-kids/",
    "https://bedfordbalabusta.com/2023/02/23/barcelona-with-two-tweens-and-a-toddler/",
    "https://www.thefork.co.uk/restaurants/barcelona-c41710/kid-friendly-t1460"
]

df_all = prepare_documents(vtt_folder, sites)
vectorstore, llm_embeddings, retriever, multiquery_retriever, qa_chain = prepare_vectorstore(df_all)

# ------------------ SETUP ------------------

tools = [
    Tool(
        name="Local RAG QA",
        func=lambda q: qa_chain.invoke({"query": q})["result"],
        description="Responde com base na base local (YouTube + sites + conteúdos web já aprendidos)."
    ),
    Tool(
        name="MultiQuery RAG QA",
        func=lambda q: multiquery_qa_tool(q, multiquery_retriever),
        description="Expande a pergunta em várias queries e responde usando os documentos recuperados."
    ),
    Tool(
        name="DuckDuckGo + Learn",
        func=lambda q: duckduckgo_tool(q, vectorstore),
        description="Pesquisa na web e incorpora conteúdos relevantes ao vectorstore antes de responder."
    )
]

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent = initialize_agent(
    tools,
    myLlmAt0,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose=True,
    max_iterations=8,
    callbacks=[tracer],
    memory=memory,
    agent_kwargs={"prompt": agent_prompt}
)

# ------------------ STREAMLIT ------------------

st.set_page_config(page_title="Travel Kids Bot", page_icon="🧳")
st.title("🧒👶 Travel Assistant for Kids")


cidade = st.text_input("Cidade:", "Barcelona")
idade = st.number_input("Idade da criança:", min_value=0, max_value=18, value=5)
epoca = st.selectbox("Época do ano:", ["Primavera", "Verão", "Outono", "Inverno"])

if st.button("Gerar recomendações"):
    pergunta_usuario = f"Planeamento de viagem para {cidade}, criança de {idade} anos no {epoca}"
    resposta = agent.invoke({"input": pergunta_usuario})["output"]

    # Show assistant response
    with st.chat_message("assistant"):
        with st.spinner("A pensar..."):
            st.markdown(resposta)

    # Save assistant response
    st.session_state.messages.append({"role": "assistant", "content": resposta})

# Store conversation in Streamlit session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input box
if prompt := st.chat_input("Escreva a sua mensagem..."):
    # Save user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate GPT response
    with st.chat_message("assistant"):
        with st.spinner("A pensar..."):
            response = myLlmAt0.invoke(st.session_state.messages)
            reply = response.content
            st.markdown(reply)

    # Save assistant message
    st.session_state.messages.append({"role": "assistant", "content": reply})
