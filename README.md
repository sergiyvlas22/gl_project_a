# Installing dependencies(debian):
sudo apt install python3 python3-venv python3-pip
# ollama
curl -fsSL https://ollama.com/install.sh | sh
# we are using gemma:2b (8gb ram)
ollama pull gemma:2b
# install streamlit
pip install streamlit
# install chromadb
pip install chromadb
# launch streamlit project through console
cd ~/yourfoldername/gl_project

source .venv/bin/activate

streamlit run app.py
