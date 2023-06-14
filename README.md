<!-- # ema-take-home
EMA [take home challenge](https://docs.google.com/document/d/1J99em3zttLEwtQ9oJEOP5nVeDricSMmqE7j_ig6kVqU) -->

Conversational Natural Language Query Agent that reads [Stanford Lecture Notes](https://stanford-cs324.github.io/winter2022/lectures/) and accommodates different data types (images, text, hyperlinks, tables, PDFs) while performing vector indexing over the course/lecture content using LangChain and LlamaIndex that uses ChatGPT in the background. 

### [Design Doc](https://docs.google.com/document/d/1vRxxKQiYI0jttjN885Ij-J0IYJQMZkg46loVui3a9A4/edit?usp=sharing) for the implementation.

### Main Files
- **utils** - utility funtions
- **ema1.ipynb** - main file with output for certain queries. Only/Main file and uses utils.py
- **prev_run.ipynb** contains results of same queries for an earlier run. The code predates that of ema1.ipynb

### Other Approaches/Attempts:
- More sophisticated architecture with llama index - **llama.py**
- using langchain model - **lchain.py**
