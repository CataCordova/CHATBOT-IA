import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.llms import HuggingFaceHub
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
import pandas as pd
from PIL import Image
import os
import pickle
from prompts import * 
import prompts as prm
import re
from urllib.parse import urlparse
import torch
#from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from flair.embeddings import TransformerDocumentEmbeddings
from transformers import FalconModel, FalconConfig
from transformers import AutoTokenizer, pipeline
from langchain import HuggingFacePipeline
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings

