import streamlit as st
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.output_parsers import PydanticOutputParser
import json
load_dotenv()

model = ChatMistralAI(
    model="mistral-small-2506"
)

class Movie(BaseModel):
    title: str
    release_year: Optional[int]
    genre: List[str]
    director: Optional[str]
    cast: List[str]
    rating: Optional[float]
    summary: str

parser = PydanticOutputParser(pydantic_object=Movie)

prompt = ChatPromptTemplate.from_messages([
    ("system", """
        Extract movie information from the paragraph 
        {format_instructions}
    """),
    ("human", """
        {paragraph}
    """)
])

st.set_page_config(page_title="CineGuru", page_icon="🎬", layout="centered")
st.markdown("<h1 style='text-align: center; color: #E50914;'>🎬 CineGuru</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #888888;'>The Intelligent Movie Information Extractor</h4>", unsafe_allow_html=True)
st.write("---")

st.subheader("📥 Input Movie Text")
para = st.text_area(
    label="Paste your text or paragraph below:", 
    placeholder="Example: 3 Idiots (2009) is an Indian Hindi-language coming-of-age comedy-drama film directed by Rajkumar Hirani...",
    height=200
)

if st.button("Extract Movie Insights ✨", type="primary"):
    if para.strip() == "":
        st.warning("Please enter a paragraph first!")
    else:
        with st.spinner("Analyzing text and extracting details..."):
            try:

                final_prompt = prompt.invoke({
                    "paragraph": para,
                    "format_instructions": parser.get_format_instructions()
                })
                
                response = model.invoke(final_prompt)
              
                parsed_data = parser.parse(response.content)
                
                st.success("Extraction Complete!")
                st.subheader(f"🎥 {parsed_data.title}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(label="📆 Release Year", value=str(parsed_data.release_year) if parsed_data.release_year else "N/A")
                with col2:
                    st.metric(label="⭐ Rating", value=str(parsed_data.rating) if parsed_data.rating else "N/A")
                with col3:
                    st.metric(label="🎬 Director", value=parsed_data.director if parsed_data.director else "N/A")
                
                st.write("---")
                
                st.write(f"**🎭 Genres:** {', '.join(parsed_data.genre)}")
                st.write(f"**👥 Cast Members:** {', '.join(parsed_data.cast)}")
                
                st.info(f"**📝 Summary:** \n{parsed_data.summary}")
              
                with st.expander("View Raw Output Verification"):
                    st.json(response.content)
                    
            except Exception as e:
                st.error(f"An error occurred during extraction: {e}")