# AI-Driven Academic Research Pipeline

**AI-Driven Academic Research Pipeline** is a modular, multi-agent Python application that streamlines the academic research process. Using advanced AI tools and models, this system automates the tasks of generating search queries, retrieving academic papers, selecting relevant results, and presenting findings in a professional HTML report.

---

## 🚀 Features

1. **Automated Search Query Generation**
   - Dynamically generates diverse search queries based on user input.
2. **Academic Paper Retrieval**
   - Integrates with the Arxiv API to fetch academic paper metadata.
3. **Relevance-Based Paper Selection**
   - Filters and prioritizes papers based on relevance, recency, and quality.
4. **HTML Report Generation**
   - Creates professional reports summarizing the search and selection process using Bootstrap for clean and responsive design.
5. **Code Review for HTML Output**
   - Validates and refines the final HTML report for direct execution in a web browser.

---

## 📚 How It Works

### 1. Multi-Agent Workflow
The project uses the **CrewAI** library to coordinate tasks among multiple agents:
- **Search Query Generator Agent:** Creates search queries based on user input.
- **Search Engine Agent:** Retrieves academic papers using the Arxiv API.
- **Paper Selector Agent:** Filters and prioritizes papers for relevance and quality.
- **HTML Report Maker Agent:** Generates a comprehensive academic report in HTML format.
- **HTML Reviewer Agent:** Validates and optimizes the HTML output for browser use.

### 2. Sequential Task Execution
The entire process is orchestrated through the `Crew` object, executing tasks in the following order:
1. Generate search queries.
2. Retrieve academic papers.
3. Filter and prioritize papers.
4. Create an HTML report.
5. Validate the final HTML report.

---

## 🧑‍💻 Usage

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Add your Hugging Face API key to environment variables:
```bash
HUGGINGFACE_API_KEY = your_api_key_here
```

3. Modify the input parameters in the code:
   - `user_input`: The research topic (e.g., "Deep Learning").
   - `queries_no`: Number of search queries to generate.
   - `papers_no`: Number of papers to prioritize and include in the report.

4. Run the script:
```bash
python app.py
```

5. The output includes:
   - **Search Queries:** Stored in `step_1_search_queries.txt`.
   - **Search Results:** Stored in `step_2_all_search_results.txt`.
   - **Selected Papers:** Stored in `step_3_chosen_papers.txt`.
   - **HTML Report:** Stored in `step_4_academic_report.html` and validated in `step_5_final_report.html`.

---

## 📝 Report Structure

The generated HTML report includes the following sections:
1. **Executive Summary:** Brief overview of the process and findings.
2. **Introduction:** Purpose and scope of the research.
3. **Methodology:** Tools and criteria used in the selection process.
4. **Results:** Table of selected academic papers (title, authors, date, summary, reason for choosing).
5. **Analysis:** Key insights and relevance to the input topic.
6. **Conclusion:** Summary and next steps.

---

## 🤖 Technologies Used

- **CrewAI:** Multi-agent system orchestration.
- **LangChain Community Tools:** Seamless API integration with Arxiv.
- **Hugging Face:** Mistral LLM for generating and analyzing outputs.
- **Bootstrap CSS:** Responsive and professional HTML design.

---

## 📂 Project Structure

```
academic-search-assisant/
│
├── app.py                      
├── pipeline_demo.ipynb    
├── output/                     
│   ├── step_1_search_queries.txt   
│   ├── step_2_all_search_results.txt 
│   ├── step_3_chosen_papers.txt    
│   ├── step_4_academic_report.html  
│   └── step_5_final_report.html 
├── requirements.txt             
└── README.md                    

```
