from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from langchain_community.utilities import ArxivAPIWrapper

from pydantic import BaseModel, Field

import re
import json
from typing import List

from google.colab import userdata

HUGGINGFACE_API_KEY = userdata.get('HUGGINGFACE_API_KEY')

import os

output_dir = "./ai-agent-output"
os.makedirs(output_dir, exist_ok=True)

llm = LLM(
    model="huggingface/mistralai/Mistral-7B-Instruct-v0.3",
    api_key=HUGGINGFACE_API_KEY
)

"""### Agent A"""

class GeneratedSearchQueries(BaseModel):
    queries: List[str] = Field(description="Generated search query to be passed to search API")

search_queries_generator_agent = Agent(
    role="Search Queries Generator",
    goal="Generate concise and diverse search queries related to the user-provided topic to ensure comprehensive coverage of relevant academic papers.",
    backstory="""
        The agent is an experienced research assistant specializing in crafting precise search queries to search for academic papers.
        Familiar with keyword optimization and topic variations to maximize the retrieval of relevant results from databases and APIs.
    """,
    llm=llm,
    verbose=True
)

search_queries_generator_task = Task(
    description="""
        Based on the following user input: "{user_input}", generate a diverse and comprehensive set of {queries_no} search queries to search for academic papers about this topic.
        These search queries should be concise and broad, covering general and widely-used terms related to the topic.
    """,
    expected_output="A list of search queries",
    output_json=GeneratedSearchQueries,
    output_file=os.path.join(output_dir, "step_1_search_queries.txt"),
    agent=search_queries_generator_agent
)

"""### Agent B"""

# Output Format
class SingleSearchResult(BaseModel):
    title: str = Field(..., description="Title of the academic paper")
    authors: str = Field(..., description="Authors of the paper")
    publication_date: str = Field(description="Date of publication", default=None)
    summary: str = Field(..., description="Brief summary of the paper")
    source: str = Field(..., description="Source of the paper: the website")

class AllSearchResults(BaseModel):
    results: List[SingleSearchResult] = Field(..., description="List of search results")

# Tools
class ArxivSearchTool(BaseTool):
    name: str = "Arxiv Search Tool"
    description: str = "Search for academic papers on Arxiv with structured output"
    arxiv_search: ArxivAPIWrapper(top_k_results=2) = Field(default_factory=ArxivAPIWrapper)

    def _parse_results(self, raw_results: str) -> List[SingleSearchResult]:
        papers = raw_results.split('\n\n')
        parsed_results = []
        for paper in papers:
            date_match = re.search(r'Published: (\d{4}-\d{2}-\d{2})', paper)
            publication_date = date_match.group(1) if date_match else "N/A"
            title_match = re.search(r'Title: (.+?)(?=\nAuthors:)', paper, re.DOTALL)
            title = title_match.group(1).strip() if title_match else "N/A"
            authors_match = re.search(r'Authors: (.+?)(?=\nSummary:)', paper, re.DOTALL)
            authors = authors_match.group(1).strip() if authors_match else "N/A"
            summary_match = re.search(r'Summary: (.+)', paper, re.DOTALL)
            summary = summary_match.group(1).strip() if summary_match else "N/A"
            parsed_results.append(SingleSearchResult(
                title=title,
                authors=authors,
                publication_date=publication_date,
                summary=summary,
                source='Arxiv'
            ))
        return parsed_results

    def _run(self, query: str) -> str:
        try:
            raw_results = self.arxiv_search.run(query)
            parsed_results = self._parse_results(raw_results)
            structured_results = AllSearchResults(results=parsed_results)
            return structured_results.model_dump_json(indent=2)
        except Exception as e:
            return f"Error performing structured search on Arxiv: {str(e)}"

    def _arun(self, query: str):
        raise NotImplementedError("Async method not implemented for this tool")

search_engine_agent = Agent(
    role="Search Engine",
    goal="Retrieve relevant academic papers based on the generated search queries.",
    backstory="The agent is an experienced research assistant specializing in searching for academic papers using websites APIs.",
    llm=llm,
    verbose=True,
    tools=[ArxivSearchTool()]
)

search_engine_task = Task(
    description="""
        Given this set of search queries, use the provided API tools to search for relevant academic papers using these queries.
        Use the tools to search in academic paper websites: Arxiv, to retrieve relevant metadata about the papers.
        Return the metadata in JSON format, including details such as paper titles, abstracts, authors, publication dates, and the source of each paper.
    """,
    expected_output="A JSON object containing papers with their details",
    output_json=AllSearchResults,
    output_file=os.path.join(output_dir, "step_2_all_search_results.txt"),
    agent=search_engine_agent
)

"""### Agent C"""

# Output Format
class SelectedSearchResult(BaseModel):
    title: str = Field(..., description="Title of the academic paper")
    authors: str = Field(..., description="Authors of the paper")
    publication_date: str = Field(description="Date of publication", default=None)
    summary: str = Field(..., description="Brief summary of the paper")
    source: str = Field(..., description="Source of the paper: the website")
    reason: str = Field(..., description="Reason for selecting this paper")

class AllSelectedResults(BaseModel):
    results: List[SelectedSearchResult] = Field(..., description="List of selected search results")

selector_agent = Agent(
    role="Paper Selector",
    goal="Filter and prioritize academic papers based on relevance to the user input and the generated queries, giving preference to more recent papers.",
    backstory="The agent is an expert at evaluating academic papers for relevance and quality based on user criteria.",
    llm=llm,
    verbose=True
)

selector_task = Task(
    description="""
    Your task is to evaluate and prioritize a list of academic papers based on the following criteria:
    1. Relevance: Assess how closely the title and summary of each paper match the user's input and the generated queries. The relevance score is the most important factor in determining the paper's priority.
    2. Recency: Prefer more recent papers (e.g., papers published in the last few years) over older ones. However, recency should not outweigh relevance.
    3. Quality of Information: Ensure the paper has a meaningful and well-structured summary, as well as comprehensive details about its authors and publication date.
    Use the LLM to compute a relevance score for each paper based on the user's input and the generated queries. Combine this score with the recency of the paper to assign an overall priority ranking. Return -in order- the top {papers_no} papers that meet the criteria.
    Your output should be a structured JSON object containing the selected papers with their details: title, authors, publication date, summary, source, and the reason for its rate.
    """,
    expected_output="A JSON object containing selected papers",
    output_json=AllSelectedResults,
    output_file=os.path.join(output_dir, "step_3_chosen_papers.txt"),
    agent=selector_agent
)

"""### Agent D"""

report_maker_agent = Agent(
    role="Academic Report Author Agent",
    goal="Generate a professional HTML report summarizing the selection process, and presenting the selected papers.",
    backstory="The agent is an expert in presenting research data in a structured and visually appealing HTML format for academic purposes.",
    llm=llm,
    verbose=True
)

report_maker_task = Task(
    description="\n".join([
        "The task is to generate a professional HTML page summarizing the academic paper search and selection process.",
        "You should use Bootstrap CSS framework to create a clean and responsive UI.",
        "The report will include the search results and selected academic papers, organized in a structured and visually appealing format.",
        "The report should have the following sections:",
        "1. Executive Summary: A brief overview of the search and selection process, including key findings.",
        "2. Introduction: An introduction to the purpose and scope of the academic report.",
        "3. Methodology: A description of the search and selection process, including tools used and criteria applied.",
        "4. Results: A detailed table of the selected academic papers with these details only (title, authors, publication date, summary), you MUST insert the papers details Manually in the table (Don't use a loop)",
        "5. Analysis: An analysis of the findings, highlighting the relevance of the selected papers to the user's input and generated queries.",
        "6. Conclusion: A summary of the report and next steps for the user."
    ]),
    expected_output="A professional HTML page summarizing the academic paper search and selection process.",
    output_file=os.path.join(output_dir, "step_4_academic_report.html"),
    agent=report_maker_agent
)

"""### Agent E"""

html_reviewer_agent = Agent(
    role="HTML Code Reviewer",
    goal="Review the generated HTML report to ensure it is directly runnable, starts with <!DOCTYPE html>, ends with </html>, contains only the HTML code, and removes any markdown-style code fences like ```html```.",
    backstory="The agent specializes in validating, refining, and cleaning HTML structures for direct execution in web browsers.",
    llm=llm,
    verbose=True,
)

html_reviewer_task = Task(
    description="\n".join([
        "The task is to review the provided HTML structure of the academic report.",
        "Ensure the HTML code is complete, valid, and directly runnable in a web browser.",
        "Make sure the HTML document starts with `<!DOCTYPE html>` and ends with `</html>`.",
        "Remove any markdown-style code fences like ```html``` from the start and end of the code.",
        "Remove any additional text, comments, or code that is not part of the valid HTML structure itself.",
        "Verify that the HTML file includes properly structured <html>, <head>, and <body> tags.",
        "Check that any linked stylesheets or external resources (like Bootstrap CSS) are correctly referenced and functional.",
        "Don't add any comment after the code, you must retrieve the code only starting with <!DOCTYPE html>"
    ]),
    expected_output="A clean and valid HTML file starting with <!DOCTYPE html> and ending with </html>, ready for direct execution in a web browser, with no markdown-style code fences.",
    output_file=os.path.join(output_dir, "step_5_final_report.html"),
    agent=html_reviewer_agent,
)

"""### AI Crew Kickoff"""

crew = Crew(
    agents=[
        search_queries_generator_agent,
        search_engine_agent,
        selector_agent,
        report_maker_agent,
        html_reviewer_agent
        ],
    tasks=[
        search_queries_generator_task,
        search_engine_task,
        selector_task,
        report_maker_task,
        html_reviewer_task
        ],
    process=Process.sequential
)

crew_results = crew.kickoff(
    inputs={
        'user_input': 'Deep Learning',
        'queries_no': 4,
        'papers_no': 4
        }
    )



