from flask import Flask, request, jsonify
import subprocess
import os
import json
from datetime import datetime
import sqlite3
import requests
from bs4 import BeautifulSoup
import csv
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Helper function to run shell commands
def run_command(command):
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"Command failed: {result.stderr}")
    return result.stdout

# Helper function to call the LLM via AI Proxy
def call_llm(prompt):
    try:
        headers = {
            "Authorization": f"Bearer {os.environ['AIPROXY_TOKEN']}",
            "Content-Type": "application/json",
        }
        data = {"prompt": prompt, "max_tokens": 50}
        response = requests.post(
            "https://api.aiproxy.io/v1/completions", headers=headers, json=data
        )
        response.raise_for_status()
        return response.json()["choices"][0]["text"].strip()
    except Exception as e:
        raise Exception(f"LLM API error: {e}")

# Task Handlers
def task_a1(email):
    try:
        run_command("pip install uv")
        run_command("curl -O https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py")
        run_command(f"python datagen.py {email}")
        return True
    except Exception as e:
        print(f"Error in Task A1: {e}")
        return False

def task_a2():
    try:
        run_command("npx prettier@3.4.2 --write /data/format.md")
        return True
    except Exception as e:
        print(f"Error in Task A2: {e}")
        return False

def task_a3():
    try:
        with open("/data/dates.txt", "r") as file:
            dates = file.readlines()
        wednesdays = 0
        for date_str in dates:
            date = datetime.strptime(date_str.strip(), "%Y-%m-%d")
            if date.weekday() == 2:  # Wednesday
                wednesdays += 1
        with open("/data/dates-wednesdays.txt", "w") as file:
            file.write(str(wednesdays))
        return True
    except Exception as e:
        print(f"Error in Task A3: {e}")
        return False

def task_a4():
    try:
        with open("/data/contacts.json", "r") as file:
            contacts = json.load(file)
        contacts_sorted = sorted(contacts, key=lambda x: (x["last_name"], x["first_name"]))
        with open("/data/contacts-sorted.json", "w") as file:
            json.dump(contacts_sorted, file, indent=2)
        return True
    except Exception as e:
        print(f"Error in Task A4: {e}")
        return False

def task_a5():
    try:
        log_files = sorted(
            [f for f in os.listdir("/data/logs") if f.endswith(".log")],
            key=lambda f: os.path.getmtime(os.path.join("/data/logs", f)),
            reverse=True,
        )[:10]
        with open("/data/logs-recent.txt", "w") as output_file:
            for log_file in log_files:
                with open(os.path.join("/data/logs", log_file), "r") as input_file:
                    first_line = input_file.readline()
                    output_file.write(first_line)
        return True
    except Exception as e:
        print(f"Error in Task A5: {e}")
        return False

def task_a6():
    try:
        index = {}
        for root, _, files in os.walk("/data/docs"):
            for file in files:
                if file.endswith(".md"):
                    file_path = os.path.join(root, file)
                    with open(file_path, "r") as f:
                        for line in f:
                            if line.startswith("# "):
                                index[os.path.relpath(file_path, "/data/docs")] = line.strip("# ").strip()
                                break
        with open("/data/docs/index.json", "w") as f:
            json.dump(index, f, indent=2)
        return True
    except Exception as e:
        print(f"Error in Task A6: {e}")
        return False

def task_a7():
    try:
        with open("/data/email.txt", "r") as file:
            email_content = file.read()
        sender_email = call_llm(f"Extract the sender's email address from this email:\n{email_content}")
        with open("/data/email-sender.txt", "w") as file:
            file.write(sender_email)
        return True
    except Exception as e:
        print(f"Error in Task A7: {e}")
        return False

def task_a8():
    try:
        # Simulate LLM call for image processing (replace with actual logic)
        card_number = call_llm("Extract the credit card number from this image: /data/credit-card.png")
        with open("/data/credit-card.txt", "w") as file:
            file.write(card_number.replace(" ", ""))
        return True
    except Exception as e:
        print(f"Error in Task A8: {e}")
        return False

def task_a9():
    try:
        with open("/data/comments.txt", "r") as file:
            comments = file.readlines()
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(comments)
        similarity_matrix = cosine_similarity(tfidf_matrix)
        np.fill_diagonal(similarity_matrix, -1)  # Ignore self-similarity
        most_similar_pair_indices = np.unravel_index(np.argmax(similarity_matrix), similarity_matrix.shape)
        most_similar_pair = [comments[i].strip() for i in most_similar_pair_indices]
        with open("/data/comments-similar.txt", "w") as file:
            file.write("\n".join(most_similar_pair))
        return True
    except Exception as e:
        print(f"Error in Task A9: {e}")
        return False

def task_a10():
    try:
        conn = sqlite3.connect("/data/ticket-sales.db")
        cursor = conn.cursor()
        cursor.execute("SELECT SUM(units * price) FROM tickets WHERE type = 'Gold'")
        total_sales = cursor.fetchone()[0]
        conn.close()
        with open("/data/ticket-sales-gold.txt", "w") as file:
            file.write(str(total_sales))
        return True
    except Exception as e:
        print(f"Error in Task A10: {e}")
        return False

# Phase B Task Handlers
def task_b3(api_url):
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        with open("/data/api-data.json", "w") as file:
            json.dump(response.json(), file)
        return True
    except Exception as e:
        print(f"Error in Task B3: {e}")
        return False

def task_b4(repo_url):
    try:
        run_command(f"git clone {repo_url} /data/repo")
        run_command("cd /data/repo && git commit --allow-empty -m 'Initial commit'")
        return True
    except Exception as e:
        print(f"Error in Task B4: {e}")
        return False

# Add other Phase B task handlers here (B5-B10)

@app.route("/run", methods=["POST"])
def run_task():
    task_description = request.args.get("task")
    if not task_description:
        return jsonify({"error": "Task description is required"}), 400

    # Parse task description and call the appropriate handler
    if "install uv and run datagen.py" in task_description:
        email = "user@example.com"  # Replace with actual email extraction logic
        success = task_a1(email)
    elif "format format.md using prettier" in task_description:
        success = task_a2()
    elif "count Wednesdays in dates.txt" in task_description:
        success = task_a3()
    elif "sort contacts.json by last_name" in task_description:
        success = task_a4()
    elif "write first line of 10 most recent .log files" in task_description:
        success = task_a5()
    elif "create index of Markdown files" in task_description:
        success = task_a6()
    elif "extract sender's email address" in task_description:
        success = task_a7()
    elif "extract credit card number from image" in task_description:
        success = task_a8()
    elif "find most similar pair of comments" in task_description:
        success = task_a9()
    elif "calculate total sales for Gold ticket type" in task_description:
        success = task_a10()
    elif "fetch data from API" in task_description:
        api_url = "https://api.example.com/data"  # Replace with actual API URL
        success = task_b3(api_url)
    elif "clone git repo and make a commit" in task_description:
        repo_url = "https://github.com/example/repo.git"  # Replace with actual repo URL
        success = task_b4(repo_url)
    else:
        return jsonify({"error": "Task not recognized"}), 400

    if success:
        return jsonify({"message": "Task executed successfully"}), 200
    else:
        return jsonify({"error": "Task execution failed"}), 500

@app.route("/read", methods=["GET"])
def read_file():
    file_path = request.args.get("path")
    if not file_path:
        return jsonify({"error": "File path is required"}), 400

    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404

    with open(file_path, "r") as file:
        content = file.read()

    return content, 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
