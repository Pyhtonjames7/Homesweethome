import math
import time
import random
import sys
import os
import json
from typing import List, Tuple, Callable, Dict, Any
from dataclasses import dataclass, asdict
from threading import Thread
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

@dataclass
class Color:
    r: int
    g: int
    b: int
    a: int = 255

    def __str__(self):
        return f"#{self.r:02x}{self.g:02x}{self.b:02x}{self.a:02x}"

    @staticmethod
    def lerp(c1: 'Color', c2: 'Color', t: float) -> 'Color':
        return Color(
            int(c1.r + (c2.r - c1.r) * t),
            int(c1.g + (c2.g - c1.g) * t),
            int(c1.b + (c2.b - c1.b) * t),
            int(c1.a + (c2.a - c1.a) * t)
        )

class Canvas:
    # ... (Canvas class implementation remains the same)

class AnimatedValue:
    # ... (AnimatedValue class implementation remains the same)

@dataclass
class Task:
    name: str
    description: str
    estimated_time: int  # in minutes
    priority: int  # 1 (highest) to 5 (lowest)
    completed: bool = False
    vector: np.ndarray = None

class VectorMemory:
    def __init__(self, dim: int = 100):
        self.dim = dim
        self.vectorizer = TfidfVectorizer(max_features=dim)
        self.vectors: List[np.ndarray] = []
        self.data: List[Any] = []

    def add(self, text: str, data: Any):
        if not self.vectors:
            self.vectorizer.fit([text])
        vector = self.vectorizer.transform([text]).toarray()[0]
        self.vectors.append(vector)
        self.data.append(data)

    def search(self, query: str, k: int = 5) -> List[Tuple[Any, float]]:
        query_vector = self.vectorizer.transform([query]).toarray()[0]
        similarities = cosine_similarity([query_vector], self.vectors)[0]
        top_k = sorted(zip(self.data, similarities), key=lambda x: x[1], reverse=True)[:k]
        return top_k

    def cluster(self, n_clusters: int = 5) -> Dict[int, List[Any]]:
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(self.vectors)
        clusters = {i: [] for i in range(n_clusters)}
        for data, label in zip(self.data, kmeans.labels_):
            clusters[label].append(data)
        return clusters

class LongTermMemory:
    def __init__(self, filename: str = "long_term_memory.json"):
        self.filename = filename
        self.data: Dict[str, Any] = self.load()

    def load(self) -> Dict[str, Any]:
        try:
            with open(self.filename, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def save(self):
        with open(self.filename, "w") as f:
            json.dump(self.data, f)

    def add(self, key: str, value: Any):
        self.data[key] = value
        self.save()

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def update(self, key: str, value: Any):
        if key in self.data:
            self.data[key] = value
            self.save()

class ActiveLearner:
    def __init__(self, vector_memory: VectorMemory):
        self.vector_memory = vector_memory
        self.model = None

    def train(self):
        X = np.array(self.vector_memory.vectors)
        y = np.array([task.priority for task in self.vector_memory.data])
        self.model = KMeans(n_clusters=5)
        self.model.fit(X)

    def predict_priority(self, task: Task) -> int:
        if self.model is None:
            self.train()
        task_vector = self.vector_memory.vectorizer.transform([task.description]).toarray()[0]
        distances = cdist([task_vector], self.model.cluster_centers_, metric='euclidean')
        closest_cluster = distances.argmin()
        return closest_cluster + 1  # Priority from 1 to 5

class SelfHealingSystem:
    def __init__(self, app: 'StudyBotApp'):
        self.app = app
        self.error_count = 0
        self.last_error_time = 0

    def check_system_health(self):
        current_time = time.time()
        if current_time - self.last_error_time > 3600:  # Reset error count every hour
            self.error_count = 0

    def handle_error(self, error: Exception):
        self.error_count += 1
        self.last_error_time = time.time()
        print(f"Error occurred: {str(error)}")

        if self.error_count > 5:
            print("Too many errors. Attempting self-healing...")
            self.perform_self_healing()

    def perform_self_healing(self):
        print("Performing self-healing...")
        # Implement self-healing strategies here, such as:
        # 1. Resetting the timer
        self.app.timer_value = 25 * 60
        self.app.timer_running = False

        # 2. Clearing any stuck animations
        self.app.animations.clear()

        # 3. Resetting the active task
        self.app.active_task = None
        self.app.task_completion = 0

        # 4. Reloading tasks from long-term memory
        self.app.load_tasks()

        print("Self-healing complete. Resuming normal operation.")
        self.error_count = 0

class StudyBotApp:
    def __init__(self):
        self.width = 100
        self.height = 50
        self.canvas = Canvas(self.width, self.height)
        self.tasks: List[Task] = []
        self.timer_running = False
        self.timer_value = 25 * 60  # 25 minutes in seconds
        self.animations: List[AnimatedValue] = []
        self.buttons: List[dict] = []
        self.active_task: Task = None
        self.task_completion = 0
        self.pomodoro_count = 0
        self.long_break_interval = 4
        self.short_break_duration = 5 * 60  # 5 minutes
        self.long_break_duration = 15 * 60  # 15 minutes
        self.is_break = False

        self.vector_memory = VectorMemory()
        self.long_term_memory = LongTermMemory()
        self.active_learner = ActiveLearner(self.vector_memory)
        self.self_healing = SelfHealingSystem(self)

    # ... (previous methods remain the same)

    def add_task(self, task: Task):
        self.tasks.append(task)
        self.vector_memory.add(task.description, task)
        self.active_learner.train()
        self.add_animation(0, 1, 0.3)  # Task added animation

    def complete_task(self):
        if self.active_task:
            self.active_task.completed = True
            self.tasks.remove(self.active_task)
            self.long_term_memory.add(f"completed_task_{int(time.time())}", asdict(self.active_task))
            self.active_task = None
            self.task_completion = 0
            self.add_animation(1, 0, 0.3)  # Task completed animation

    def select_task(self):
        if self.tasks and not self.active_task:
            # Use active learning to suggest the next task
            uncompleted_tasks = [task for task in self.tasks if not task.completed]
            if uncompleted_tasks:
                priorities = [self.active_learner.predict_priority(task) for task in uncompleted_tasks]
                self.active_task = min(zip(uncompleted_tasks, priorities), key=lambda x: x[1])[0]
                self.task_completion = 0

    def save_tasks(self):
        self.long_term_memory.add("tasks", [asdict(task) for task in self.tasks])
        self.long_term_memory.add("pomodoro_count", self.pomodoro_count)

    def load_tasks(self):
        saved_tasks = self.long_term_memory.get("tasks", [])
        self.tasks = [Task(**task_dict) for task_dict in saved_tasks]
        self.pomodoro_count = self.long_term_memory.get("pomodoro_count", 0)

        # Rebuild vector memory
        self.vector_memory = VectorMemory()
        for task in self.tasks:
            self.vector_memory.add(task.description, task)
        self.active_learner = ActiveLearner(self.vector_memory)

    def run(self):
        try:
            self.load_tasks()
        except Exception as e:
            self.self_healing.handle_error(e)
            print("Error loading tasks. Starting with an empty task list.")

        input_thread = Thread(target=self.handle_input)
        input_thread.daemon = True
        input_thread.start()

        while True:
            try:
                self.update()
                self.draw()
                self.canvas.render()
                self.self_healing.check_system_health()
                time.sleep(0.1)
            except Exception as e:
                self.self_healing.handle_error(e)

    def handle_input(self):
        while True:
            command = input("Enter command (add/complete/search/quit): ").strip().lower()
            if command == "add":
                self.add_task_input()
            elif command == "complete":
                self.complete_task_input()
            elif command == "search":
                self.search_tasks_input()
            elif command == "quit":
                self.save_tasks()
                print("Tasks saved. Exiting...")
                os._exit(0)
            else:
                print("Invalid command. Try 'add', 'complete', 'search', or 'quit'.")

    def add_task_input(self):
        name = input("Enter task name: ")
        description = input("Enter task description: ")
        estimated_time = int(input("Enter estimated time (in minutes): "))
        priority = self.active_learner.predict_priority(Task(name, description, estimated_time, 0))
        task = Task(name, description, estimated_time, priority)
        self.add_task(task)
        print(f"Task added with predicted priority: {priority}")

    def complete_task_input(self):
        if self.active_task:
            self.complete_task()
            print("Active task completed.")
        else:
            print("No active task to complete.")

    def search_tasks_input(self):
        query = input("Enter search query: ")
        results = self.vector_memory.search(query)
        print("Search results:")
        for task, similarity in results:
            print(f"- {task.name} (Similarity: {similarity:.2f})")

    def timer_expired(self):
        if self.is_break:
            print("Break time is over. Starting a new Pomodoro session.")
            self.is_break = False
            self.timer_value = 25 * 60
        else:
            self.pomodoro_count += 1
            print(f"Pomodoro {self.pomodoro_count} completed!")
            if self.pomodoro_count % self.long_break_interval == 0:
                print("Starting a long break.")
                self.is_break = True
                self.timer_value = self.long_break_duration
            else:
                print("Starting a short break.")
                self.is_break = True
                self.timer_value = self.short_break_duration

        self.timer_running = False
        self.save_tasks()  # Save progress after each Pomodoro

if __name__ == "__main__":
    app = StudyBotApp()
    app.run()