"""
locustfile.py
Load testing script for Traffic-Net API
Usage: locust -f locustfile.py --host=http://localhost:5000
"""

from locust import HttpUser, task, between, events
import random
import time
import json
from pathlib import Path
import io
from PIL import Image
import numpy as np

class TrafficNetUser(HttpUser):
    """
    Simulates user behavior for load testing
    """
    
    # Wait between 1-3 seconds between tasks
    wait_time = between(1, 3)
    
    def on_start(self):
        """
        Called when a simulated user starts
        """
        self.test_images = self.generate_test_images(5)
        print(f"User started with {len(self.test_images)} test images")
    
    def generate_test_images(self, count=5):
        """
        Generate synthetic test images for load testing
        """
        images = []
        for i in range(count):
            # Create random image
            img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            
            # Convert to bytes
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='JPEG')
            img_bytes.seek(0)
            
            images.append(('test_image_{}.jpg'.format(i), img_bytes.getvalue()))
        
        return images
    
    @task(5)
    def get_status(self):
        """
        Task: Get model status
        Weight: 5 (highest frequency)
        """
        with self.client.get(
            "/api/status",
            catch_response=True,
            name="GET /api/status"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status check failed: {response.status_code}")
    
    @task(3)
    def predict_single_image(self):
        """
        Task: Predict single image
        Weight: 3 (medium-high frequency)
        """
        # Select random test image
        filename, img_bytes = random.choice(self.test_images)
        
        files = {
            'file': (filename, io.BytesIO(img_bytes), 'image/jpeg')
        }
        
        start_time = time.time()
        
        with self.client.post(
            "/api/predict",
            files=files,
            catch_response=True,
            name="POST /api/predict"
        ) as response:
            latency = (time.time() - start_time) * 1000  # Convert to ms
            
            if response.status_code == 200:
                try:
                    result = response.json()
                    if 'predicted_class' in result and 'confidence' in result:
                        response.success()
                        # Log latency
                        events.request.fire(
                            request_type="INFERENCE",
                            name="prediction_latency",
                            response_time=latency,
                            response_length=len(response.content),
                            exception=None,
                            context={}
                        )
                    else:
                        response.failure("Invalid response format")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Prediction failed: {response.status_code}")
    
    @task(1)
    def predict_batch(self):
        """
        Task: Predict multiple images
        Weight: 1 (lowest frequency)
        """
        # Select 2-3 random test images
        num_images = random.randint(2, 3)
        selected_images = random.sample(self.test_images, min(num_images, len(self.test_images)))
        
        files = [
            ('files', (filename, io.BytesIO(img_bytes), 'image/jpeg'))
            for filename, img_bytes in selected_images
        ]
        
        start_time = time.time()
        
        with self.client.post(
            "/api/predict/batch",
            files=files,
            catch_response=True,
            name="POST /api/predict/batch"
        ) as response:
            latency = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                try:
                    result = response.json()
                    if 'results' in result:
                        response.success()
                        # Log batch latency
                        events.request.fire(
                            request_type="BATCH_INFERENCE",
                            name=f"batch_prediction_{num_images}_images",
                            response_time=latency,
                            response_length=len(response.content),
                            exception=None,
                            context={}
                        )
                    else:
                        response.failure("Invalid batch response format")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Batch prediction failed: {response.status_code}")
    
    @task(2)
    def get_metrics(self):
        """
        Task: Get model metrics
        Weight: 2 (medium frequency)
        """
        with self.client.get(
            "/api/metrics",
            catch_response=True,
            name="GET /api/metrics"
        ) as response:
            if response.status_code in [200, 404]:
                # 404 is acceptable if metrics don't exist yet
                response.success()
            else:
                response.failure(f"Metrics request failed: {response.status_code}")
    
    @task(2)
    def get_visualizations(self):
        """
        Task: Get visualization data
        Weight: 2 (medium frequency)
        """
        with self.client.get(
            "/api/visualizations/data",
            catch_response=True,
            name="GET /api/visualizations/data"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Visualization data request failed: {response.status_code}")
    
    @task(1)
    def health_check(self):
        """
        Task: Health check endpoint
        Weight: 1 (low frequency)
        """
        with self.client.get(
            "/health",
            catch_response=True,
            name="GET /health"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")


class StressTestUser(HttpUser):
    """
    Aggressive user for stress testing
    """
    wait_time = between(0.1, 0.5)  # Faster requests
    
    def on_start(self):
        self.test_image = self.generate_single_image()
    
    def generate_single_image(self):
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        return ('stress_test.jpg', img_bytes.getvalue())
    
    @task
    def rapid_predictions(self):
        """
        Rapid-fire predictions for stress testing
        """
        filename, img_bytes = self.test_image
        files = {'file': (filename, io.BytesIO(img_bytes), 'image/jpeg')}
        
        with self.client.post("/api/predict", files=files, catch_response=True, name="STRESS /api/predict") as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Prediction failed: {response.status_code}")


# Custom events for detailed metrics
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """
    Called when the test starts
    """
    print("\n" + "="*60)
    print("LOAD TEST STARTED")
    print("="*60)
    print(f"Host: {environment.host}")
    print(f"Users: {environment.runner.user_count if hasattr(environment.runner, 'user_count') else 'N/A'}")
    print("="*60 + "\n")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """
    Called when the test stops
    """
    print("\n" + "="*60)
    print("LOAD TEST COMPLETED")
    print("="*60)
    
    if environment.stats.total.num_requests > 0:
        print(f"Total Requests: {environment.stats.total.num_requests}")
        print(f"Total Failures: {environment.stats.total.num_failures}")
        print(f"Average Response Time: {environment.stats.total.avg_response_time:.2f}ms")
        print(f"Min Response Time: {environment.stats.total.min_response_time:.2f}ms")
        print(f"Max Response Time: {environment.stats.total.max_response_time:.2f}ms")
        print(f"Requests/sec: {environment.stats.total.total_rps:.2f}")
        print(f"Failure Rate: {environment.stats.total.fail_ratio*100:.2f}%")
    
    print("="*60 + "\n")


# Example usage commands:
"""
# Basic load test with 10 users
locust -f locustfile.py --host=http://localhost:5000 --users 10 --spawn-rate 2 --run-time 2m

# Stress test with 100 users
locust -f locustfile.py --host=http://localhost:5000 --users 100 --spawn-rate 10 --run-time 5m

# Web UI mode (access at http://localhost:8089)
locust -f locustfile.py --host=http://localhost:5000

# Headless mode with CSV output
locust -f locustfile.py --host=http://localhost:5000 --users 50 --spawn-rate 5 --run-time 3m --headless --csv=results

# Test with specific user class
locust -f locustfile.py --host=http://localhost:5000 --user-classes StressTestUser --users 50 --spawn-rate 10 --run-time 2m
"""