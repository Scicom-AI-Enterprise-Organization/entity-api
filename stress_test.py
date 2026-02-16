from locust import HttpUser, task, between


class EntityAPIUser(HttpUser):
    wait_time = between(0.1, 0.5)

    # Test texts with fake placeholder entities
    test_texts = [
        "nama saya Ahmad dari Kuala Lumpur",
        "hubungi saya di 0123456789",
        "email fake@example.com untuk maklumat lanjut",
        "IC saya 900101-01-0101",
        "saya tinggal di Johor Bahru",
        "nama Ali dari Penang, hubungi 0123456789",
        "alamat saya di Sungai Petani, email test@example.com",
        "IC saya 900101-01-0101, hubungi 0123456789",
        "nama saya Ahmad dari Kuala Lumpur, IC 900101-01-0101, hubungi 0123456789",
        "contact me at user@example.com or call 0123456789",
    ]

    @task
    def predict_single(self):
        """Test predict endpoint with single text."""
        import random
        text = random.choice(self.test_texts)
        self.client.post(
            "/predict",
            json={"text": text},
            headers={"Content-Type": "application/json"},
        )
