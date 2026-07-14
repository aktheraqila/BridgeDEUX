from evaluation.metrics import CometMetric

def test_comet():
    print("Initializing CometMetric...")
    metric = CometMetric()
    
    print("Running smoke test inference...")
    score = metric.compute(
        predictions=["Hello world."],
        references=[["Hello world."]],
        sources=["Hallo Welt."]
    )
    
    print(f"Success! COMET Score: {score}")

if __name__ == "__main__":
    try:
        test_comet()
    except Exception as e:
        print(f"\n[!] Smoke Test Failed: {e}")