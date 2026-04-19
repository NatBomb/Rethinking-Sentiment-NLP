import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def analyze_review(review_text):
    """
    Performs 'NLP Analysis' and 'Problem Classification' on the review. 
    Extracts sentiment, keywords, and entities.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system", 
                    "content": (
                        "You are an operational analyst for a gas station. "
                        "Analyze the review and return the result exclusively in JSON format. ALL VALUES IN THE JSON MUST BE IN ENGLISH."
                        "Extract the following fields: "
                        "sentiment (positive/neutral/negative), "
                        "category (cleanliness, customer service, queue, product shortage, infrastructure), "
                        "location (city or station number), "
                        "entities (list of services/products: Stop.Cafe, Verva, car wash, etc.), "
                        "is_ironic (boolean), "
                        "manager_action (specific corrective recommendation)."
                    )
                },
                {"role": "user", "content": review_text}
            ],
            response_format={"type": "json_object"} # JSON
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error during analysis: {e}")
        return None

def main():
    input_file = "reviews.txt"
    output_file = "results.json"
    
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} not found!")
        return

    print("Starting review analysis...")
    
    all_results = []
    
    with open(input_file, "r", encoding="utf-8") as f:
        reviews = f.readlines()

    for i, review in enumerate(reviews):
        review = review.strip()
        if not review:
            continue
            
        print(f"Processing {i+1}/{len(reviews)}...")
        analysis = analyze_review(review)
        
        if analysis:
            # Add original text for context
            analysis["original_review"] = review
            all_results.append(analysis)

    # Save results
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)

    print(f"\nAnalysis complete! Results saved to: {output_file}")
    
    # Simple summary for the 'Region Manager'
    negatives = [r for r in all_results if r['sentiment'] == 'negative']
    if negatives:
        print(f"\n🚨 ALERT: Detected {len(negatives)} negative reviews requiring attention!")

if __name__ == "__main__":
    main()






