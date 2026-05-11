import os
import json
import argparse
from openai import OpenAI
from time import sleep
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

load_dotenv()

PROMPT_NEUTRAL = """
You are a linguist generating completely standard, grammatically correct English sentences.
These sentences MUST NOT contain any slang, idioms, or highly informal phrasing.
They should be ordinary sentences people use in daily life, work, travel, or casual conversation.
Generate exactly {n} pairs.

IMPORTANT: Because these are neutral sentences, the "informal" and "formal" fields MUST BE EXACTLY IDENTICAL.

Return a JSON object with a "data" property containing an array of exactly {n} objects. Format: {{"informal": "...", "formal": "..."}}

Guidelines:
1. Both "informal" and "formal" MUST be exactly the same string.
2. Ensure high diversity in topics: questions, statements, commands, different verb tenses.
3. Keep sentences under 20 words.

Examples:
[
  {{"informal": "I am going to the supermarket later today.", "formal": "I am going to the supermarket later today."}},
  {{"informal": "What time is the meeting scheduled for?", "formal": "What time is the meeting scheduled for?"}},
  {{"informal": "She decided to read a book instead of watching television.", "formal": "She decided to read a book instead of watching television."}}
]
"""

def systematic_filter(item):
    informal = item.get("informal", "").strip()
    formal = item.get("formal", "").strip()
    
    if not informal or not formal:
        return False
        
    # For neutral sentences, they MUST be identical
    if informal.lower() != formal.lower():
        return False
        
    if len(informal.split()) > 25:
        return False
        
    return True

def generate_data(client, total_examples, batch_size=20):
    all_data = []
    num_batches = (total_examples + batch_size - 1) // batch_size
    
    print(f"Generating {total_examples} NEUTRAL sentence pairs in {num_batches} batches of {batch_size}...")
    
    for i in range(num_batches):
        current_batch_size = min(batch_size, total_examples - i * batch_size)
        prompt = PROMPT_NEUTRAL.format(n=current_batch_size)
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a data generation assistant. Always return valid JSON objects."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=1.0,
            )
            
            content = response.choices[0].message.content
            
            try:
                parsed_data = json.loads(content)
                items = parsed_data.get("data", [])
                
                if not items and len(parsed_data.keys()) == 1:
                    first_key = list(parsed_data.keys())[0]
                    if isinstance(parsed_data[first_key], list):
                        items = parsed_data[first_key]
                elif not items and isinstance(parsed_data, list):
                    items = parsed_data
                
                valid_items = [item for item in items if systematic_filter(item)]
                filtered_out = len(items) - len(valid_items)
                
                all_data.extend(valid_items)
                print(f"  Batch {i+1}/{num_batches}: +{len(valid_items)} valid pairs (Filtered out {filtered_out} bad pairs). Total: {len(all_data)}")
                
            except Exception as e:
                print(f"  JSON parsing error: {e}")
            
            sleep(1)
            
        except Exception as e:
            print(f"  OpenAI API Error: {e}")
            sleep(5)
            
    return all_data

def process_and_save(data_list, filename_prefix):
    if not data_list:
        return
        
    os.makedirs("data", exist_ok=True)
    train_data, test_data = train_test_split(data_list, test_size=0.1, random_state=42)
    
    train_path = f"data/{filename_prefix}_train.json"
    test_path = f"data/{filename_prefix}_test.json"
    
    with open(train_path, "w", encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
        
    with open(test_path, "w", encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
        
    print(f"-> Saved {len(train_data)} examples to {train_path}")
    print(f"-> Saved {len(test_data)} examples to {test_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate neutral sentence dataset")
    parser.add_argument("--test-mode", action="store_true", help="Generate only 50 examples")
    args = parser.parse_args()
    
    total_examples = 50 if args.test_mode else 5000
    batch_size = 25 if total_examples >= 25 else total_examples
    
    print("="*60)
    print(f"NEUTRAL SYNTHETIC DATA GENERATION {'(TEST MODE)' if args.test_mode else '(PRODUCTION)'}")
    print(f"Target: {total_examples} pairs")
    print("="*60)
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    sentences_data = generate_data(client, total_examples, batch_size)
    process_and_save(sentences_data, "neutral_sentences")
    print("GENERATION COMPLETE!")

if __name__ == "__main__":
    main()
