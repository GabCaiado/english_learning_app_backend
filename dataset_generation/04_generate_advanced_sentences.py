import os
import json
import argparse
from openai import OpenAI
from time import sleep
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

load_dotenv()

PROMPT_SENTENCES = """
You are an expert linguist generating highly diverse informal English slang sentences and their formal, grammatically correct equivalents.
Generate exactly {n} pairs. 

Return a JSON object with a "data" property containing an array of exactly {n} objects. Format: {{"informal": "...", "formal": "..."}}

Guidelines:
1. DIVERSITY IS CRITICAL: Use a wide variety of internet slang, AAVE, regional slang, contractions, and modern expressions.
2. The informal sentence MUST contain slang or highly informal conversational phrasing.
3. The formal sentence MUST be standard, grammatically correct English that preserves the exact meaning.
4. Keep sentences under 25 words.
5. REQUIRED VOCABULARY: You MUST include at least one sentence using each of the following specific slang terms in this batch: {required_slangs}

Examples (Few-Shot):
[
  {{"informal": "I ain't gonna lie, that new track is an absolute bop.", "formal": "I am not going to lie, that new song is extremely catchy."}},
  {{"informal": "She was lowkey throwing shade at him during the meeting.", "formal": "She was subtly disrespecting him during the meeting."}},
  {{"informal": "We out here grinding trying to get this bread.", "formal": "We are out here working hard trying to earn money."}},
  {{"informal": "Bruh, you're doing too much rn.", "formal": "Brother, you are overreacting right now."}}
]
"""

def systematic_filter(item):
    """
    Applies quality assurance filters to prevent hallucinations and low-quality data.
    Returns True if the item is good, False if it should be discarded.
    """
    informal = item.get("informal", "").strip()
    formal = item.get("formal", "").strip()
    
    # Check for empty strings
    if not informal or not formal:
        return False
        
    # Check for identical strings (Failed normalization hallucination)
    if informal.lower() == formal.lower():
        return False
        
    # Check length constraint (Avoid overly long sentences)
    if len(informal.split()) > 30 or len(formal.split()) > 30:
        return False
        
    return True

MUST_INCLUDE_SLANGS = [
    "sucks", "nailed it", "wild", "dope", "sick", "cap", "no cap", "bet", 
    "bussin", "fire", "lit", "lowkey", "highkey", "sus", "snatched", 
    "slay", "tea", "spill the tea", "shade", "ghosted", "simp", "cringe",
    "based", "goated", "rizz", "mid", "salty", "vibes", "on god", "fr",
    "deadass", "caught in 4k", "period", "periodt", "basic"
]

def generate_data(client, total_examples, batch_size=20):
    all_data = []
    num_batches = (total_examples + batch_size - 1) // batch_size
    
    print(f"Generating {total_examples} advanced sentence pairs in {num_batches} batches of {batch_size}...")
    
    import random
    
    for i in range(num_batches):
        current_batch_size = min(batch_size, total_examples - i * batch_size)
        
        # Pick 5 random slang words that MUST be included in this batch
        batch_slangs = random.sample(MUST_INCLUDE_SLANGS, min(5, len(MUST_INCLUDE_SLANGS)))
        slangs_str = ", ".join([f'"{s}"' for s in batch_slangs])
        
        prompt = PROMPT_SENTENCES.format(n=current_batch_size, required_slangs=slangs_str)
        
        try:
            # High temperature for diversity, top_p for reasonable sampling
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a data generation assistant. Always return valid JSON objects."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=1.2,
                top_p=0.9
            )
            
            content = response.choices[0].message.content
            
            try:
                parsed_data = json.loads(content)
                items = parsed_data.get("data", [])
                
                # Fallback for weird JSON structures
                if not items and len(parsed_data.keys()) == 1:
                    first_key = list(parsed_data.keys())[0]
                    if isinstance(parsed_data[first_key], list):
                        items = parsed_data[first_key]
                elif not items and isinstance(parsed_data, list):
                    items = parsed_data
                
                # Apply Systematic Filtering
                valid_items = [item for item in items if systematic_filter(item)]
                filtered_out = len(items) - len(valid_items)
                
                all_data.extend(valid_items)
                print(f"  Batch {i+1}/{num_batches}: +{len(valid_items)} valid pairs (Filtered out {filtered_out} bad pairs). Total: {len(all_data)}")
                
            except Exception as e:
                print(f"  JSON parsing error: {e}")
            
            sleep(1) # Rate limit protection
            
        except Exception as e:
            print(f"  OpenAI API Error: {e}")
            sleep(5)
            
    return all_data

def process_and_save(data_list, filename_prefix):
    if not data_list:
        print(f"No data generated for {filename_prefix}. Skipping.")
        return
        
    os.makedirs("data", exist_ok=True)
    
    # Split train and test
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
    parser = argparse.ArgumentParser(description="Generate advanced sentence dataset with OpenAI")
    parser.add_argument("--test-mode", action="store_true", help="Generate only 50 examples for testing.")
    args = parser.parse_args()
    
    total_examples = 50 if args.test_mode else 10000
    batch_size = 25 if total_examples >= 25 else total_examples
    
    print("="*60)
    print(f"ADVANCED SYNTHETIC DATA GENERATION {'(TEST MODE)' if args.test_mode else '(PRODUCTION)'}")
    print(f"Target: {total_examples} high-quality sentence pairs")
    print("="*60)
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    sentences_data = generate_data(client, total_examples, batch_size)
    process_and_save(sentences_data, "advanced_sentences")
    
    print("\n" + "="*60)
    print("GENERATION COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()
