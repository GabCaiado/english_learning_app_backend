import os
import json
import argparse
from openai import OpenAI
from time import sleep
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

load_dotenv()

PROMPT_DETECTOR = """
Generate {n} examples for slang detection training.
Return JSON object with a "data" property containing an array of exactly {n} objects. Format: {{"text": "...", "is_slang": true/false, "confidence": 0.0-1.0}}

Requirements:
- {half} examples WITH slang (is_slang: true)
- {half} examples WITHOUT slang (is_slang: false)
- Include full sentences, not just single words
- Vary formality levels
- Include edge cases (words that LOOK like slang but aren't)

Examples of slang: lit, fire, bro, gonna, wanna, no cap, bussin, slay, vibe, lowkey
Examples of NOT slang: light, fire (actual fire), brother, going to, want to
"""

PROMPT_NORMALIZER = """
Generate {n} slang normalization pairs.
Return JSON object with a "data" property containing an array of exactly {n} objects. Format: {{"slang": "...", "formal": "...", "context": "..."}}

Requirements:
- Include contractions: gonna, wanna, gotta, kinda, dunno
- Include internet slang: lol, lmao, omg, brb, tbh
- Include modern slang: lit, fire, slay, bussin, no cap
- Include variations: "u" -> "you", "r" -> "are", "ur" -> "your"
- Provide full sentence context for each

IMPORTANT: The formal version must be grammatically correct standard English.
"""

PROMPT_SENTENCES = """
Generate {n} pairs of informal/formal English sentences.
Return JSON object with a "data" property containing an array of exactly {n} objects. Format: {{"informal": "...", "formal": "...", "translation_pt": "..."}}

Requirements:
- Informal should contain slang, contractions, or casual language
- Formal should be the same meaning in standard English
- Include Portuguese translation
- Vary topics: greetings, opinions, reactions, descriptions

Example:
{{"informal": "That movie was lowkey fire ngl", "formal": "That movie was surprisingly excellent, to be honest", "translation_pt": "Aquele filme foi surpreendentemente excelente, para ser honesto"}}
"""

def generate_data(client, prompt_template, total_examples, batch_size=50):
    all_data = []
    num_batches = (total_examples + batch_size - 1) // batch_size
    
    print(f"Gerando total de {total_examples} exemplos em {num_batches} batches de {batch_size}...")
    
    for i in range(num_batches):
        current_batch_size = min(batch_size, total_examples - i * batch_size)
        half_n = current_batch_size // 2
        
        prompt = prompt_template.format(n=current_batch_size, half=half_n)
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a data generation assistant. Always return valid JSON objects."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.9
            )
            
            content = response.choices[0].message.content
            
            try:
                parsed_data = json.loads(content)
                items = parsed_data.get("data", [])
                
                # Falha segura: às vezes o modelo devolve a chave principal diferente
                if not items and len(parsed_data.keys()) == 1:
                    first_key = list(parsed_data.keys())[0]
                    if isinstance(parsed_data[first_key], list):
                        items = parsed_data[first_key]
                elif not items and isinstance(parsed_data, list):
                    items = parsed_data
                    
                all_data.extend(items)
                print(f"  Batch {i+1}/{num_batches}: +{len(items)} exemplos (Total: {len(all_data)})")
            except Exception as e:
                print(f"  Erro no JSON parse: {e}")
            
            sleep(1) # Rate limit protection
            
        except Exception as e:
            print(f"  Erro na chamada OpenAI: {e}")
            sleep(5)
            
    return all_data

def process_and_save(data_list, filename_prefix):
    if not data_list:
        print(f"Nenhum dado gerado para {filename_prefix}. Ignorando.")
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
        
    print(f"-> Salvo {len(train_data)} exemplos em {train_path}")
    print(f"-> Salvo {len(test_data)} exemplos em {test_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate slang dataset with OpenAI")
    parser.add_argument("--test-mode", action="store_true", help="Gerar apenas 100 exemplos por tipo para testes.")
    args = parser.parse_args()
    
    total_examples_per_type = 100 if args.test_mode else 10000
    batch_size = 50 if total_examples_per_type >= 50 else total_examples_per_type
    
    print("="*60)
    print(f"GERACAO DE DATASET {'(TEST MODE)' if args.test_mode else '(PRODUCTION)'}")
    print(f"Alvo: {total_examples_per_type} exemplos por tarefa")
    print("="*60)
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    print("\n1. Gerando exemplos para o DETECTOR...")
    detector_data = generate_data(client, PROMPT_DETECTOR, total_examples_per_type, batch_size)
    process_and_save(detector_data, "detector")
    
    print("\n2. Gerando pares para o NORMALIZER...")
    normalizer_data = generate_data(client, PROMPT_NORMALIZER, total_examples_per_type, batch_size)
    process_and_save(normalizer_data, "normalizer")
    
    print("\n3. Gerando pares completos de SENTENCES...")
    sentences_data = generate_data(client, PROMPT_SENTENCES, total_examples_per_type, batch_size)
    process_and_save(sentences_data, "sentences")
    
    print("\n" + "="*60)
    print("GERACAO CONCLUIDA!")
    print("="*60)

if __name__ == "__main__":
    main()
