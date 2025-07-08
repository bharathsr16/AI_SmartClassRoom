import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# It's good practice to specify the model name in a variable
MODEL_NAME = "distilbert-base-cased-distilled-squad" # Using a cased model as it can sometimes be better for QA

tokenizer = None
model = None

# --- General Knowledge Base (Simple Dictionary) ---
GENERAL_KNOWLEDGE_BASE = {
    "what is photosynthesis": "Photosynthesis is a process used by plants, algae, and some bacteria to convert light energy into chemical energy, using sunlight, water, and carbon dioxide. Oxygen is released as a byproduct.",
    "what is the capital of france": "The capital of France is Paris.",
    "who wrote romeo and juliet": "Romeo and Juliet was written by William Shakespeare.",
    "what is h2o": "H2O is the chemical formula for water, meaning it is composed of two hydrogen atoms and one oxygen atom.",
    "how many continents are there": "There are seven continents: Asia, Africa, North America, South America, Antarctica, Europe, and Australia (Oceania).",
    "what is ai": "Artificial Intelligence (AI) is the theory and development of computer systems able to perform tasks normally requiring human intelligence, such as visual perception, speech recognition, decision-making, and translation between languages."
}
DEFAULT_NO_CONTEXT_MESSAGE = "No specific context has been set yet."

def initialize_qa_model():
    """
    Initializes the tokenizer and model.
    This function should be called once before using answer_text_question.
    """
    global tokenizer, model
    if tokenizer is None or model is None:
        print(f"Loading tokenizer and model for '{MODEL_NAME}'...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)
            print("QA model and tokenizer loaded successfully.")
            return True
        except Exception as e:
            print(f"Error loading QA model/tokenizer: {e}")
            tokenizer = None
            model = None
            raise
    return True

def answer_text_question(question: str, context: str) -> str:
    """
    Answers a question. Tries contextual QA if model is available and context is meaningful.
    If not, falls back to general knowledge.
    Assumes initialize_qa_model() has been successfully called by the main application if models are to be used.
    """
    global tokenizer, model

    # This variable determines if we can even attempt to use the Hugging Face model
    can_attempt_hf_qa = model is not None and tokenizer is not None

    contextual_answer_found = False
    contextual_answer_text = "I couldn't find a specific answer in the provided text." # Default response if context search fails

    if can_attempt_hf_qa and context and context.strip() and context != DEFAULT_NO_CONTEXT_MESSAGE:
        print("(Attempting contextual QA...)")
        try:
            inputs = tokenizer(question, context, add_special_tokens=True, return_tensors="pt", truncation=True, max_length=512)
            input_ids = inputs["input_ids"].tolist()[0]

            with torch.no_grad():
                outputs = model(**inputs)

            answer_start_scores = outputs.start_logits
            answer_end_scores = outputs.end_logits

            answer_start = torch.argmax(answer_start_scores)
            answer_end = torch.argmax(answer_end_scores) + 1

            if answer_start <= answer_end:
                ans = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
                if ans.strip() and ans != tokenizer.cls_token and ans != tokenizer.sep_token and len(ans.strip()) > 2:
                    contextual_answer_text = ans.strip() # Store the actual answer
                    contextual_answer_found = True
        except Exception as e:
            print(f"Error during contextual QA: {e}. Will try General Knowledge.")
            # contextual_answer_text remains "I couldn't find..."

    if contextual_answer_found:
        return contextual_answer_text # Return the answer found from context

    # If no contextual answer, try General Knowledge Base
    normalized_question = question.strip().lower()
    if normalized_question.endswith("?"):
        normalized_question = normalized_question[:-1]

    if normalized_question in GENERAL_KNOWLEDGE_BASE:
        print(f"(Answering '{question}' from general knowledge base)")
        return GENERAL_KNOWLEDGE_BASE[normalized_question]

    # If not in GK, decide final message based on whether HF model was available and context status
    if not can_attempt_hf_qa: # Model was never loaded/initialized
        return "My apologies, my main question answering ability is currently offline, and that's not in my local knowledge."

    # Model IS available, but answer not found contextually and not in GK
    if context == DEFAULT_NO_CONTEXT_MESSAGE:
         return f"Sorry, I don't have specific information on that (no context provided), and it's not in my general knowledge base."

    # Model IS available, context was provided, but answer not found in context and not in GK
    return contextual_answer_text # This will be "I couldn't find a specific answer in the provided text."


if __name__ == "__main__":
    try:
        model_initialized_for_test = initialize_qa_model()
        if not model_initialized_for_test:
             print("Running tests with contextual QA likely disabled (if model init failed). General Knowledge should still work.")
    except Exception as e_main_init:
        model_initialized_for_test = False
        print(f"CRITICAL: Could not initialize QA model due to: {e_main_init}. Contextual QA will fail.")

    print("\n--- Testing General Knowledge ---")
    gk_q1 = "What is photosynthesis?"
    ans_gk_q1 = answer_text_question(gk_q1, DEFAULT_NO_CONTEXT_MESSAGE)
    print(f"Q: {gk_q1}\nA: {ans_gk_q1}")

    gk_q2 = "What is the capital of France?"
    ans_gk_q2 = answer_text_question(gk_q2, "Some irrelevant context about sports like football and cricket.")
    print(f"\nQ: {gk_q2} (with irrelevant context)")
    print(f"A: {ans_gk_q2}")

    gk_q_ai = "what is ai?"
    ans_gk_q_ai = answer_text_question(gk_q_ai, DEFAULT_NO_CONTEXT_MESSAGE)
    print(f"\nQ: {gk_q_ai}\nA: {ans_gk_q_ai}")

    gk_q_unknown = "What is the meaning of the universe?"
    ans_gk_q_unknown = answer_text_question(gk_q_unknown, DEFAULT_NO_CONTEXT_MESSAGE)
    print(f"\nQ: {gk_q_unknown}\nA: {ans_gk_q_unknown}")

    if model_initialized_for_test:
        print("\n--- Testing Contextual QA (should override General Knowledge if context is relevant) ---")
        photosynthesis_context_specific = "In our lecture today, photosynthesis was described as the plant's way of eating sunshine."
        q_photo_context = "What is photosynthesis?"
        # This call should use the context if the model is working
        ans_photo_context = answer_text_question(q_photo_context, photosynthesis_context_specific)
        print(f"\nContext: {photosynthesis_context_specific}")
        print(f"Q: {q_photo_context}\nA: {ans_photo_context}")

        # Test: GK question when a specific context (that doesn't contain the GK answer) is loaded
        test_context_apollo = "The Apollo 11 mission was a major achievement."
        test_question_gk_in_apollo_context = "What is H2O?"
        answer_gk_in_apollo_context = answer_text_question(test_question_gk_in_apollo_context, test_context_apollo)
        print(f"\nContext: {test_context_apollo}")
        print(f"Q_GK: {test_question_gk_in_apollo_context} (asked against Apollo context)")
        print(f"A_GK: {answer_gk_in_apollo_context}") # Should be GK H2O answer

        # Test: Contextual question that is NOT in GK
        q_apollo_specific = "What was a major achievement?"
        ans_apollo_specific = answer_text_question(q_apollo_specific, test_context_apollo)
        print(f"\nContext: {test_context_apollo}")
        print(f"Q: {q_apollo_specific}\nA: {ans_apollo_specific}")


    else:
        print("\nSkipping some contextual QA tests as model did not initialize.")

    print("\nText QA module test run complete.")
