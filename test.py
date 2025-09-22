def ask_bc():
    # --- This setup happens only once when your application starts ---
    
    # Get the searcher class based on a type name
    searcher_class = SearcherType.get_searcher_class("bm25") # Example: using 'bm25'
    
    # You might need to create a simple object or dict for the searcher arguments
    class SearcherArgs:
        index_path = "indexes/bm25/" # Example path
    
    # 1. Create an instance of the agent
    my_agent = QueryAgent(
        searcher_class=searcher_class,
        searcher_args=SearcherArgs()
    )

    # --- Now you can use the agent anywhere, anytime ---

    # 2. Ask a question
    user_question_1 = "What is the largest mammal on Earth?"
    response_1 = my_agent.ask(user_question_1)
    print("\n--- Response 1 ---")
    print(response_1)

    # 3. Ask another question
    user_question_2 = "Who wrote the novel 'Dune'?"
    response_2 = my_agent.ask(user_question_2)
    print("\n--- Response 2 ---")
    print(response_2)

if __name__ == "__main__":
    ask_bc()